import torch
from lightning.pytorch import LightningModule
from torch.distributions import Independent, Normal

from moc.metrics.distribution_metrics import energy_score, nll
from moc.models.utils import CustomConditionalTransformedDistribution

from .conditional_transform import ConditionalMQF2Transform
from .model import get_flow


class MQF2LightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        icnn_hidden_size: int = 40,
        icnn_num_layers: int = 2,
        loss: str = 'nll',
        es_num_samples: int = 50,
        estimate_logdet: bool = False,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.flow = get_flow(
            input_dim=input_dim,
            output_dim=output_dim,
            icnn_hidden_size=icnn_hidden_size,
            icnn_num_layers=icnn_num_layers,
            is_energy_score=loss == 'es',
            estimate_logdet=estimate_logdet,
        )

        self.cond_transform = ConditionalMQF2Transform(self.flow, reverse=loss != 'es')

    def latent_dist(self):
        return Independent(
            Normal(torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device)).expand(
                [self.hparams.output_dim]
            ),
            1,
        )

    def predict(self, x):
        batch_shape = x.shape[:-1]
        return CustomConditionalTransformedDistribution(
            self.latent_dist().expand(batch_shape), [self.cond_transform]
        ).condition(x)

    def forward(self, x):
        return self.predict(x)

    def compute_loss(self, dist, y):
        if self.hparams.loss == 'es':
            return energy_score(dist, y, n_samples=self.hparams.es_num_samples, rsample=True).mean()
        if self.hparams.loss == 'nll':
            return nll(dist, y).mean()
        raise ValueError(f'Unknown loss: {self.hparams.loss}')

    def step(self, batch):
        x, y = batch
        dist = self(x)
        return self.compute_loss(dist, y)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(
            'train/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.flow.set_estimate_log_det(False)
        loss = self.step(batch)
        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.flow.set_estimate_log_det(self.hparams.estimate_logdet)
        return loss

    def on_train_end(self):
        self.flow.set_estimate_log_det(False)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    @classmethod
    def output_type(cls):
        return 'distribution'
