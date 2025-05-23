import torch
from lightning.pytorch import LightningModule
from pyro.distributions import Independent, Normal
from pyro.distributions.transforms import (
    ConditionalAffineAutoregressive,
    ConditionalSplineAutoregressive,
    Permute,
)
from pyro.nn import ConditionalAutoRegressiveNN
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from moc.models.utils import CustomConditionalTransformedDistribution


class CustomPermute(Permute):
    def _call(self, x):
        return x.index_select(self.dim, self.permutation.to(x.device))

    def _inverse(self, y):
        return y.index_select(self.dim, self.inv_permutation.to(y.device))


def get_transform(transform_type, input_dim, output_dim, hidden_size, num_layers):
    if transform_type == 'affine':
        arn = ConditionalAutoRegressiveNN(
            output_dim, input_dim, [hidden_size] * num_layers, skip_connections=True
        )
        transform = ConditionalAffineAutoregressive(arn).inv
    elif transform_type in ['spline-linear', 'spline-quadratic']:
        count_bins = 8
        if transform_type == 'spline-linear':
            order = 'linear'
            param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
        elif transform_type == 'spline-quadratic':
            order = 'quadratic'
            param_dims = [count_bins, count_bins, count_bins - 1]
        arn = ConditionalAutoRegressiveNN(
            output_dim, input_dim, [hidden_size] * num_layers, param_dims=param_dims, skip_connections=True
        )
        transform = ConditionalSplineAutoregressive(output_dim, arn, count_bins=count_bins, order=order).inv
    else:
        raise ValueError(f'Unknown transform type: {transform_type}')
    return transform


class ARFlowLightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_flows: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        transform_type: str = 'affine',
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_scheduler_patience: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.modulelist = nn.ModuleList()
        self.transforms = []

        for i in range(num_flows):
            transform = get_transform(transform_type, input_dim, output_dim, hidden_size, num_layers)
            self.transforms.append(transform)
            perm = torch.arange(output_dim - 1, -1, -1, dtype=torch.long)
            permute = CustomPermute(perm, dim=-1)
            self.transforms.append(permute)
            # Register permutation as buffer such that it moves to the correct device
            self.register_buffer(f'perm_{i}', permute.permutation)
            self.modulelist.append(transform)

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
            self.latent_dist().expand(batch_shape), list(self.transforms)
        ).condition(x)

    def forward(self, x):
        return self.predict(x)

    def compute_loss(self, dist, x, y):
        log_fy = dist.log_prob(y)
        return -log_fy.mean()

    def step(self, batch):
        x, y = batch
        dist = self(x)
        return self.compute_loss(dist, x, y)

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
        loss = self.step(batch)
        self.log(
            'val/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,  # Factor by which the learning rate will be reduced
            patience=self.hparams.lr_scheduler_patience,  # Number of epochs with no improvement
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 2,
            },
        }

    @classmethod
    def output_type(cls):
        return 'distribution'
