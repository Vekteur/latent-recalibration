import logging
import tempfile

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.distributions.transforms import AffineTransform, ComposeTransform, CumulativeDistributionTransform

from moc.datamodules.utils import DimSlicedTensorDataset, split_tensor_dataset, VectorizedDataLoader

log = logging.getLogger(__name__)


class DistLightningModule(LightningModule):
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

    def forward(self):
        return self.model()

    def step(self, batch):
        (y,) = batch
        dist = self()
        return -dist.log_prob(y).mean()

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.current_epoch % 200 == 0 and batch_idx == 0:
            log.info(f'epoch: {self.current_epoch}, loss: {loss.item():.4f}')
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


def _train(model, scaler, data, val_ratio, train_device='cpu'):
    module = DistLightningModule(model)
    temp_dir = tempfile.TemporaryDirectory()
    data = scaler.inv(data)

    es_dataset = 'train' if val_ratio == 0 else 'val'

    data = data.to(train_device)
    dataset = DimSlicedTensorDataset([data])
    if es_dataset == 'val':
        train_size = int(len(dataset) * (1 - val_ratio))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = split_tensor_dataset(dataset, [train_size, val_size], shuffle=True)
        train_dl = VectorizedDataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dl = VectorizedDataLoader(val_dataset, batch_size=256, shuffle=False)
    else:
        train_dl = VectorizedDataLoader(dataset, batch_size=256, shuffle=True)

    mc = ModelCheckpoint(
        monitor=f'{es_dataset}/loss',
        mode='min',
        save_top_k=1,  # save k best models (determined by above metric)
        save_last=False,  # save model from last epoch
        verbose=False,
        dirpath=temp_dir.name,
        filename='epoch_{epoch:04d}',
        auto_insert_metric_name=False,
    )
    es = EarlyStopping(
        mode='min',
        min_delta=1e-4,
        monitor=f'{es_dataset}/loss',
        patience=50,
        check_on_train_epoch_end=es_dataset == 'train',
    )

    trainer = Trainer(
        max_epochs=-1,
        callbacks=[mc, es],
        accelerator=train_device,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        check_val_every_n_epoch=1 if es_dataset == 'val' else 1000000,
        detect_anomaly=False,
    )

    is_grad_enabled = torch.is_grad_enabled()
    trainer.fit(
        model=module, train_dataloaders=train_dl, val_dataloaders=val_dl if es_dataset == 'val' else None
    )
    torch.set_grad_enabled(is_grad_enabled)
    temp_dir.cleanup()
    log.info(f'Number of epochs: {trainer.current_epoch}')
    return trainer


def train(model, data, val_ratio, train_device='cpu'):
    data = data[:, None].to(model.device).to(model.dtype)
    scaler = AffineTransform(data.mean(), data.std())
    current_device = model.device

    # if ckpt_path.exists():
    #     log.info(f'Loading from {ckpt_path}')
    #     return DistLightningModule.load_from_checkpoint(ckpt_path)
    # log.info(f'Loading failed, checkpoint {ckpt_path} not found.')
    # log.info(f'Training {ckpt_path}')
    trainer = _train(model, scaler, data, val_ratio, train_device)
    # trainer.save_checkpoint(ckpt_path)

    model = model.to(current_device)
    dist = model()
    cdf = ComposeTransform(
        [
            scaler.inv,
            CumulativeDistributionTransform(dist),
        ]
    )
    return cdf
