from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint


class CustomLogger(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'].item()
        self.train_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs.item()
        self.val_losses.append(loss)


def get_lightning_trainer(rc):
    ckpt = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=1,  # save k best models (determined by above metric)
        save_last=False,  # save model from last epoch
        verbose=False,
        dirpath=str(rc.checkpoints_path),
        filename='epoch_{epoch:04d}',
        auto_insert_metric_name=False,
    )

    es = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=rc.config.patience,
        min_delta=0,
    )

    callbacks = [ckpt, es, CustomLogger()]

    accelerator = {
        'cpu': 'cpu',
        'cuda': 'gpu',
    }[rc.config.device]

    return Trainer(
        accelerator=accelerator,
        devices=1,
        min_epochs=1,
        max_epochs=2 if rc.config.fast else rc.config.max_epochs,
        # number of validation steps to execute at the beginning of the training
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        check_val_every_n_epoch=2,
        enable_model_summary=False,
        enable_progress_bar=rc.config.progress_bar,
        callbacks=callbacks,
        logger=False,
        gradient_clip_val=10.0,
    )
