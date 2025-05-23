def test_example():
    import torch

    from moc.configs.config import get_config
    from moc.recalibration import LatentRecalibrator
    from moc.datamodules.real_datamodule import RealDataModule
    from moc.metrics.distribution_metrics import nll
    from moc.metrics.calibration import latent_calibration_error
    from moc.models.mqf2.lightning_module import MQF2LightningModule
    from moc.models.trainers.lightning_trainer import get_lightning_trainer
    from moc.utils.run_config import RunConfig

    config = get_config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rc = RunConfig(config, 'mulan', 'sf2')
    datamodule = RealDataModule(rc)
    p, q = datamodule.input_dim, datamodule.output_dim
    model = MQF2LightningModule(p, q)
    trainer = get_lightning_trainer(rc)
    trainer.fit(model, datamodule)
    model.eval()

    recalibrated_model = LatentRecalibrator(model, datamodule)
    test_batch = next(iter(datamodule.test_dataloader()))
    x, y = test_batch
    x, y = x.to(model.device), y.to(model.device)
    dist = recalibrated_model.predict(x)
    with torch.no_grad():
        nll_value = nll(dist, y).mean()
        latent_calibration = latent_calibration_error(dist, y)
    print(nll_value)
    print(latent_calibration)
