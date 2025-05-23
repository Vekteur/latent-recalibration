import torch

from moc.analysis.dataframes import load_datamodule
from moc.configs.config import get_config
from moc.run_experiment import train
from moc.utils.run_config import RunConfig


def _test_transform(y, transform):
    z = transform.inv(y)
    y_reconstructed = transform(z)
    assert y_reconstructed is y
    y_reconstructed = transform(z.clone())
    assert y_reconstructed is not y
    assert y.shape == z.shape == y_reconstructed.shape
    assert torch.allclose(y, y_reconstructed, atol=1e-3)


def test_bijectivity():
    config = get_config(
        {
            'fast': True,
            'device': 'cuda',
        }
    )
    for model_name in ['MQF2', 'ARFlow']:
        rc = RunConfig(config, 'toy_2dim', 'two_moons_heteroscedastic', 0, hparams={'model': model_name})
        datamodule = load_datamodule(rc)
        module = train(rc, datamodule)

        x, y = next(iter(datamodule.train_dataloader()))
        x, y = x.to(module.device), y.to(module.device)

        transform = module.predict(x).transform
        with torch.no_grad():
            _test_transform(y, transform)
