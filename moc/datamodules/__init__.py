from moc.configs.datasets import real_dataset_groups, toy_dataset_groups

from .afhq_datamodule import AFHQDataModule
from .real_datamodule import RealDataModule
from .toy_datamodule import ToyDataModule


def get_datamodule(group):
    if group in toy_dataset_groups:
        return ToyDataModule
    if group in real_dataset_groups:
        return RealDataModule
    if group == 'afhq':
        return AFHQDataModule
    raise ValueError(f'Unknown datamodule {group}')


def load_datamodule(rc):
    datamodule_cls = get_datamodule(rc.dataset_group)
    return datamodule_cls(rc)
