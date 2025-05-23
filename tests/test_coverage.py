import shutil

import torch

from moc.configs.config import get_config
from moc.configs.datasets import get_dataset_groups
from moc.datamodules import load_datamodule
from moc.parallel_runner import run_all
from moc.utils import configure_logging
from moc.utils.run_config import RunConfig


def _test_tuning(name, datasets='test', fast=True, **params):
    configure_logging()
    config = get_config(
        {
            'tuning_type': name,
            'name': name,
            'datasets': datasets,
            'fast': fast,
            'test': True,
            **params,
        }
    )
    if config.device == 'cuda':
        assert torch.cuda.is_available()
    run_all(config)
    shutil.rmtree(config.log_dir)


def test_lr_arflow():
    _test_tuning('lr_arflow', device='cuda', train_val_calib_test_split_ratio=[0.65, 0.2, 0, 0.15])


def test_lr_mqf2():
    _test_tuning('lr_mqf2', device='cuda', train_val_calib_test_split_ratio=[0.65, 0.2, 0, 0.15])


def test_lr_tarflow():
    _test_tuning(
        'lr_tarflow', device='cuda', datasets='afhq', default_batch_size=256, only_cheap_metrics=True
    )


def test_datasets_loading():
    config = get_config(
        {
            'datasets': 'all',
            'fast': True,
            'test': True,
        }
    )
    dataset_groups = get_dataset_groups(config.datasets)
    for dataset_group, datasets in dataset_groups.items():
        for dataset in datasets:
            rc = RunConfig(config, dataset_group, dataset)
            load_datamodule(rc)
