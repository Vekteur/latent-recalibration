from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


@dataclass
class RunConfig:
    config: DictConfig
    dataset_group: str
    dataset: str
    run_id: int = 0
    hparams: dict = None
    metrics: dict = None
    options: dict = None

    def __post_init__(self):
        assert 0 <= self.run_id < 1000, 'run_id must be in [0, 1000) to avoid seed collisions'

    @property
    def dataset_group_config(self):
        return self.config.dataset_groups[self.dataset_group]

    @property
    def config_path(self):
        return Path(self.config.log_dir) / 'config.yaml'

    @property
    def dataset_path(self):
        return Path(self.config.log_dir) / self.dataset_group / self.dataset

    @property
    def run_path(self):
        path = self.dataset_path / self.hparams_str() / str(self.run_id)
        if self.options is not None:
            path = path / self.options_str()
        return path

    @property
    def storage_path(self):
        return self.run_path / 'run_config.pickle'

    @property
    def checkpoints_path(self):
        return self.run_path / 'checkpoints'

    def hparams_str(self, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = ['conformal_grid', 'recalibration_grid']
        hparams = self.hparams
        if hparams is None:
            return 'None'
        hparams = {key: value for key, value in hparams.items() if key not in ignore_keys}
        return ','.join(f'{key}={value}' for key, value in hparams.items())

    def options_str(self):
        return ','.join(f'{key}={value}' for key, value in self.options.items())

    def summary_str(self, bold=False):
        run_id = self.run_id
        if bold:
            run_id = f'\033[1m{self.run_id}\033[0m'
        summary_dict = {
            'dataset': self.dataset,
            'run': run_id,
            'hparams': self.hparams_str(),
        }
        if self.options is not None:
            summary_dict['options'] = self.options_str()
        return ','.join(f'{key}:{value}' for key, value in summary_dict.items())

    def to_series(self):
        return pd.Series(self.__dict__)
