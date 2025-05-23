from datetime import datetime
from enum import IntEnum
from pathlib import Path

from omegaconf import OmegaConf


class PrecomputationLevel(IntEnum):
    NONE = 0  # Everything is computed on the fly
    MODELS = 1  # Models like MQF^2 are loaded if precomputed
    POSTHOC_MODELS = 2 # Recalibration models (TODO)
    RESULTS = 3  # All results are loaded if precomputed


def get_log_dir(config):
    assert config.name not in ['fast', 'debug', 'test'], 'config.name cannot be "fast" or "debug"'
    optional_dirs = []
    for dir in ['fast', 'debug', 'test']:
        if config.get(dir):
            optional_dirs.append(dir)
    log_dir = Path(config.log_base_dir)
    if optional_dirs:
        log_dir /= '-'.join(optional_dirs)
    if config.name is not None:
        log_dir /= config.name
    else:
        log_dir /= datetime.now().strftime(r'%Y-%m-%d')
        log_dir /= datetime.now().strftime(r'%H-%M-%S')
    return log_dir


def general_config(config):
    work_dir = Path()
    default_config = OmegaConf.create(
        {
            'work_dir': str(work_dir),
            'data_dir': str(work_dir / 'data'),
            'log_base_dir': str(work_dir / 'logs'),
            # Name of the experiment
            # If no name is specified, the name will be the current date and time
            'name': None,
            'device': 'cpu',
            'train_val_calib_test_split_ratio': (0.65, 0.2, 0, 0.15),
            'default_batch_size': 512,
            'tuning_type': 'default',
            'print_config': True,
            'progress_bar': False,
            'only_cheap_metrics': False,
            # Optional path to json file with hyperparameters to use for each dataset and model
            'hparams_path': None,
            # Number of samples
            'n_samples_energy_score': 100,
            # Whether to noramlize the data to have mean 0 and std 1
            'normalize': True,
            # Whether to remove checkpoints to avoid using a large amount of disk space
            'remove_checkpoints': False,
            # Indicates which subset of datasets to select
            'datasets': 'default',
            'max_dataset_size': 50000,
            'afhq_noise': 0.07,
            # If True, the experiment will be repeated only once on a few batches of the dataset
            'fast': False,
            'debug': False,
            'test': False,
            # Which manager to use for parallelization in ['dask', 'joblib', 'sequential']
            # If None, 'sequential' is used if nb_workers=1, else 'joblib'
            'manager': None,
            'nb_workers': 1,
            # Trainer
            'max_epochs': 5000,
            'patience': 15,
            # This selects runs with run_id in the range [start_repeat_tuning, repeat_tuning)
            'start_repeat_tuning': 0,
            'repeat_tuning': 1,
            'precomputation_level': PrecomputationLevel.RESULTS,
            'selected_models': None,
        }
    )
    config = OmegaConf.merge(default_config, config)
    if config.fast:
        config.repeat_tuning = 1
    log_dir = get_log_dir(config)
    if config.name == 'unnamed' and log_dir.exists():
        raise RuntimeError('Unnamed experiment already exists')
    config.log_dir = str(log_dir)
    return config
