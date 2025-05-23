import logging
import pickle
import shutil
import traceback
from pathlib import Path

import yaml
from dask.distributed import as_completed, Client, get_client
from joblib import delayed, Parallel
from omegaconf import DictConfig, OmegaConf

from moc.configs.datasets import get_dataset_groups
from moc.configs.general import PrecomputationLevel
from moc.models.tuning import get_tuning
from moc.run_experiment import run
from moc.utils import configure_logging
from moc.utils.run_config import RunConfig

log = logging.getLogger(__name__)


def train_and_save(rc, hparams, pm):
    rc.hparams = hparams
    if rc.config.precomputation_level >= PrecomputationLevel.RESULTS and rc.storage_path.exists():
        # # Workaround to overwrite results with specific hyperparameters
        # if rc.hparams['model'] == 'DRF-KDE':
        #     pass # We overwrite the results
        # else:
        with rc.storage_path.open('rb') as f:
            return pickle.load(f)

    index = 0 if pm is None else pm.request().result()
    # Run training
    rcs = run(rc, index)

    if pm is not None:
        pm.free(index).result()

    rc.storage_path.parent.mkdir(parents=True, exist_ok=True)
    if rc.config.remove_checkpoints:
        assert len(list(rc.checkpoints_path.rglob('*'))) <= 2, list(rc.checkpoints_path.rglob('*'))
        # In case of an error here, check that the same runs (with same hyperparameters) are not run concurrently!
        shutil.rmtree(rc.checkpoints_path)
    with rc.storage_path.open('wb') as f:
        # Avoid saving the whole config for each run to save space.
        for rc_posthoc in rcs:
            rc_posthoc.config = None
        pickle.dump(rcs, f)
    return rc


class PositionManager:
    """
    The Position Manager allows to keep track of which process is currently running.
    It can be useful e.g. to show multiple progress bars in parallel.
    """

    def __init__(self, size):
        self.slots = [False for _ in range(size)]

    def free(self, i):
        self.slots[i] = False

    def request(self):
        for i, slot in enumerate(self.slots):
            if not slot:
                self.slots[i] = True
                return i
        log.info('No slot available')
        return 0


class ParallelRunner:
    def __init__(self, config):
        self.config = config
        assert config.manager in ['sequential', 'dask', 'joblib']
        if config.manager == 'dask':
            Client(n_workers=config.nb_workers, threads_per_worker=1, memory_limit=None)
        self.manager = config.manager
        self.tasks = []
        if self.manager == 'dask':
            pm_future = self.submit(
                PositionManager,
                config.nb_workers,
                actor=True,
            )
            self.pm = pm_future.result()
        else:
            self.pm = None

    def submit(self, fn, *args, priority=None, **kwargs):
        if self.manager == 'dask':
            return get_client().submit(fn, *args, **kwargs, priority=priority)
        if self.manager == 'joblib':
            return (fn, args, kwargs)
        return fn(*args, **kwargs)

    def train_in_parallel(self, rc, hparams, priority):
        return self.submit(
            train_and_save,
            rc,
            hparams,
            self.pm,
            priority=priority,
        )

    def run_grid_search(self, rc, priority):
        grid = get_tuning(rc.config)
        for hparams in grid:
            future_rc = self.train_in_parallel(rc, hparams, priority)
            self.tasks.append(future_rc)

    def close(self):
        if self.manager == 'dask':
            for future in as_completed(self.tasks):
                if future.status == 'error':
                    message = (
                        'Error in parallel task\n'
                        f'{"=" * 60}\n'
                        'Traceback\n'
                        f'{"=" * 60}\n'
                        f'{future.traceback()}\n'
                        f'Exception: {future.exception()}'
                    )
                    log.info(message)
        elif self.manager == 'joblib':
            Parallel(n_jobs=self.config.nb_workers)(
                delayed(self.joblib_wrapped_fn)(fn, *args, **kwargs) for fn, args, kwargs in self.tasks
            )

    def joblib_wrapped_fn(self, fn, *args, **kwargs):
        configure_logging()
        if self.config.debug:
            return fn(*args, **kwargs)
        # This function is used to catch exceptions in parallel tasks without stopping the other tasks
        try:
            return fn(*args, **kwargs)
        except Exception:
            log.error(f'{" Start of error in wrapped function ":=^80}')
            log.error('args:')
            for arg in args:
                if isinstance(arg, RunConfig):
                    log.error(arg.summary_str(bold=False))
                else:
                    log.error(arg)
            log.error('kwargs:')
            log.error(kwargs)
            log.error('Traceback:')
            log.error(traceback.format_exc())
            log.error(f'{" End of error in wrapped function ":=^80}')
            # We do not raise the exception here to avoid stopping the other tasks
            # raise e


def run_all(config: DictConfig):
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    config_repr = OmegaConf.to_container(config, resolve=True, enum_to_str=False)
    with open(Path(config.log_dir) / 'config.yaml', 'w') as f:
        f.write(yaml.dump(config_repr))

    if config.manager is None:
        config.manager = 'sequential' if config.nb_workers == 1 else 'joblib'
    runner = ParallelRunner(config)
    priority = 0
    for run_id in range(config.start_repeat_tuning, config.repeat_tuning):
        for dataset_group, datasets in get_dataset_groups(config.datasets).items():
            for dataset in datasets:
                rc = RunConfig(
                    config=config,
                    dataset_group=dataset_group,
                    dataset=dataset,
                    run_id=run_id,
                )
                runner.run_grid_search(rc, priority)
                priority -= 1
    runner.close()
