import logging
from copy import copy

from moc.utils.general import elapsed_timer, seed_everything, SeedOffset

from . import recalibrators

log = logging.getLogger(__name__)


class RecalibratorModule:
    def __init__(self, model, datamodule, rc, hparams):
        log.info(f'Initializing {hparams}')
        seed_everything(rc.run_id + SeedOffset.POSTHOC_INIT)
        self.rc = rc
        self.hparams = hparams
        kwargs = hparams.copy()
        method = kwargs.pop('method')
        with elapsed_timer() as timer:
            if method is not None:
                recalibrator_cls = recalibrators[method]
                self.model = recalibrator_cls(model, datamodule, **kwargs).to(model.device)
            else:
                self.model = model
        calib_time = timer()
        self.metrics = {'calib_time': calib_time}

    def compute_metrics(self, metrics_computer):
        log.info(f'Computing metrics for {self.hparams}')
        seed_everything(self.rc.run_id + SeedOffset.POSTHOC_METRICS)
        metrics = metrics_computer.compute_metrics(self.model)
        self.metrics.update(metrics)

    def make_run_config(self):
        rc = copy(self.rc)
        hparams_with_prefix = {f'posthoc_{key}': value for key, value in self.hparams.items()}
        rc.hparams = {**rc.hparams, **hparams_with_prefix, 'method_type': 'recalibration'}
        rc.metrics = self.metrics
        return rc


class RecalibratorsManager:
    def __init__(self, model, datamodule, rc, recalibration_grid):
        self.rc = rc
        self.modules = []
        for hparams in recalibration_grid:
            self.modules.append(RecalibratorModule(model, datamodule, rc, hparams))

    def compute_metrics(self, metrics_computer):
        rcs = []
        for module in self.modules:
            module.compute_metrics(metrics_computer)
            rcs.append(module.make_run_config())
        return rcs
