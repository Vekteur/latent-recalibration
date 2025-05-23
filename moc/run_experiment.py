import logging

import torch

from moc.datamodules import load_datamodule
from moc.metrics.distribution_metrics_computer import DistributionMetricsComputer
from moc.models.train import train
from moc.recalibration.manager import RecalibratorsManager

log = logging.getLogger(__name__)


def evaluate(rc, model, datamodule, recalibration_grid=None):
    rcs = []

    if recalibration_grid is not None:
        metrics_computer = DistributionMetricsComputer(
            datamodule,
            only_cheap_metrics=rc.config.only_cheap_metrics,
            n_samples_energy_score=rc.config.n_samples_energy_score,
        )

        manager = RecalibratorsManager(
            model=model,
            datamodule=datamodule,
            rc=rc,
            recalibration_grid=recalibration_grid,
        )
        rcs.extend(manager.compute_metrics(metrics_computer))

    return rcs


def run(rc, process_index):
    log.info(f'Starting {rc.summary_str()}')
    datamodule = load_datamodule(rc)

    recalibration_grid = rc.hparams.pop('recalibration_grid', None)
    model = train(rc, datamodule)

    with torch.no_grad():
        rcs = evaluate(rc, model, datamodule, recalibration_grid)

    log.info(f'Finished {rc.summary_str()}')
    return rcs
