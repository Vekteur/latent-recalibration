import logging
from collections import defaultdict

import torch

from moc.models.utils import CustomTransformedDistribution
from moc.utils.general import elapsed_timer

from .calibration import hpd_from_sample, latent_distance, uniform_calibration_error
from .distribution_metrics import (
    energy_score_from_samples,
    gaussian_kernel_score_from_samples,
    nll,
    variogram_score_from_sample,
)

log = logging.getLogger(__name__)


class DistributionMetricsComputer:
    def __init__(self, datamodule, only_cheap_metrics=False, n_samples_energy_score=100):
        self.datamodule = datamodule
        self.only_cheap_metrics = only_cheap_metrics
        self.n_samples_energy_score = n_samples_energy_score

    def compute_cheap_metrics_on_batch(self, x, y, model):
        # Default metric values
        nan = torch.full((x.shape[0],), torch.nan, device=x.device)

        dist = model.predict(x)
        with elapsed_timer() as timer:
            nll_value = nll(dist, y) if getattr(dist, 'has_log_prob', True) else nan
        nll_time = timer()

        with elapsed_timer() as timer:
            if isinstance(dist, CustomTransformedDistribution):
                latent_distance_value = latent_distance(dist, y)
            else:
                latent_distance_value = nan
        latent_distance_time = timer()

        metrics = {
            'nll': nll_value,
            'latent_distance': latent_distance_value,
        }
        times = {
            'nll_time': nll_time,
            'latent_distance_time': latent_distance_time,
        }
        return metrics, times

    def compute_metrics_on_batch(self, x, y, model):
        metrics, times = self.compute_cheap_metrics_on_batch(x, y, model)
        if self.only_cheap_metrics:
            return metrics, times

        dist = model.predict(x)
        with elapsed_timer() as timer:
            s = dist.sample((2 * self.n_samples_energy_score,))
        sampling_time = timer()
        times.update(
            {
                'sampling_time': sampling_time,
            }
        )
        s1, s2 = s.chunk(2, dim=0)
        # Energy score
        betas = [0.5, 1, 1.7]
        metrics.update(
            {f'energy_score_{beta}': energy_score_from_samples(y, s1, s2, beta=beta) for beta in betas}
        )
        # Gaussian kernel score
        sigmas = [0.5, 1, 2]
        metrics.update(
            {
                f'gaussian_kernel_score_{sigma}': gaussian_kernel_score_from_samples(y, s1, s2, sigma)
                for sigma in sigmas
            }
        )
        # Variogram score
        ps = [0.5, 1, 2]
        metrics.update({f'variogram_score_{p}': variogram_score_from_sample(y, s, p=p) for p in ps})
        # HPD
        hpd_value = hpd_from_sample(dist, y, s)
        metrics.update(
            {
                'hpd': hpd_value,
            }
        )
        return metrics, times

    def compute_test_metrics(self, model):
        metrics_per_batch = defaultdict(list)
        times = defaultdict(lambda: 0)
        for x, y in self.datamodule.test_dataloader():
            x, y = x.to(model.device), y.to(model.device)
            metrics_on_batch, times_on_batch = self.compute_metrics_on_batch(x, y, model)
            for name, values in metrics_on_batch.items():
                metrics_per_batch[name].append(values)
            for name, time in times_on_batch.items():
                times[name] += time
        metrics_cat = {name: torch.cat(values).float().cpu() for name, values in metrics_per_batch.items()}
        # Take the mean of the metrics
        metrics = {name: values.mean().item() for name, values in metrics_cat.items()}
        # Add time metrics
        metrics.update(times)
        # Keep non-aggregated metrics for reliability diagrams
        metrics['latent_distance'] = metrics_cat['latent_distance'].numpy()
        # Calibration error based on all samples (not per batch)
        metrics['latent_calibration'] = uniform_calibration_error(metrics_cat['latent_distance']).item()
        if not self.only_cheap_metrics:
            metrics['hpd'] = metrics_cat['hpd'].numpy()
            metrics['hdr_calibration'] = uniform_calibration_error(metrics_cat['hpd']).item()
        log.debug(f'NLL: {metrics["nll"]:.4f}')
        return metrics

    def compute_val_metrics(self, model):
        metrics_per_batch = defaultdict(list)
        for x, y in self.datamodule.val_dataloader():
            x, y = x.to(model.device), y.to(model.device)
            metrics_on_batch, _ = self.compute_cheap_metrics_on_batch(x, y, model)
            for name, values in metrics_on_batch.items():
                metrics_per_batch[name].append(values)
        metrics_cat = {name: torch.cat(values).float().cpu() for name, values in metrics_per_batch.items()}
        # Take the mean of the metrics
        metrics_mean = {name: values.mean().item() for name, values in metrics_cat.items()}
        metrics = {
            'val/nll': metrics_mean['nll'],
        }
        metrics['val/latent_calibration'] = uniform_calibration_error(metrics_cat['latent_distance']).item()
        return metrics

    def compute_metrics(self, model):
        metrics = self.compute_test_metrics(model)
        metrics.update(self.compute_val_metrics(model))
        return metrics
