import torch
from torch.distributions import ComposeTransform

from .bandwidth_cv import select_bandwidth_cv
from .smooth_empirical_cdf import SmoothEmpiricalCDF
from .spline import select_count_bins, SplineModule
from .spline_trainer import train


def get_cdf_estimator(density_estimator, scores, datamodule):
    if density_estimator == 'kde':
        b, _ = select_bandwidth_cv(scores)
        cdf_scores = SmoothEmpiricalCDF(scores, b=b)
    elif density_estimator == 'spline':
        count_bins = select_count_bins(datamodule)
        spline = SplineModule(count_bins=count_bins).to(scores.device).to(torch.float64)
        cdf_scores = train(spline, scores, val_ratio=0, train_device='cpu')
    else:
        raise ValueError(f'Unknown density estimator: {density_estimator}')
    return cdf_scores


def get_scaled_cdf_estimator(density_estimator, scores, datamodule, g):
    cdf_transformed_scores = get_cdf_estimator(density_estimator, g(scores), datamodule)
    return ComposeTransform([g, cdf_transformed_scores])
