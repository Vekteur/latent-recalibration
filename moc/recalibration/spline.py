import torch
from torch import nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from torch.distributions.transforms import constraints, Transform

from .spline_pyro import Spline


def select_count_bins(datamodule):
    bins_thresholds = {
        30: 4,
        50: 5,
        70: 6,
        80: 7,
        90: 8,
        100: 9,
    }
    n = len(datamodule.data_val)
    dataset = datamodule.dataset
    count_bins = 3
    for threshold, bins in bins_thresholds.items():
        if n >= threshold:
            count_bins = bins
    print(dataset, n, count_bins)
    return count_bins


class DTypeTransform(Transform):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, from_dtype, to_dtype):
        super().__init__()
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype

    def _call(self, x):
        return x.to(self.to_dtype)

    def _inverse(self, y):
        return y.to(self.from_dtype)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


# We have to set Spline.sign = 1 to make it work with TransformedDistribution
class CustomSpline(Spline):
    sign = 1


class SplineModule(nn.Module):
    def __init__(self, count_bins=8):
        super().__init__()
        self.spline = CustomSpline(1, count_bins=count_bins, bound=1.0, order='quadratic')

    def forward(self):
        transforms = [
            DTypeTransform(torch.float32, torch.float64),
            # x is in R
            TanhTransform(),
            # x is in [-1, 1]
            self.spline.inv,
            # x is in [-1, 1]
            TanhTransform().inv,
            # x is in R
            DTypeTransform(torch.float64, torch.float32),
        ]
        base_dist = Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        return TransformedDistribution(base_dist, transforms)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
