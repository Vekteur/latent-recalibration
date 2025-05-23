import torch
from torch.distributions import (
    Categorical,
    constraints,
    CumulativeDistributionTransform,
    Distribution,
    Gamma,
    MixtureSameFamily,
)

from .invert_vectorized import invert_increasing_function


class SmoothEmpiricalCDF(CumulativeDistributionTransform):
    def __init__(self, x, b, **kwargs):
        assert x.dim() == 1
        rate = torch.tensor(1 / b, device=x.device)
        dist = MixtureSameFamily(
            Categorical(probs=torch.ones_like(x)),
            Gamma(x * rate, rate),
        )
        super().__init__(dist, **kwargs)

    def _inverse(self, y):
        low = torch.full(y.shape, 0.0, device=y.device)
        return invert_increasing_function(self._call, y, low=low)
