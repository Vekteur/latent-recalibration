import logging

import torch

log = logging.getLogger(__name__)


def adjust_tensor(x, a=0.0, b=1.0, *, epsilon=1e-4):
    # We accept that, due to rounding errors, x is not in the interval up to epsilon
    mask = (a - epsilon <= x) & (x <= b + epsilon)
    assert mask.all(), (x[~mask], a, b)
    return x.clamp(a, b)


def adjust_unit_tensor(x, epsilon=1e-4):
    return adjust_tensor(x, a=0.0, b=1.0, epsilon=epsilon)


def invert_increasing_function(f, alpha, epsilon=1e-5, warn_precision=1e-3, low=None, high=None):
    """
    Invert a strictly increasing function using binary search, in a vectorized way.
    """

    alpha = adjust_unit_tensor(alpha)
    # alpha, _ = torch.broadcast_tensors(alpha, torch.zeros(dist.batch_shape))
    # Expand to the left and right until we are sure that the quantile is in the interval
    expansion_factor = 4
    if low is None:
        low = torch.full(alpha.shape, -1.0, device=alpha.device)
        while (mask := f(low) > alpha + epsilon).any():
            low[mask] *= expansion_factor
    else:
        low = low.clone()
    if high is None:
        high = torch.full(alpha.shape, 1.0, device=alpha.device)
        while (mask := f(high) < alpha - epsilon).any():
            high[mask] *= expansion_factor
    else:
        high = high.clone()
    low, high, _ = torch.broadcast_tensors(low, high, torch.zeros(alpha.shape))
    assert f(low).shape == alpha.shape

    # Binary search
    prev_precision = None
    while True:
        # To avoid "UserWarning: Use of index_put_ on expanded tensors is deprecated".
        low = low.clone()
        high = high.clone()
        precision = (high - low).max()
        # Stop if we have enough precision
        if precision < epsilon:
            break
        # Stop if we can not improve the precision anymore
        if prev_precision is not None and precision >= prev_precision:
            break
        mid = (low + high) / 2
        mask = f(mid) < alpha
        low[mask] = mid[mask]
        high[~mask] = mid[~mask]
        prev_precision = precision

    if precision > warn_precision:
        log.warning(f'Imprecise quantile computation with precision {precision}')
    return low
