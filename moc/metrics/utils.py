import torch


def get_sample(dist, n_samples, cache={}, sample_key=None):
    """
    Helper function to get samples from the model or the cache if available.
    """
    samples = cache.get(sample_key)
    return dist.sample((n_samples,)) if samples is None else samples[:n_samples]


def get_sample_and_log_prob(dist, n_samples, cache={}, sample_key=None, log_prob_key=None):
    """
    Helper function to get samples with their log probabilities from the model or the cache if available.
    """
    sample = cache.get(sample_key)
    log_prob = cache.get(log_prob_key)
    if sample is None or log_prob is None:
        sample = dist.sample((n_samples,))
        log_prob = dist.log_prob(sample)
    else:
        if sample.shape[0] < n_samples:
            raise RuntimeError('Not enough samples in cache. Consider increasing n_samples_cache.')
        sample = sample[:n_samples]
        log_prob = log_prob[:n_samples]
    return sample, log_prob


def fast_empirical_cdf(a, b):
    """
    Returns the empirical CDF of a at b.

    Parameters:
    a: Tensor of shape (s, n) where s is the number of samples per element in batch
    b: Tensor of shape (..., n) where n is the batch size

    Returns:
    Tensor of shape (..., n) representing the empirical CDF values
    """
    assert a.dim() == 2
    assert b.dim() >= 1
    assert a.shape[-1] == b.shape[-1]
    b_shape = b.shape
    # We move n to the first dimension.
    # This will be useful because searchsorted has to be applied on the last dimension of a.
    a = a.movedim(-1, 0)
    b = b.movedim(-1, 0)

    if b.dim() == 1:
        # The naive implementation is faster for this case.
        cdf = (a <= b[:, None]).float().mean(dim=-1)
    else:
        a_sorted = torch.sort(a, dim=1)[0]
        # These operations are needed because the first N - 1 dimensions of a and b
        # have to be the same.
        view = (a.shape[0],) + (1,) * len(b.shape[1:-1]) + (a.shape[1],)
        repeat = (1,) + b.shape[1:-1] + (1,)
        a_broadcast = a_sorted.view(*view).repeat(*repeat)
        cdf = torch.searchsorted(a_broadcast, b.contiguous(), side='right') / a.shape[-1]

    # We move n back to the last dimension.
    cdf = cdf.movedim(0, -1)
    assert cdf.shape == b_shape
    return cdf
