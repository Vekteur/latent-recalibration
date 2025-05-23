import torch

from .chi import chi_cdf
from .utils import fast_empirical_cdf, get_sample_and_log_prob


def uniform_calibration_error(c, L=1):
    c = torch.sort(c, dim=-1).values
    n = c.shape[-1]
    lin = (torch.arange(n, device=c.device) + 1) / (n + 1)
    return (c - lin).abs().pow(L).mean(dim=-1)


def hpd(dist, y, n_samples, cache={}):
    """
    Returns the highest predictive density region of y conditionally to x.
    x is a tensor of shape (b, d_x).
    y is a tensor of shape (..., b, d_y), where the first dimensions are arbitrary
    and will be evaluated for the same x.
    """
    y_sample_shape = y.shape[:-2]
    batch_size = y.shape[-2]
    samples, sample_log_prob = get_sample_and_log_prob(
        dist,
        n_samples,
        cache,
        'sample',
        'sample_log_prob',
    )
    assert samples.shape == (n_samples, batch_size, y.shape[-1])
    assert sample_log_prob.shape == (n_samples, batch_size)
    y_log_prob = cache.get('y_log_prob')
    if y_log_prob is None:
        y_log_prob = dist.log_prob(y)
    assert y_log_prob.shape == (*y_sample_shape, batch_size)
    cdf = fast_empirical_cdf(sample_log_prob, y_log_prob)
    assert cdf.shape == (*y_sample_shape, batch_size)
    return 1 - cdf


def hpd_from_sample(dist, y, sample):
    """
    Returns the highest predictive density region of y conditionally to x.
    x is a tensor of shape (b, d_x).
    y is a tensor of shape (..., b, d_y), where the first dimensions are arbitrary
    and will be evaluated for the same x.
    """
    sample_log_prob = dist.log_prob(sample)
    y_log_prob = dist.log_prob(y)
    cdf = fast_empirical_cdf(sample_log_prob, y_log_prob)
    return 1 - cdf


def hdr_calibration_error(dist, y, n_samples=100, L=1, cache={}):
    c = hpd(dist, y, n_samples=n_samples, cache=cache)
    return uniform_calibration_error(c, L)


def latent_norm(dist, y, cache={}):
    y_latent = cache.get('y_latent')
    if y_latent is None:
        y_latent = dist.transform.inv(y)
    return torch.linalg.norm(y_latent, dim=-1)


def latent_distance(dist, y, cache={}):
    d = torch.tensor(y.shape[-1], dtype=torch.float32, device=y.device)
    chi_d_cdf = chi_cdf(d)
    return chi_d_cdf(latent_norm(dist, y, cache=cache))


def latent_calibration_error(dist, y, L=1, cache={}):
    c = latent_distance(dist, y, cache=cache)
    return uniform_calibration_error(c, L)
