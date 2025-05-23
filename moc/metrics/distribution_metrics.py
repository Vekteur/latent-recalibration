import torch


def nll(dist, y):
    return -dist.log_prob(y)


def kernel_score_from_samples(y, s1, s2, kernel):
    """
    Returns the kernel score evaluated in `y` using the samples `s1` and `s2`.
    `s1` and `s2` are tensors of shape (n_samples, b, d).
    `y` is a tensor of shape (..., b, d), where the first dimensions are arbitrary
    and will be evaluated for the same batch element.
    `kernel` is a callable that takes two broadcastable tensors of shape (..., d) and returns a tensor of shape (...,).
    """

    n_samples, b, d = s1.shape
    assert s1.shape == s2.shape
    assert y.shape[-2:] == (b, d)

    first_term = kernel(
        s1.unsqueeze(-3),
        s2.unsqueeze(-4),
    ).mean(dim=(-3, -2))

    second_term = kernel(
        s1,
        y.unsqueeze(-3),
    ).mean(dim=-2)

    return 0.5 * first_term - second_term


def Lnorm_kernel(beta):
    def kernel(y1, y2):
        return -(torch.linalg.vector_norm(y1 - y2, dim=-1) ** beta)

    return kernel


def gaussian_kernel(sigma):
    def kernel(y1, y2):
        return torch.exp(-0.5 * torch.linalg.vector_norm(y1 - y2, dim=-1) ** 2 / sigma**2)

    return kernel


def energy_score_from_samples(y, s1, s2, beta=1.0):
    return kernel_score_from_samples(y, s1, s2, Lnorm_kernel(beta))


def gaussian_kernel_score_from_samples(y, s1, s2, sigma=1.0):
    return kernel_score_from_samples(y, s1, s2, gaussian_kernel(sigma))


def sample(dist, sample_shape, rsample):
    if rsample:
        return dist.rsample(sample_shape)
    return dist.sample(sample_shape)


def energy_score(dist, y, n_samples=100, beta=1.0, rsample=False):
    s1 = sample(dist, (n_samples,), rsample)
    s2 = sample(dist, (n_samples,), rsample)
    return energy_score_from_samples(y, s1, s2, beta)


def variogram_score_from_sample(y, s, p=0.5):
    """
    Returns the variogram score evaluated in `y` using the sample `s` .
    `s` is a tensor of shape (n_samples, b, d).
    `y` is a tensor of shape (..., b, d), where the first dimensions are arbitrary
    and will be evaluated for the same batch element.
    """
    term1 = (y.unsqueeze(-1) - y.unsqueeze(-2)).abs() ** p
    term2 = ((s.unsqueeze(-1) - s.unsqueeze(-2)).abs() ** p).mean(dim=0)
    return ((term1 - term2) ** 2).mean(dim=(-2, -1))


def variogram_score(dist, y, n_samples=100, p=0.5, rsample=False):
    s = sample(dist, (n_samples,), rsample)
    return variogram_score_from_sample(y, s, p, rsample)
