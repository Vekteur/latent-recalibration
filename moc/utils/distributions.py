import torch
from torch.distributions import constraints, Distribution


class Degenerate(Distribution):
    arg_constraints = {'value': constraints.real}
    support = constraints.real

    def __init__(self, value, validate_args=None):
        if not isinstance(value, torch.Tensor):
            raise TypeError('`value` must be a torch.Tensor')
        self.value = value
        super().__init__(
            batch_shape=self.value.shape, event_shape=torch.Size([]), validate_args=validate_args
        )

    def sample(self, sample_shape=torch.Size()):
        return self.value.expand(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        zero = torch.zeros_like(self.value)
        neg_inf = torch.full_like(self.value, -float('inf'))
        return torch.where(value == self.value, zero, neg_inf)

    @property
    def mean(self):
        return self.value

    @property
    def mode(self):
        return self.value

    @property
    def variance(self):
        return torch.zeros_like(self.value)

    @property
    def stddev(self):
        return torch.zeros_like(self.variance)

    def entropy(self):
        return torch.zeros_like(self.variance)
