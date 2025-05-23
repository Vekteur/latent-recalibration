from pyro.distributions import ConditionalTransformedDistribution
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import ComposeTransform
from torch.distributions.utils import _sum_rightmost


class CustomTransformedDistribution(TransformedDistribution):
    @property
    def transform(self):
        return ComposeTransform(self.transforms)

    def log_prob_intermediate_values(self, value):
        if self._validate_args:
            self._validate_sample(value)
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim
            log_prob = log_prob - _sum_rightmost(
                transform.log_abs_det_jacobian(x, y),
                event_dim - transform.domain.event_dim,
            )
            y = x

        base_dist_log_prob = _sum_rightmost(
            self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
        )
        log_prob = log_prob + base_dist_log_prob
        return log_prob, base_dist_log_prob


class CustomConditionalTransformedDistribution(ConditionalTransformedDistribution):
    def condition(self, context):
        base_dist = self.base_dist.condition(context)
        transforms = [t.condition(context) for t in self.transforms]
        return CustomTransformedDistribution(base_dist, transforms)
