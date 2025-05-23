from pyro.distributions import ConditionalTransform, constraints
from pyro.distributions.transforms import Transform


class ConditionalMQF2Transform(ConditionalTransform):
    def __init__(self, flow, reverse=False):
        self.flow = flow
        self.reverse = reverse

    def condition(self, x):
        transform = MQF2Transform(self.flow, x)
        if self.reverse:
            transform = transform.inv
        return transform


class MQF2Transform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    eps = 1e-6

    def __init__(self, flow, x):
        super().__init__(cache_size=1)
        self.flow = flow
        self.x = x
        self._cached_forward_logdet = None

    def _call(self, z):
        """
        `z` is of shape (..., batch_size, d)
        `self.x` is of shape (batch_size, p)
        """
        batch_shape = self.x.shape[:-1]
        sample_shape = z.shape[: -(len(batch_shape) + 1)]
        # We repeat `self.x` to match the shape of `z`
        x_repeat = self.x.view((1,) * len(sample_shape) + self.x.shape).expand(sample_shape + self.x.shape)
        # We flatten `z` and `x_repeat` to pass them to the flow
        z_flat, x_repeat_flat = (
            z.reshape(-1, z.shape[-1]),
            x_repeat.reshape(-1, x_repeat.shape[-1]),
        )

        z_flat, logdet_flat = self.flow.forward_transform(z_flat, context=x_repeat_flat)

        # We reshape back to the original shape
        logdet = logdet_flat.reshape(sample_shape + batch_shape)
        self._cached_forward_logdet = logdet
        return z_flat.reshape(sample_shape + batch_shape + (self.flow.dim,))

    def _inverse(self, z):
        """
        `z` is of shape (..., batch_size, d)
        `self.x` is of shape (batch_size, p)
        """
        batch_shape = self.x.shape[:-1]
        sample_shape = z.shape[: -(len(batch_shape) + 1)]
        # We repeat `self.x` to match the shape of `z`
        x_repeat = self.x.view((1,) * len(sample_shape) + self.x.shape).expand(sample_shape + self.x.shape)
        # We flatten `z` and `x_repeat` to pass them to the flow
        z_flat, x_repeat_flat = (
            z.reshape(-1, z.shape[-1]),
            x_repeat.reshape(-1, x_repeat.shape[-1]),
        )

        z_flat = self.flow.reverse(z_flat, context=x_repeat_flat)

        self._cached_forward_logdet = None
        # We reshape back to the original shape
        return z_flat.reshape(sample_shape + batch_shape + (self.flow.dim,))

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if x is x_old and y is y_old and self._cached_forward_logdet is not None:
            return self._cached_forward_logdet

        self._call(x)
        return self._cached_forward_logdet
