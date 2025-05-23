import torch
from torch.distributions import constraints, Transform


class RadialTransform(Transform):
    """
    Radial transform T: R^d -> R^d given by
        T(z) = (t(||z||) / ||z||) * z
    where `t_transform` is a PyTorch Transform from R->R
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, r_transform, dim=-1, ldj_mode='forward', **kwargs):
        """
        Args:
            t_transform (Transform):
                A 1D transform representing t(r). Must implement:
                  - forward(r), inv(s), log_abs_det_jacobian(r, t(r))
                  - Inputs and outputs are always of shape (..., 1)
            dim (int):
                Which dimension is the 'features' dimension. Usually -1 for the last.
        """
        super().__init__(**kwargs)
        self.r_transform = r_transform
        self.dim = dim
        self.ldj_mode = ldj_mode
        if ldj_mode not in ['forward', 'inverse']:
            raise ValueError("ldj_mode must be 'forward' or 'inverse'.")

    def _helper(self, x, r):
        t = x.norm(dim=self.dim, keepdim=True)
        r_t = r(t)
        assert r_t.isnan().sum() == 0, 'NaN in transform'
        t_safe = torch.where(t == 0.0, torch.ones_like(t), t)
        return (r_t / t_safe) * x

    def _call(self, x):
        return self._helper(x, self.r_transform)

    def _inverse(self, y):
        return self._helper(y, self.r_transform.inv)

    def log_abs_det_jacobian(self, x, y):
        """
        Depending on ldj_mode:
          - forward:  use t=||x||, s=r(t), log_det = (d-1)(log s - log t) + log r'(t)
          - inverse:  use s=||y||, r=r^-1(s), log_det = same numeric result,
                      but *obtained* via the inverse's derivative if that is easier.
        """
        d = x.size(self.dim)

        if self.ldj_mode == 'forward':
            t = x.norm(dim=self.dim, keepdim=True)
            s = self.r_transform(t)
            assert s.isnan().sum() == 0, 'NaN in forward transform'
            log_rprime_t = self.r_transform.log_abs_det_jacobian(t, s)
        elif self.ldj_mode == 'inverse':
            s = y.norm(dim=self.dim, keepdim=True)
            t = self.r_transform.inv(s)
            assert t.isnan().sum() == 0, 'NaN in inverse transform'
            log_rprime_t = -self.r_transform.inv.log_abs_det_jacobian(s, t)

        # Avoid log(0) by using a small epsilon
        eps = 1e-8
        t = torch.clamp(t, min=eps)
        s = torch.clamp(s, min=eps)

        ldj = (d - 1) * (torch.log(s) - torch.log(t)) + log_rprime_t
        return ldj[..., 0]
