import torch
from torch.distributions import (
    ComposeTransform,
    TransformedDistribution,
)
from torch.distributions.transforms import PowerTransform

from moc.metrics.calibration import latent_norm
from moc.metrics.chi import chi_cdf
from moc.models.utils import CustomTransformedDistribution

from .density_estimation import get_scaled_cdf_estimator
from .radial_transform import RadialTransform


class LatentRecalibrator(torch.nn.Module):
    def __init__(self, model, datamodule, density_estimator='kde'):
        super().__init__()
        self.model = model
        self.d = datamodule.output_dim
        self.scores = self.get_scores(model, datamodule.val_dataloader())
        self.cdf_scores = get_scaled_cdf_estimator(
            density_estimator, self.scores, datamodule, PowerTransform(1 / 3)
        )
        chi_d_cdf = chi_cdf(torch.tensor(self.d, dtype=torch.float32, device=model.device))
        r = ComposeTransform([chi_d_cdf, self.cdf_scores.inv])
        self.R = RadialTransform(r, ldj_mode='inverse')

    def predict(self, x):
        uncal_dist = self.model.predict(x)
        assert isinstance(uncal_dist, TransformedDistribution)
        batch_shape = x.shape[:-1]
        latent_dist = self.model.latent_dist().expand(batch_shape)
        return CustomTransformedDistribution(latent_dist, [self.R, *uncal_dist.transforms])

    def get_scores(self, model, dl):
        scores = []
        for x, y in dl:
            x, y = x.to(model.device), y.to(model.device)
            with torch.no_grad():
                dist = model.predict(x)
                scores.append(latent_norm(dist, y))
        return torch.cat(scores)

    def latent_dist(self):
        return self.model.latent_dist()

    @classmethod
    def output_type(cls):
        return 'distribution'

    @property
    def device(self):
        return self.model.device
