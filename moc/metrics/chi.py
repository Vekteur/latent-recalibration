import torch
from scipy.stats import chi2
from torch.distributions import Chi2
from torch.distributions.transforms import ComposeTransform, CumulativeDistributionTransform, PowerTransform


class CustomChi2(Chi2):
    def icdf(self, y):
        device = y.device
        epsilon = 1e-6
        y = y.clamp(epsilon, 1.0 - epsilon)
        res = chi2(self.df.item()).ppf(y.cpu().numpy())
        return torch.tensor(res, dtype=torch.float32, device=device)


def chi_cdf(df):
    return ComposeTransform(
        [
            PowerTransform(2),
            CumulativeDistributionTransform(CustomChi2(df)),
        ]
    )
