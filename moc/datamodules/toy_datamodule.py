import numpy as np
import torch
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
    Uniform,
)

from moc.datamodules.base_datamodule import BaseDataModule
from moc.utils.distributions import Degenerate
from moc.utils.general import seed_everything


# Dataset from "Nonparametric Multiple-Output Center-Outward Quantile Regression", Equation (1.7)
# There is potentially an error in the paper, as the obtained plot looks different from Fig. 2
def generate_data_original(n):
    def e_function(X):
        v = np.random.normal(0, 1, (2, X.shape[0]))
        return np.sqrt(1 + (3 / 2) * np.sin(np.pi * X / 2) ** 2) * v

    def sample_Y1(X, e1):
        return np.sin((2 * np.pi / 3) * X) + 0.575 * e1

    def sample_Y2(X, e1, e2):
        return np.cos((2 * np.pi / 3) * X) + X**2 + (e2**3 / 2.3) + (1 / 4) * e1 + 2.65 * X**4

    x = np.random.uniform(-1, 1, n)
    e = e_function(x)
    e1, e2 = e
    y1 = sample_Y1(x, e1)
    y2 = sample_Y2(x, e1, e2)
    return x, y1, y2


# Modified dataset to have a more interesting relationship between Y1 and Y2
def generate_data_modified(n):
    def e_function(x):
        v = np.random.normal(0, 1, (2, x.shape[0]))
        return np.sqrt(1 + (3 / 2) * np.sin(np.pi * x / 2) ** 2) * v

    def sample_Y1(x, e1):
        return np.sin((2 * np.pi / 3) * x) + e1

    def sample_Y2(x, e1, e2):
        return np.cos((2 * np.pi / 3) * x) + x**2 + e2 + e1 + 2.65 * x**4

    x = np.random.uniform(-1, 1, n)
    e = e_function(x)
    e1, e2 = e
    y1 = sample_Y1(x, e1)
    y2 = sample_Y2(x, e1, e2)
    x = x[:, None]
    y = np.stack([y1, y2], axis=1)
    return x, y


def generate_eq_4_1_del_barrio(n):
    def e_function(x):
        return np.random.normal(0, 1, (x.shape[0], 2))

    x = np.random.uniform(-2, 2, n)
    e = e_function(x)
    y = np.stack([x, x**2], axis=1) + (1 + 3 / 2 * np.sin(np.pi / 2 * x) ** 2)[:, None] * e
    x = x[:, None]
    return x, y


class DatasetGenerator:
    def dist_x(self):
        raise NotImplementedError

    def dist_y(self, x):
        raise NotImplementedError

    def generate(self, n):
        x = self.dist_x().sample((n,))
        y = self.dist_y(x).sample()
        return x, y


class MVNDependent(DatasetGenerator):
    def __init__(self, d=1):
        self.d = d
        self.default_size = 2500
        self.cov = self.create_cov()

    def create_cov(self):
        cov = torch.rand(self.d, self.d) * 2 - 1
        return cov @ cov.t()

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.5]), torch.tensor([2.0])), 1)

    def dist_y(self, x):
        x = x[:, 0]
        loc = torch.full((x.shape[0], self.d), 0.0, device=x.device)
        cov = self.cov[None, :, :] * x[:, None, None]
        return MultivariateNormal(loc=loc, covariance_matrix=cov)


class MVNIsotropic(DatasetGenerator):
    def __init__(self, d=1):
        self.d = d
        self.default_size = 2500

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.5]), torch.tensor([2.0])), 1)

    def dist_y(self, x):
        x = x[:, 0]
        loc = torch.full((x.shape[0], self.d), 0.0, device=x.device)
        scale = x
        scale = scale[:, None].repeat(1, self.d)
        return Independent(Normal(loc=loc, scale=scale), 1)


class MVNDiagonal(DatasetGenerator):
    def __init__(self, d=1):
        self.d = d
        self.default_size = 2500

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.5]), torch.tensor([2.0])), 1)

    def dist_y(self, x):
        x = x[:, 0]
        loc = torch.full((x.shape[0], self.d), 0.0, device=x.device)
        scale = x
        scale = scale[:, None].repeat(1, self.d)
        scale = scale * torch.arange(1, self.d + 1, device=x.device).float()
        return Independent(Normal(loc=loc, scale=scale), 1)


class MVNMixture(DatasetGenerator):
    def __init__(self, d=1, num_components=10):
        self.d = d
        self.num_components = num_components
        self.default_size = 2500
        # Randomly generate the locations for each component
        self.locs = torch.randn(self.num_components, self.d)

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.5]), torch.tensor([2.0])), 1)

    def dist_y(self, x):
        batch_size = x.shape[0]
        x = x[:, 0]
        mixture_probs = (
            torch.ones(batch_size, self.num_components) / self.num_components
        )  # Shape: [batch_size, num_components]
        mixture_distribution = Categorical(probs=mixture_probs)  # Batch shape: [batch_size]

        # Expand locs to match the batch size
        locs = self.locs[None, :, :].expand(
            batch_size, self.num_components, self.d
        )  # Shape: [batch_size, num_components, d]

        # Make scales depend on x
        scales = x[:, None, None].expand(
            batch_size, self.num_components, self.d
        )  # Shape: [batch_size, num_components, d]
        scales = scales / 5

        # Define the component distributions
        component_distribution = Normal(
            loc=locs, scale=scales
        )  # Batch shape: [batch_size, num_components, d], Event shape: []
        component_distribution = Independent(component_distribution, 1)  # Now, event shape is [d]

        # Create the MixtureSameFamily distribution
        return MixtureSameFamily(
            mixture_distribution, component_distribution
        )  # Batch shape: [batch_size], Event shape: [d]


class UnimodalHeteroscedastic(DatasetGenerator):
    def __init__(self, scale_power=1.0):
        self.scale_power = scale_power
        self.default_size = 2500

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.5]), torch.tensor([2.0])), 1)

    def dist_y(self, x):
        x = x[:, 0]
        loc = torch.full((x.shape[0], 2), 0.0, device=x.device)
        scale = x**self.scale_power
        scale = scale[:, None].repeat(1, 2)
        return Independent(Normal(loc=loc, scale=scale), 1)


class BimodalHeteroscedastic(DatasetGenerator):
    def __init__(self, scale_power=1.0):
        self.scale_power = scale_power
        self.default_size = 10000

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.5]), torch.tensor([2.0])), 1)

    def dist_y(self, x):
        x = x[:, 0]
        n = x.shape[0]
        m1 = torch.full((n, 2), 4.0, device=x.device)
        m2 = torch.full((n, 2), -4.0, device=x.device)
        loc = torch.stack([m1, m2], dim=1)
        scale = torch.stack([x**self.scale_power, (1 / x) ** self.scale_power], dim=1)
        scale = scale[:, :, None].repeat(1, 1, 2)
        return MixtureSameFamily(
            mixture_distribution=Categorical(torch.full((n, 2), 0.5, device=x.device)),
            component_distribution=Independent(Normal(loc=loc, scale=scale), 1),
        )


class Bimodal(BimodalHeteroscedastic):
    def __init__(self):
        pass

    def dist_x(self):
        return Independent(Categorical(torch.tensor([[1]])), 1)


class OneMoonHeteroscedastic(DatasetGenerator):
    def __init__(self, k=100, noise=0.2):
        self.k = k
        self.noise = noise
        self.default_size = 10000

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.0]), torch.tensor([1.0])), 1)

    def dist_y(self, x):
        batch_size = x.shape[0]
        x = x[:, 0]

        alpha = torch.linspace(0, torch.pi, self.k, device=x.device)
        locs = torch.stack([alpha.cos(), alpha.sin()], dim=-1)
        locs -= torch.tensor([0.0, 0.5], device=x.device)
        locs[:, 1] *= -1
        locs = locs[None, :, :] * (1.3 - x[:, None, None])
        scale = torch.full((self.k, 2), self.noise, device=x.device)
        scale = scale.tile(batch_size, 1, 1)
        comp_dist = Independent(Normal(locs, scale), 1)
        cat_dist = Categorical(probs=torch.ones(batch_size, self.k, device=x.device))
        return MixtureSameFamily(cat_dist, comp_dist)


class OneMoon(OneMoonHeteroscedastic):
    def __init__(self, k=100, noise=0.2):
        self.k = k
        self.noise = noise
        self.default_size = 10000

    def dist_x(self):
        # For compatibility with the rest of the code, we use a constant input
        return Independent(Degenerate(torch.tensor([0.0])), 1)


class TwoMoonsHeteroscedastic(DatasetGenerator):
    def __init__(self, k=100, noise=0.2):
        self.k = k
        self.noise = noise
        self.default_size = 2500

    def dist_x(self):
        return Independent(Uniform(torch.tensor([0.0]), torch.tensor([1.0])), 1)

    def dist_y(self, x):
        batch_size = x.shape[0]
        x = x[:, 0]

        alpha = torch.linspace(0, np.pi, self.k, device=x.device)
        locs1 = torch.stack([alpha.cos(), alpha.sin()], dim=-1)
        locs2 = torch.stack([1 - alpha.cos(), -alpha.sin()], dim=-1)
        locs2[:, 1] -= 1.5
        locs = torch.cat([locs1, locs2], dim=0)
        scale = torch.full((2 * self.k, 2), self.noise, device=x.device)
        locs = locs.tile(batch_size, 1, 1)
        scale = scale.tile(batch_size, 1, 1)
        comp_dist = Independent(Normal(locs, scale), 1)
        weight_first = (1 - x[:, None]) * 0.5
        weight_first = weight_first.repeat(1, self.k)
        probs = torch.cat([weight_first, 1 - weight_first], dim=1)
        cat_dist = Categorical(probs=probs)
        return MixtureSameFamily(cat_dist, comp_dist)


def get_distribution_generator(name):
    if name == 'unimodal_heteroscedastic':
        return UnimodalHeteroscedastic()
    if name.startswith('unimodal_heteroscedastic_power_'):
        power = float(name.split('_')[-1])
        return UnimodalHeteroscedastic(scale_power=power)
    if name == 'bimodal':
        return Bimodal()
    if name == 'bimodal_heteroscedastic':
        return BimodalHeteroscedastic()
    if name.startswith('bimodal_heteroscedastic_power_'):
        power = float(name.split('_')[-1])
        return BimodalHeteroscedastic(scale_power=power)
    if name.startswith('mvn_isotropic_'):
        d = int(name.split('_')[-1])
        return MVNIsotropic(d)
    if name.startswith('mvn_diagonal_'):
        d = int(name.split('_')[-1])
        return MVNDiagonal(d)
    if name.startswith('mvn_mixture_'):
        d, num_components = map(int, name.split('_')[-2:])
        return MVNMixture(d, num_components)
    if name.startswith('mvn_dependent_'):
        d = int(name.split('_')[-1])
        return MVNDependent(d)
    if name == 'one_moon_heteroscedastic':
        return OneMoonHeteroscedastic()
    if name.startswith('one_moon_noise_'):
        noise = float(name.split('_')[-1])
        return OneMoon(noise=noise)
    if name == 'two_moons_heteroscedastic':
        return TwoMoonsHeteroscedastic()
    return None


class ToyDataModule(BaseDataModule):
    def __init__(self, *args, size=None, **kwargs):
        self.size = size
        super().__init__(*args, **kwargs)

    def get_data(self, seed=0):
        seed_everything(seed)
        self.distribution_generator = get_distribution_generator(self.dataset)
        size = self.size
        if self.distribution_generator is not None:
            if self.size is None:
                size = self.distribution_generator.default_size
            x, y = self.distribution_generator.generate(size)
            x, y = x.numpy(), y.numpy()
            return x, y

        # Ideally, we have access to a Distribution object such that we can also measure
        # the density of the oracle distribution.
        # However, these datasets do not give access to the density.
        if self.dataset == 'toy_hallin':
            x, y = generate_data_modified(500)
        elif self.dataset == 'toy_del_barrio':
            x, y = generate_eq_4_1_del_barrio(10000)
        else:
            raise ValueError(f'Unknown dataset: {self.dataset}')
        return x, y
