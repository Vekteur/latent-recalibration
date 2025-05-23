from copy import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import TransformedDistribution

from moc.analysis.dataframes import get_datasets_df
from moc.analysis.plot_calibration import plot_agg_runs, plot_consistency_bands
from moc.datamodules import load_datamodule
from moc.metrics.calibration import latent_distance
from moc.metrics.chi import chi_cdf
from moc.metrics.distribution_metrics_computer import DistributionMetricsComputer
from moc.models.train import train
from moc.recalibration.hdr_recalibrator import HDRRecalibratedDistribution, HDRRecalibrator
from moc.recalibration.latent_recalibrator import LatentRecalibrator
from moc.utils.run_config import RunConfig

from .utils import generate_grid

levels = [0.01, 0.1, 0.5, 0.9]
linestyles = ['dotted', 'dashdot', 'dashed', 'solid']


def plot_density_3d(dist, ax, xlim, ylim, grid_size=300):
    device = dist.sample().device
    # Define the grid
    x = torch.linspace(*xlim, grid_size, device=device)
    y = torch.linspace(*ylim, grid_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Flatten and create tensor for dist.log_prob
    xy = torch.stack([X.ravel(), Y.ravel()], axis=-1)

    # Calculate density
    with torch.no_grad():
        density = dist.log_prob(xy).exp()

    # Reshape to match the grid
    Z = density.reshape(grid_size, grid_size)

    X, Y, Z = X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy()
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')


def plot_density_2d(dist, ax, xlim, ylim, zlim, grid_size=300):
    device = getattr(dist, 'device', None)
    if device is None:
        device = dist.sample().device
    # Define the grid
    y_plot = generate_grid(grid_size, xlim, ylim, 1, device=device)

    # Calculate density
    with torch.no_grad():
        density = dist.log_prob(y_plot).exp()

    # Reshape to match the grid
    Z = density.reshape(grid_size, grid_size).swapaxes(0, 1)

    Z = Z.cpu().numpy()
    colors_list = [(1, 1, 1, 0), (1, 0.6, 0.15, 1)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom', colors_list, N=1000)
    return ax.imshow(
        Z,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        interpolation='bilinear',
        origin='lower',
        cmap=cmap,
        aspect='auto',
        norm=mpl.colors.LogNorm(vmin=zlim[0], vmax=zlim[1]),
    )


def plot_ecdf(x, axis, label='ecdf'):
    q = np.sort(x)
    q = np.append(q, q[-1])
    axis.plot(q, np.linspace(0, 1, len(x) + 1), drawstyle='steps-pre', label=label)


def add_box(ax, text):
    props = {'facecolor': 'wheat', 'alpha': 0.5}
    ax.text(
        0.95,
        0.05,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=props,
    )


class LatentRecalibratorExplicit(torch.nn.Module):
    def __init__(self, lr_model):
        super().__init__()
        self.lr_model = lr_model

    def predict(self, x):
        return self.lr_model.model.predict(x)

    def latent_dist(self):
        return TransformedDistribution(
            self.lr_model.model.latent_dist(),
            [self.lr_model.R],
        )

    @classmethod
    def output_type(cls):
        return 'distribution'

    @property
    def device(self):
        return self.lr_model.device


def plot_latent_calibration_contours(axis, dist, y, alphas, **kwargs):
    with torch.no_grad():
        ld = latent_distance(dist, y)
    ld = ld[:, :, 0]
    Y1, Y2 = y[:, :, 0, 0], y[:, :, 0, 1]
    Y1, Y2, ld = Y1.cpu().numpy(), Y2.cpu().numpy(), ld.cpu().numpy()
    assert Y1.shape == Y2.shape == ld.shape
    return axis.contour(Y1, Y2, ld, levels=alphas, **kwargs)


def gaussian_latent_calibration(z):
    norm = torch.linalg.norm(z, dim=-1)
    d = torch.tensor(z.shape[-1], dtype=torch.float32, device=z.device)
    chi_d_cdf = chi_cdf(d)
    return chi_d_cdf(norm)


def plot_gaussian_contours(axis, model, z, alphas, **kwargs):
    Y1, Y2 = z[:, :, 0, 0], z[:, :, 0, 1]

    R = getattr(model, 'R', None)
    if R is not None:
        with torch.no_grad():
            z = R.inv(z)
    with torch.no_grad():
        ld = gaussian_latent_calibration(z)
    ld = ld[:, :, 0]
    Y1, Y2, ld = Y1.cpu().numpy(), Y2.cpu().numpy(), ld.cpu().numpy()
    assert Y1.shape == Y2.shape
    return axis.contour(Y1, Y2, ld, levels=alphas, **kwargs)


def plot_visualization_per_model(ax, model, datamodule, add_title=False, add_xlabel=False, fast=False):
    def add(ax, xlabel, ylabel, title):
        if add_xlabel:
            ax.set(xlabel=xlabel)
        ax.set(ylabel=ylabel)
        if add_title:
            ax.set(title=title)

    xlim, ylim, zlim = (-3, 3), (-3, 3), (0.001, 0.2)

    # Get model with explicitly recalibrated latent distribution
    model_explicit = LatentRecalibratorExplicit(model) if isinstance(model, LatentRecalibrator) else model
    has_latent = not isinstance(model, HDRRecalibrator)

    # Get unconditional distribution
    x = torch.zeros((1,), device=model.device)
    dist = model.predict(x[None, :])

    # Get dataset points
    _, y_calib = datamodule.data_calib[:200]
    if has_latent:
        with torch.no_grad():
            z_calib = model_explicit.predict(x[None, :]).transform.inv(y_calib.to(model.device)).cpu()
        # Contours options
        kwargs = {'alphas': levels, 'linewidths': 0.5, 'linestyles': linestyles, 'colors': 'k'}

        plot_density_2d(model_explicit.latent_dist(), ax[0], xlim=xlim, ylim=ylim, zlim=zlim)
        add(ax[0], '$Z_1$', '$Z_2$', 'Latent distribution')
        ax[0].set_ylabel(r'$Z_2$', labelpad=-3)
        ax[0].scatter(z_calib[:, 0], z_calib[:, 1], s=1, c='tab:blue', alpha=0.5)
        z_plot = generate_grid(300, xlim, ylim, 1, device=model.device)
        plot_gaussian_contours(ax[0], model, z_plot, **kwargs)

    # Plot predictive distribution
    plot_density_2d(dist, ax[1], xlim=xlim, ylim=ylim, zlim=zlim)
    add(ax[1], '$Y_1$', '$Y_2$', 'Predictive distribution')
    ax[1].set_ylabel(r'$Y_2$', labelpad=-3)
    ax[1].scatter(y_calib[:, 0], y_calib[:, 1], s=1, c='tab:blue', alpha=0.5)

    if not has_latent:
        with torch.no_grad():
            samples = model.predict(x[None, :]).sample((1000,))
        ax[1].scatter(samples[:, 0, 0].cpu(), samples[:, 0, 1].cpu(), s=1, c='tab:purple', alpha=0.1)

    # Plot contours
    if has_latent:
        y_plot = generate_grid(300, xlim, ylim, 1, device=model.device)
        plot_latent_calibration_contours(ax[1], dist, y_plot, **kwargs)
    # Add metrics
    datamodule = copy(datamodule)
    datamodule.data_test = datamodule.subsample(datamodule.data_test, 10 if fast else 1000)
    with torch.no_grad():
        metrics = DistributionMetricsComputer(datamodule).compute_metrics(model)
    add_box(ax[1], f'NLL: {metrics["nll"]:.2f}, ES: {metrics["energy_score_1"]:.2f}')

    # Plot latent calibration
    def plot_reliability(ax, data):
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
        plot_agg_runs(ax, data[None])
        alpha = torch.linspace(0, 1, 1000)
        plot_consistency_bands(ax, len(data), alpha, 0.9)
        ax.set(xlim=(0, 1), ylim=(0, 1))

    plot_reliability(ax[2], metrics['latent_distance'])
    add(ax[2], xlabel=r'$\alpha$', ylabel=r'$\hat{F}_U(\alpha)$', title='Latent calibration')
    add_box(ax[2], f'L-ECE: {metrics["latent_calibration"]:.2f}')

    # Plot HDR calibration
    plot_reliability(ax[3], metrics['hpd'])
    add(ax[3], xlabel=r'$\alpha$', ylabel=r'$\hat{F}_U(\alpha)$', title='HDR calibration')
    add_box(ax[3], f'HDR-ECE: {metrics["hdr_calibration"]:.2f}')


def plot_visualization(models, datamodule, fast=False):
    nrows, ncols = len(models), 4
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2), sharex='col', sharey='col')
    for i in range(nrows):
        kwargs = {}
        if i == 0:
            kwargs['add_title'] = True
        if i == nrows - 1:
            kwargs['add_xlabel'] = True
        plot_visualization_per_model(ax[i], models[i], datamodule, fast=fast, **kwargs)
        label_x_pos = -0.28
        axis = ax[i, 0]
        text = 'Uncalibrated' if i == 0 else 'Calibrated'
        axis.text(
            label_x_pos,
            0.5,
            text,
            transform=axis.transAxes,
            rotation=90,
            va='center',
            ha='right',
            fontsize=16,
        )
    fig.tight_layout(w_pad=-0.5)
    return fig


def get_lims(log_probs_list, samples_list):
    samples = torch.cat(samples_list, dim=0)[:, 0, :].cpu()
    log_probs = torch.cat(log_probs_list, dim=0)[:, 0].cpu()
    xlim = (samples[:, 0].min().item(), samples[:, 0].max().item())
    ylim = (samples[:, 1].min().item(), samples[:, 1].max().item())
    zlim = (log_probs.min().item(), log_probs.max().item())
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    zrange = zlim[1] - zlim[0]
    xlim = (xlim[0] - 0.1 * xrange, xlim[1] + 0.1 * xrange)
    ylim = (ylim[0] - 0.35 * yrange, ylim[1] + 0.1 * yrange)
    zlim = (zlim[0] - 0.3 * zrange, zlim[1] + 0.05 * zrange)
    zlim = (np.exp(zlim[0]), np.exp(zlim[1]))
    return xlim, ylim, zlim


def plot_densities_real_datasets(config, datasets_2d, repeats=3, run_id=0):
    recalibrators = [lambda model, _datamodule: model, LatentRecalibrator]
    nrows, ncols = len(datasets_2d), len(recalibrators) * repeats
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(2.9 * ncols, 2.5 * nrows), squeeze=False, constrained_layout=True
    )

    df_ds = get_datasets_df(config)
    for row, dataset in enumerate(datasets_2d):
        dataset_group, abb = (
            df_ds.reset_index().query('dataset == @dataset').iloc[0][['dataset_group', 'abb']]
        )
        rc = RunConfig(config, dataset_group, dataset, run_id, hparams={'model': 'MQF2'})
        datamodule = load_datamodule(rc)
        base_model = train(rc, datamodule)
        models = [recalibrator(base_model, datamodule) for recalibrator in recalibrators]

        for repeat in range(repeats):
            x, y = datamodule.data_test[repeat]
            x, y = x[None, :].to(config.device), y[None, :].to(config.device)
            shift = repeat * len(recalibrators)
            ax_group = ax[row, shift : shift + 3]
            dists = [model.predict(x) for model in models]
            with torch.no_grad():
                samples_list = [dist.sample((100,)) for dist in dists]
                log_probs_list = [dist.log_prob(sample) for dist, sample in zip(dists, samples_list)]
                xlim, ylim, zlim = get_lims(log_probs_list, [*samples_list, y.unsqueeze(0)])
                for axis, dist, _sample in zip(ax_group, dists, samples_list):
                    axis.scatter(y[0, 0].cpu(), y[0, 1].cpu(), s=5, c='tab:blue')
                    im = plot_density_2d(dist, axis, xlim=xlim, ylim=ylim, zlim=zlim, grid_size=300)
                    if not isinstance(dist, HDRRecalibratedDistribution):
                        y_plot = generate_grid(300, xlim, ylim, 1, device=config.device)
                        plot_latent_calibration_contours(
                            axis, dist, y_plot, levels, linewidths=0.5, linestyles=linestyles, colors='k'
                        )
                    # axis.scatter(sample[:, 0, 0].cpu(), sample[:, 0, 1].cpu(), s=1, c='tab:green', alpha=0.2)
                add_box(ax_group[0], rf'$-\log \hat{{f}}(y | x)$: {-dists[0].log_prob(y).item():.3g}')
                add_box(ax_group[1], rf"$-\log \hat{{f}}'(y | x)$: {-dists[1].log_prob(y).item():.3g}")
            ax_group[0].set_title(r'$\hat{f}$', fontsize=14)
            ax_group[1].set_title(r"$\hat{f}'$", fontsize=14)
            cbar = fig.colorbar(im, ax=ax_group, shrink=0.9, aspect=30, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

    for row in range(nrows):
        ax[row, 0].set_ylabel('$Y_2$', fontsize=12)
    for col in range(ncols):
        ax[-1, col].set_xlabel('$Y_1$', fontsize=12)

    for row, _dataset in enumerate(datasets_2d):
        (abb,) = df_ds.reset_index().query('dataset == @dataset').iloc[0][['abb']]
        label_x_pos = -0.35
        axis = ax[row, 0]
        axis.text(
            label_x_pos, 0.5, abb, transform=axis.transAxes, rotation=90, va='center', ha='right', fontsize=16
        )


def plot_cdf_estimation(
    config, axis, dataset_group, dataset, xlim=None, plot_density=False, run_id=2, **kwargs
):
    model_name = 'MQF2'
    rc = RunConfig(config, dataset_group, dataset, run_id, hparams={'model': model_name})
    datamodule = load_datamodule(rc)
    model = train(rc, datamodule)

    lr_model = LatentRecalibrator(model, datamodule, **kwargs)
    plot_ecdf(lr_model.scores.cpu(), axis, label='Empirical CDF')

    if xlim is None:
        xlim = (lr_model.scores.min().cpu(), lr_model.scores.max().cpu())
    x = torch.linspace(*xlim, 300, device=config.device)
    with torch.no_grad():
        y = lr_model.cdf_scores(x[:, None])
    axis.plot(x.cpu(), y.cpu(), label='Smooth CDF')

    if plot_density:
        with torch.no_grad():
            log_density = lr_model.cdf_scores.log_abs_det_jacobian(x[:, None], y)
        axis_twin = axis.twinx()
        axis_twin.plot(x.cpu(), log_density.cpu(), label='Log density', color='red', alpha=0.5)

    l = xlim[1] - xlim[0]
    axis.set_xlim((xlim[0] - 0.01 * l, xlim[1] + 0.01 * l))
    axis.set_ylim((0, 1))

    return axis_twin if plot_density else None


def plot_density_estimation(config, **kwargs):
    datasets_iter = list(
        get_datasets_df(config, reload=False)
        .sort_values('n')
        .reset_index()[['dataset_group', 'dataset', 'abb']]
        .iterrows()
    )

    size = len(datasets_iter)
    ncols = 5
    nrows = int(np.ceil(size / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2), sharey=True, squeeze=False)
    ax_flat = ax.flatten()
    ax_twin_flat = []

    for i, (dataset_group, dataset, abb) in datasets_iter:
        axis_twin = plot_cdf_estimation(
            config, ax_flat[i], dataset_group, dataset, plot_density=True, **kwargs
        )
        ax_twin_flat.append(axis_twin)
        ax_flat[i].set_title(f'{abb}', fontsize=12)
    for i in range(i + 1, nrows * ncols):
        ax_flat[i].axis('off')
        ax_twin_flat.append(None)

    ax_twin = np.array(ax_twin_flat).reshape(nrows, ncols)
    for i in range(nrows):
        ax[i, 0].set_ylabel('CDF value', fontsize=12)
        if ax_twin[i, -1] is not None:
            ax_twin[i, -1].set_ylabel('Log density', fontsize=12)
    for i in range(ncols):
        ax[-1, i].set_xlabel('$t$', fontsize=12)

    handles, labels = ax_flat[0].get_legend_handles_labels()
    handles_twinx, labels_twix = ax_twin_flat[0].get_legend_handles_labels()
    handles += handles_twinx
    labels += labels_twix
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        frameon=False,
        ncol=3,
        fontsize=14,
    )
    fig.tight_layout()
