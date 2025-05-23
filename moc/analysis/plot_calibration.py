import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from moc.analysis.cmaps import get_cmap


def plot_consistency_bands(axis, n, p, coverage):
    low = (1 - coverage) / 2
    high = 1 - low
    assert p.ndim == 1
    low_band, high_band = stats.binom(n, p).ppf(np.array([low, high])[..., None]) / n
    axis.fill_between(p, low_band, high_band, alpha=0.1, color='orange')


def plot_sem(axis, lin, mean, sem, color=None, **kwargs):
    axis.plot(lin, mean, color=color, **kwargs)
    axis.fill_between(lin, mean - sem, mean + sem, alpha=0.2, color=color, zorder=10)


def plot_agg_runs(axis, data, label=None, color=None):
    data = np.sort(data, axis=1)
    n = data.shape[1]
    lin = (np.arange(n) + 1) / (n + 1)
    mean = data.mean(axis=0)
    sem = np.zeros_like(mean) if len(data) == 1 else stats.sem(data, axis=0, ddof=1)
    plot_sem(axis, lin, mean, sem, label=label, color=color)


def plot_reliability_diagrams(df, metric, config, ncols=5, ncols_legend=3):
    df = df.query('metric == @metric')
    df = df[['abb', 'name', 'run_id', 'value']]
    abbs = df['abb'].unique()

    colors_dict, _ = get_cmap(df, 'recalibrators')

    size = len(abbs)
    nrows = math.ceil(size / ncols)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 1.8, nrows * 1.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    ax_flatten = ax.flatten()
    for i in range(size, len(ax_flatten)):
        ax_flatten[i].set_visible(False)

    for axis, (abb, df_dataset) in zip(ax_flatten, df.groupby('abb', observed=True)):
        for model, df_model in df_dataset.groupby('name'):
            data = np.stack(df_model['value'].tolist(), axis=0)
            plot_agg_runs(
                axis,
                data,
                label=model,
                color=colors_dict[model],
            )
        axis.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
        n_test = df_dataset['value'].iloc[0].shape[0]
        alpha = torch.linspace(0, 1, 1000)
        plot_consistency_bands(axis, n_test, alpha, 0.9)
        title = f'{abb} ({n_test})'
        axis.set(title=title)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.tick_params(axis='both', which='major', labelsize=8)
        axis.tick_params(axis='both', which='minor', labelsize=6)

    for i in range(nrows):
        ax[i, 0].set_ylabel(r'$\hat{F}_U(\alpha)$')
    for i in range(ncols):
        ax[-1, i].set_xlabel(r'$\alpha$')

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        frameon=False,
        ncol=ncols_legend,
        fontsize=14,
        title_fontsize=14,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    return fig
