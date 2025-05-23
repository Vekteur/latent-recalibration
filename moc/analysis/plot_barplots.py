import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .cmaps import get_cmap
from .helpers import get_metric_name

baseline_query = 'posthoc_method.isna()'


def relative_improvement(df):
    baseline_value = df.query(baseline_query)['value'].item()
    df['value'] = (df['value'] - baseline_value) / abs(baseline_value)
    return df


def compute_cohen_d(df, metric):
    df = df.copy()
    df = df.query('metric == @metric')
    df = df.groupby(['dataset_group', 'dataset', 'run_id', 'model'], dropna=False, observed=True).apply(
        relative_improvement, include_groups=False
    )
    df['metric'] = f'relative_{metric}'
    return df


def add_cohen_d(df, metric):
    cohen_d_df = compute_cohen_d(df, metric)
    return pd.concat([df, cohen_d_df])


def barplot(axis, df, metric, relative=False):
    if relative:
        df = add_cohen_d(df, metric)
        df = df.query(f'metric == "relative_{metric}"')
        df = df.query(f'not ({baseline_query})')
    else:
        df = df.query(f'metric == "{metric}"')
    if metric in ['latent_calibration', 'nll']:
        df = df.query('posthoc_method != "HDR"')
    cmap, _ = get_cmap(df, 'recalibrators')

    names = df['name'].unique()
    names_not_in_cmap = [name for name in names if name not in cmap]
    assert len(names_not_in_cmap) == 0, f'Names not in cmap: {names_not_in_cmap}'
    df = df.copy()  # Avoid warning
    df['name'] = pd.Categorical(df['name'], categories=[name for name in cmap if name in names], ordered=True)

    g = sns.barplot(df, x='abb', y='value', hue='name', palette=cmap, ax=axis)
    g.legend().remove()
    if metric in ['nll', 'energy_score']:
        axis.set_yscale('symlog', linthresh=0.1)
    ylabel = f'Relative {get_metric_name(metric)}' if relative else get_metric_name(metric)
    axis.set(xlabel=None)
    axis.set_ylabel(ylabel, fontsize=11)
    if metric == 'energy_score':
        axis.set_ylim(-1, 1)
    plt.xticks(rotation=90)
    axis.tick_params(axis='y', which='major', labelsize=9)
    if relative:
        axis.axhline(0, color=cmap[r'\texttt{BASE}'], lw=2, label=r'\texttt{BASE}')


def barplots(df, metrics, width=7):
    nrows = len(metrics)
    fig, ax = plt.subplots(nrows, figsize=(width, 1.2 * nrows), sharex=True)
    for metric, axis in zip(metrics, ax):
        barplot(axis, df, metric, relative=metric in ['nll', 'energy_score'])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=11, frameon=False
    )
    fig.tight_layout()
