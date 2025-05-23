import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
import yaml
from omegaconf import OmegaConf
from pandas.io.formats.style_render import _escape_latex

from moc.configs.datasets import get_dataset_groups
from moc.datamodules import load_datamodule
from moc.recalibration.manager import recalibrators
from moc.utils.run_config import RunConfig

from .helpers import create_name_from_dict


def load_config(path):
    with (Path(path) / 'config.yaml').open() as f:
        config = OmegaConf.create(yaml.load(f, Loader=yaml.Loader), flags={'allow_objects': True})
    # We assume that the config file is in the root directory of log_dir
    config.log_dir = path
    return config


def make_df(config, dataset_group, dataset, reload=True):
    dataset_path = Path(config.log_dir) / dataset_group / dataset
    if not dataset_path.exists():
        return None
    df_path = dataset_path / 'df.pickle'
    if not reload and df_path.exists():
        with df_path.open('rb') as f:
            return pickle.load(f)
    series_list = []
    for run_config_path in dataset_path.rglob('run_config.pickle'):
        with run_config_path.open('rb') as f:
            rcs = pickle.Unpickler(f).load()
        for rc in rcs:
            series_list.append(rc.to_series())
    if len(series_list) == 0:
        return None
    df = pd.concat(series_list, axis=1).T
    with df_path.open('wb') as f:
        pickle.dump(df, f)
    return df


def load_df(config, dataset_group=None, dataset=None, reload=True):
    assert config is not None
    dfs = []
    path = Path(config.log_dir)
    dataset_groups = (
        [p.name for p in path.iterdir() if p.is_dir()] if dataset_group is None else [dataset_group]
    )
    for curr_dataset_group in dataset_groups:
        path = Path(config.log_dir) / curr_dataset_group
        datasets = [p.name for p in path.iterdir() if p.is_dir()] if dataset is None else [dataset]
        for curr_dataset in datasets:
            df = make_df(config, curr_dataset_group, curr_dataset, reload=reload)
            if df is not None:
                dfs.append(df)
    if not dfs:
        raise RuntimeError('Dataframe not found')
    return pd.concat(dfs)


def union(sets):
    res = set()
    for s in sets:
        res |= s
    return res


def get_metric_df(config, df):
    # Extract hyperparameters
    hparams = df.hparams.map(set).agg(union) | {'posthoc_method'}
    for hparam in hparams:
        df[hparam] = df.apply(lambda df: df.hparams.get(hparam, None), axis=1)
    # Extract metrics
    metrics = df.metrics.map(set).agg(union)
    for metric in metrics:
        df[metric] = df.apply(lambda df: df.metrics.get(metric, np.nan), axis=1)
    # Drop unnecessary columns
    df = df.drop(columns=['hparams', 'metrics', 'config', 'options'])

    # Add dataset infos
    df_ds = get_datasets_df(config, reload=True)
    df = df.merge(df_ds.reset_index(), on=['dataset_group', 'dataset'], how='inner')

    # Add scaled calibration metrics
    for metric in ['hdr_calibration', 'latent_calibration']:
        if metric in df.columns:
            df[f'{metric}_100'] = df[metric] * 100
            metrics.add(f'{metric}_100')
    
    # Add standard metrics
    standard_metrics = {
        'energy_score': 'energy_score_1',
    }
    for metric, column in standard_metrics.items():
        if column in df.columns:
            df[metric] = df[column]
            metrics.add(metric)

    # Stack the metrics
    other_columns = [col for col in df.columns if col not in metrics]
    df = df.set_index(other_columns)
    df = df.stack(future_stack=True).rename_axis(index={None: 'metric'}).to_frame(name='value')
    # Sort some columns
    names = df.index.names
    df = df.reset_index()
    df['metric'] = pd.Categorical(df['metric'], metrics)
    order = list(set(recalibrators.keys()))
    df['posthoc_method'] = pd.Categorical(df['posthoc_method'], order)
    order = df_ds.sort_values('n').reset_index()['dataset']
    df['dataset'] = pd.Categorical(df['dataset'], order)
    order = df_ds.sort_values('n').reset_index()['abb']
    df['abb'] = pd.Categorical(df['abb'], order)
    df = df.sort_values(['dataset', 'metric'])
    # Add `name` column
    model_name_partial = partial(create_name_from_dict, config=config)
    df['name'] = df.apply(model_name_partial, axis='columns').astype('string')

    df = df.set_index([*names, 'name'])
    return df


def agg_mean_sem(x):
    mean = np.mean(x)
    std = None
    if len(x) > 1:
        std = scipy.stats.sem(x, ddof=1)
    return (mean, std)


def format_cell_latex(x, mean_digits=3, sem_digits=2):
    if pd.isna(x):
        return 'NA'
    mean, sem = x
    if pd.isna(mean):
        return 'NA'
    if np.isposinf(mean):
        return r'$\infty$'
    if np.isneginf(mean):
        return r'$-\infty$'
    s = rf'\text{{{mean:#.{mean_digits}}}}'
    if sem is not None:
        sem = float(sem)
        s += rf'_{{\text{{{sem:#.{sem_digits}}}}}}'
    return f'${s}$'


def format_cell_jupyter(x, add_sem=False, mean_digits=3, sem_digits=2):
    if pd.isna(x):
        return 'NA'
    mean, sem = x
    if pd.isna(mean):
        return 'NA'
    if np.isposinf(mean):
        return '∞'
    if np.isneginf(mean):
        return '-∞'
    s = f'{mean:#.{mean_digits}}'
    if add_sem and sem is not None:
        s += f' ± {sem:#.{sem_digits}}'
    return s


def make_df_abb(datasets):
    assert len(datasets) > 0

    df_abb = pd.DataFrame({'dataset': datasets}).sort_values('dataset')
    df_abb['abb'] = df_abb['dataset'].str[:3].str.upper()
    if len(datasets) == 1:  # Special case here else droplevel(0) causes an error
        return df_abb

    def agg(x):
        x = x['abb']
        if len(x) > 1:
            x = x.str[:-1]
            x += np.arange(1, len(x) + 1).astype(str)
        return x

    df_abb['abb'] = df_abb.groupby('abb').apply(agg).droplevel(0)
    return df_abb.sort_values('dataset')


def compute_datasets_df(config):
    data = defaultdict(list)
    for dataset_group, datasets in get_dataset_groups(config.datasets).items():
        for dataset in datasets:
            rc = RunConfig(
                config=config,
                dataset_group=dataset_group,
                dataset=dataset,
            )
            datamodule = load_datamodule(rc)
            data_test = datamodule.data_test
            nb_instances = datamodule.total_size
            first_item = next(iter(data_test))
            x, y = first_item
            x_dim, y_dim = x.shape[0], y.shape[0]
            description = {
                'dataset_group': dataset_group,
                'dataset': dataset,
                'n': nb_instances,
                'p': x_dim,
                'd': y_dim,
            }
            for key, value in description.items():
                data[key].append(value)
    df = pd.DataFrame(data)
    df_abb = make_df_abb(df['dataset'].unique().astype(str))
    df = df.merge(df_abb, on='dataset')
    return df.set_index(['dataset_group', 'dataset', 'abb'])


def get_datasets_df(config, reload=False):
    path = Path(config.log_dir) / 'datasets_df.pickle'
    if not reload and path.exists():
        return pd.read_pickle(path)
    df = compute_datasets_df(config)
    df.to_pickle(path)
    return df


def get_value_counts(y):
    values, counts = torch.unique(y, return_counts=True, dim=0)
    indices = counts.argsort(descending=True)
    values, counts = values[indices], counts[indices]
    return values, counts


def get_info(y):
    values, counts = get_value_counts(y)
    N = y.shape[0]
    proportions = counts / N
    return {
        'Proportion of top 1 classes': proportions[:1].sum().item(),
        'Proportion of top 10 classes': proportions[:10].sum().item(),
        'Proportion of duplicated values': proportions[counts > 1].sum().item(),
    }


def apply_duplication_style(df):
    def bold_values(val):
        return 'font-weight: bold' if val > 0.5 else ''

    cols = [
        'Proportion of top 1 classes',
        'Proportion of top 10 classes',
        'Proportion of duplicated values',
    ]
    return df.style.map(bold_values, subset=cols).format(precision=3)


def compute_duplication_df(config):
    data = defaultdict(list)
    for dataset_group, datasets in get_dataset_groups(config.datasets).items():
        for dataset in datasets:
            rc = RunConfig(
                config=config,
                dataset_group=dataset_group,
                dataset=dataset,
            )
            datamodule = load_datamodule(rc)
            x, y = datamodule.data_train[:]
            data['Group'].append(dataset_group)
            data['Dataset'].append(dataset)
            data['Nb instances'].append(datamodule.total_size)
            data['Nb features'].append(x.shape[1])
            data['Nb targets'].append(y.shape[1])
            for key, value in get_info(y).items():
                data[key].append(value)
    return pd.DataFrame(data).set_index(['Group', 'Dataset'])


def get_duplication_df(config, reload=False):
    path = Path(config.log_dir) / 'duplication_df.pickle'
    if not reload and path.exists():
        return pd.read_pickle(path)
    df = compute_duplication_df(config)
    df.to_pickle(path)
    return df


def latex_style(styler):
    if styler.columns.names != [None]:
        styler.columns.names = list(map(_escape_latex, styler.columns.names))
    return styler.format_index(escape='latex', axis=0).format_index(escape='latex', axis=1)


def to_latex(
    styler,
    path=None,
    hrules=True,
    multicol_align='c',
    multirow_align='t',
    convert_css=True,
    **kwargs,
):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    return latex_style(styler).to_latex(
        path,
        hrules=hrules,
        multicol_align=multicol_align,
        multirow_align=multirow_align,
        convert_css=convert_css,
        **kwargs,
    )


def update_name(df, config, **kwargs):
    model_name_partial = partial(create_name_from_dict, config=config, **kwargs)
    df['name'] = df.apply(model_name_partial, axis='columns').astype('string')
