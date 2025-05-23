import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import rich.syntax
import rich.tree
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from omegaconf import DictConfig, OmegaConf


def configure_logging(level=logging.INFO):
    log = logging.getLogger('moc')
    log.setLevel(level)

    for h in log.handlers[:]:
        log.removeHandler(h)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    class IgnorePLFilter(logging.Filter):
        def filter(self, record):
            keywords = ['available:', 'CUDA', 'LOCAL_RANK:']
            return not any(keyword in record.getMessage() for keyword in keywords)

    logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(IgnorePLFilter())
    logging.getLogger('lightning.pytorch.accelerators.cuda').addFilter(IgnorePLFilter())

    warnings.filterwarnings(
        'ignore',
        r'.*GPU available but not used\. Set `accelerator` and `devices` using.*',
        category=PossibleUserWarning,
    )
    warnings.filterwarnings('ignore', message='brute force', module='cpflows\.flows\.cpflows')


def set_notebook_options(log_level=logging.INFO):
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 80)
    pd.set_option('display.float_format', '{:.3f}'.format)
    plt.rcParams.update(
        {
            'axes.formatter.limits': (-2, 4),
            'axes.formatter.use_mathtext': True,
            'text.usetex': True,
            'font.family': 'serif',
            'text.latex.preamble': r"""
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{amssymb}
            """,
        }
    )
    configure_logging(level=log_level)


def savefig(path, fig=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    fig.savefig(
        path,
        bbox_extra_artists=fig.legends or None,
        bbox_inches='tight',
        **kwargs,
    )
    plt.close(fig)


def plot_or_savefig(path=None, fig=None, **kwargs):
    if path is None:
        plt.show()
    else:
        savefig(path, fig=fig, **kwargs)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Optional[Sequence[str]] = None,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = 'dim'
    tree = rich.tree.Tree('config', style=style, guide_style=style)

    if fields is None:
        fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))

    rich.print(tree)

    path = Path(config.log_dir)
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / 'config_tree.log'
    with config_path.open('w') as fp:
        rich.print(tree, file=fp)
