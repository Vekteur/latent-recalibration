import logging
import math

import torch
from scipy import optimize

from .smooth_empirical_cdf import SmoothEmpiricalCDF

log = logging.getLogger(__name__)


def eval_nll_cv(z, b, n_folds=10):
    n_folds = min(n_folds, len(z))
    folds = torch.chunk(z, n_folds)
    nlls = []
    for i in range(len(folds)):
        train = torch.cat(folds[:i] + folds[i + 1 :])
        val = folds[i]
        cdf = SmoothEmpiricalCDF(train, b=b)
        nlls_fold = -cdf.distribution.log_prob(val)
        nlls.append(nlls_fold)
    return torch.cat(nlls).mean().item()


def select_bandwidth_cv_opti(z, n_folds=10):
    def f(b):
        return eval_nll_cv(z, b, n_folds=n_folds)

    bmin, fval, iter, funcalls = optimize.minimize_scalar(
        lambda b: f(math.exp(b)), bounds=(2e-5, None), tol=1e-5, full_output=True
    )
    return math.exp(bmin), fval


def select_bandwidth_cv_grid(z, n_folds=10):
    bs = torch.logspace(-5, 5, 100)
    nlls = torch.tensor([eval_nll_cv(z, b.item(), n_folds=n_folds) for b in bs])
    index = torch.argmin(nlls)
    return bs[index].item(), nlls[index].item()


def select_bandwidth_cv(z, n_folds=10):
    b_grid, nll_grid = select_bandwidth_cv_grid(z, n_folds=10)
    log.debug(f'Grid: {b_grid:.4g}, NLL: {nll_grid:.4g}')
    return b_grid, nll_grid
