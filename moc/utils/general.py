import random
from contextlib import contextmanager
from enum import IntEnum
from timeit import default_timer

import numpy as np
import torch


def filter_dict(d, keys):
    return {key: d[key] for key in keys if key in d}


def inter(l1, l2):
    return [value for value in l1 if value in l2]


@contextmanager
def elapsed_timer():
    start = default_timer()

    def elapser():
        return default_timer() - start

    yield lambda: elapser()
    end = default_timer()

    def elapser():
        return end - start


done_once = set()


def once(key):
    if key in done_once:
        return False
    done_once.add(key)
    return True


def print_once(key, string, box=True):
    if once(key):
        if box:
            '=' * (len(string) + 6)
        else:
            pass


# We don't use the version of cycle from itertools because it returns
# a saved copy of the iterable, which will not be reshuffled.
def cycle(iterable):
    while True:
        yield from iterable


# The goals are:
# 1. Ensure reproducibility across different components (e.g., dataloaders, models, etc.)
# 2. Ensure reproducibility across different runs
class SeedOffset(IntEnum):
    DATAMODULE = 0
    DATA_SHUFFLING = 1000
    TRAIN_DATALOADER = 2000
    MODEL = 3000
    POSTHOC_INIT = 4000
    POSTHOC_METRICS = 5000


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def op_without_index(df, op):
    names = df.index.names
    df = op(df.reset_index())
    if names != [None]:
        df = df.set_index(names)
    return df
