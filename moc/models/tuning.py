"""
Selects which models, methods and hyperparameters to run.
Models, methods or hyperparameters are declared using HP(name=value) or HP(name=[value1, value2, ...]).
Join combines sets of hyperparameters in a cartesian product (similar to a grid search).
Union concatenates sets of hyperparameters.
"""

import collections

from moc.utils.hparams import HP, Join, Union


def get_tuning_lr_arflow(config):
    recalibration_grid = Union(
        HP(method=None),
        Join(HP(method='Latent'), HP(density_estimator=['kde', 'spline'])),
        Join(HP(method='HDR'), HP(B=[20])),
    )

    return Join(
        HP(model=['ARFlow']),
        HP(transform_type=['spline-quadratic']),
        HP(hidden_size=[64]),
        HP(num_layers=[2]),
        HP(num_flows=[3]),
        HP(recalibration_grid=recalibration_grid),
    )


def get_tuning_lr_mqf2(config):
    recalibration_grid = Union(
        HP(method=None),
        Join(HP(method='Latent'), HP(density_estimator=['kde', 'spline'])),
        Join(HP(method='HDR'), HP(B=[20])),
    )

    return Join(
        HP(model=['MQF2']),
        HP(recalibration_grid=recalibration_grid),
    )


def get_tuning_lr_tarflow(config):
    recalibration_grid = HP(method=[None, 'Latent'])

    return Join(
        HP(model=['TarFlow']),
        HP(recalibration_grid=recalibration_grid),
    )


tuning_dict = {
    'lr_arflow': get_tuning_lr_arflow,
    'lr_mqf2': get_tuning_lr_mqf2,
    'lr_tarflow': get_tuning_lr_tarflow,
}


def _get_tuning(config):
    if config.tuning_type not in tuning_dict:
        raise ValueError(f'Invalid tuning type: {config.tuning_type}.')
    return tuning_dict[config.tuning_type](config)


def duplicates(choices):
    def frozendict(d):
        return frozenset(d.items())

    frozen_choices = map(frozendict, choices)
    return [choice for choice, count in collections.Counter(frozen_choices).items() if count > 1]


def remove_duplicates(seq_of_dicts):
    seen = set()
    deduped_seq = []

    for d in seq_of_dicts:
        t = tuple(frozenset(d.items()))
        if t not in seen:
            seen.add(t)
            deduped_seq.append(d)

    return deduped_seq


def filter(config, seq_of_dicts):
    if config.selected_models is not None:
        seq_of_dicts = [d for d in seq_of_dicts if d['model'] in config.selected_models]
    return seq_of_dicts


def get_tuning(config):
    tuning = _get_tuning(config)
    tuning = remove_duplicates(tuning)
    dup = duplicates(tuning)
    assert len(dup) == 0, dup
    tuning = filter(config, tuning)
    return tuning
