camehl_datasets = [
    'households',
]

cevid_datasets = [
    'air',
    'births1',
    'births2',
    'wage',
]

del_barrio_datasets = [
    'ansur2',
    'calcofi',
]

feldman_datasets = [
    'bio',
    'blog_data',
    'house',
    'meps_19',
    'meps_20',
    'meps_21',
]

mulan_datasets = [
    'atp1d',
    'atp7d',
    'oes97',
    'oes10',
    'rf1',
    #'rf2', # Same as rf1 after preprocessing
    'scm1d',
    'scm20d',
    'edm',
    'sf1',
    'sf2',
    'jura',
    'wq',
    'enb',
    'slump',
    #'andro', # Too small
    #'osales', # Outliers
    'scpf',
]

wang_datasets = [
    #'energy', # Same as enb in MULAN
    'taxi',
]

real_dataset_groups = {
    'camehl': camehl_datasets,
    'cevid': cevid_datasets,
    'del_barrio': del_barrio_datasets,
    'feldman': feldman_datasets,
    'mulan': mulan_datasets,
    'wang': wang_datasets,
}

toy_2dim_datasets = [
    'unimodal_heteroscedastic',
    'unimodal_heteroscedastic_power_0',
    'unimodal_heteroscedastic_power_0.1',
    'unimodal_heteroscedastic_power_0.5',
    'unimodal_heteroscedastic_power_2',
    'unimodal_heteroscedastic_power_10',
    'bimodal_heteroscedastic',
    'bimodal_heteroscedastic_power_0',
    'bimodal_heteroscedastic_power_0.1',
    'bimodal_heteroscedastic_power_0.5',
    'bimodal_heteroscedastic_power_2',
    'bimodal_heteroscedastic_power_10',
    'toy_hallin',
    'toy_del_barrio',
    'one_moon_heteroscedastic',
    'two_moons_heteroscedastic',
]

toy_ndim_datasets = [
    'mvn_isotropic_1',
    'mvn_isotropic_3',
    'mvn_isotropic_10',
    'mvn_isotropic_30',
    'mvn_isotropic_100',
    'mvn_diagonal_1',
    'mvn_diagonal_3',
    'mvn_diagonal_10',
    'mvn_diagonal_30',
    'mvn_diagonal_100',
    'mvn_dependent_1',
    'mvn_dependent_3',
    'mvn_dependent_10',
    'mvn_dependent_30',
    'mvn_dependent_100',
    'mvn_mixture_1_10',
    'mvn_mixture_3_10',
    'mvn_mixture_10_10',
    'mvn_mixture_30_10',
    'mvn_mixture_100_10',
]

toy_dataset_groups = {
    'toy_2dim': toy_2dim_datasets,
    'toy_ndim': toy_ndim_datasets,
}

image_dataset_groups = {
    'cifar10': ['cifar10'],
    'mnist': ['mnist'],
    'afhq': ['afhq'],
}

all_dataset_groups = {
    **real_dataset_groups,
    **toy_dataset_groups,
    **image_dataset_groups,
}

filtered_datasets = {
    'camehl': ['households'],
    'del_barrio': ['calcofi'],
    'feldman': feldman_datasets,
    'mulan': ['scm20d', 'rf1', 'rf2', 'scm1d'],
    'wang': ['taxi'],
}

small_datasets = {
    'toy_2dim': [
        'unimodal_heteroscedastic',
        'two_moons_heteroscedastic',
    ]
}

toy_visualization_datasets = {
    'toy_2dim': [
        'one_moon_heteroscedastic',
        'bimodal_heteroscedastic',
    ]
}


def get_dataset_groups(key):
    # General key for any group
    if key in all_dataset_groups:
        return {key: all_dataset_groups[key]}
    # General key for any dataset
    if key.startswith('single-'):
        group, dataset = key.split('-')[1:]
        return {group: [dataset]}
    # Specific keys
    if key == 'default':
        key = 'filtered'
    if key == 'all':
        return all_dataset_groups
    if key == 'filtered':
        return filtered_datasets
    if key == 'real':
        return real_dataset_groups
    if key == 'toy_visualization':
        return toy_visualization_datasets
    if key == 'small':
        return small_datasets
    if key == 'test':
        return {'toy_ndim': ['mvn_mixture_10_10']}

    raise ValueError(f'Unknown dataset group {key}')
