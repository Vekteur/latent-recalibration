import pandas as pd


def create_name_from_dict(
    d, config
):
    name = get_posthoc_method_name(d['posthoc_method'], config)
    return name


def get_posthoc_method_name(method, config):
    d = {}
    if config.name.startswith('lr'):
        if pd.isna(method):
            method = r'\texttt{BASE}'
        d = {
            'Latent': r'\texttt{LR}',
            'HDR': r'\texttt{HDR-R}',
        }
    return d.get(method, method)


def get_metric_name(metric):
    d = {
        'nll': 'NLL',
        'energy_score': 'ES',
        'gaussian_kernel_score': 'GKS',
        'hdr_calibration': 'HDR-ECE',
        'latent_calibration': 'L-ECE',
    }
    return d.get(metric, metric)
