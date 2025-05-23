import json
import logging

import torch
from lightning import Trainer

from moc.configs.general import PrecomputationLevel
from moc.utils.general import seed_everything, SeedOffset

from . import models, trainers

log = logging.getLogger(__name__)


def train(rc, datamodule):
    seed_everything(rc.run_id + SeedOffset.MODEL)
    model_path = rc.checkpoints_path / 'best.pth'

    # Instantiate the model with the correct arguments
    model_kwargs = rc.hparams.copy()
    model_name = model_kwargs.pop('model')
    model_cls = models[model_name]
    p, d = datamodule.input_dim, datamodule.output_dim
    model_kwargs['input_dim'], model_kwargs['output_dim'] = p, d

    # Load tuned hyperparameters
    if rc.config.hparams_path:
        with open(rc.config.hparams_path) as f:
            hparams_loaded = json.load(f)[rc.dataset][model_name]
        for key in hparams_loaded:
            if key in model_kwargs:
                log.warning(f'Loaded hparam {key} overrides {model_kwargs[key]} with {hparams_loaded[key]}')
            model_kwargs[key] = hparams_loaded[key]

    model = model_cls(**model_kwargs)
    model.to(rc.config.device)
    # Instantiate the trainer
    trainer = trainers[model_name](rc=rc)

    # Load the model if it exists, otherwise train it
    load_cpkt_condition = (
        rc.config.precomputation_level >= PrecomputationLevel.MODELS
        and model_path.exists()
        and isinstance(trainer, Trainer)
    )
    if load_cpkt_condition:  # Load model if it exists
        model = model_cls.load_from_checkpoint(model_path)
        log.info(f'Finished loading {rc.summary_str()}')
    else:  # Train
        trainer.fit(model=model, datamodule=datamodule)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(trainer, Trainer):
            trainer.save_checkpoint(model_path)
        log.info(f'Finished training {rc.summary_str()}')
    # The device of the model can sometimes change to CPU after training, so we transfer to the correct device
    model.to(rc.config.device)
    # Set the model to eval mode if it is a torch model
    if isinstance(model, torch.nn.Module):
        model.eval()
    return model
