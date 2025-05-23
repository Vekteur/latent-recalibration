from moc.models.arflow.arflow import ARFlowLightningModule
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.tarflow.tarflow import TarFlowPretrained
from moc.models.trainers.default_trainer import DefaultTrainer
from moc.models.trainers.lightning_trainer import get_lightning_trainer

models = {
    'ARFlow': ARFlowLightningModule,
    'MQF2': MQF2LightningModule,
    'TarFlow': TarFlowPretrained,
}

trainers = {
    'ARFlow': get_lightning_trainer,
    'MQF2': get_lightning_trainer,
    'TarFlow': DefaultTrainer,
}
