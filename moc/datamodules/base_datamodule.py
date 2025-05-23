import logging
from abc import abstractmethod

import numpy as np
import torch
from lightning.pytorch import LightningDataModule

from moc.utils.general import seed_everything, SeedOffset

from .utils import (
    DimSlicedTensorDataset,
    scale_tensor_dataset,
    shuffle_tensor_dataset,
    split_tensor_dataset,
    StandardScaler,
    VectorizedDataLoader,
)

log = logging.getLogger(__name__)


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        rc=None,
    ):
        super().__init__()

        self.rc = rc
        self.dataset_group = rc.dataset_group
        self.dataset = rc.dataset
        self.train_val_calib_test_split_ratio = rc.config.train_val_calib_test_split_ratio
        self.load_datasets()

    @abstractmethod
    def get_data(self):
        pass

    def subsample(self, dataset, max_dataset_size):
        n = len(dataset)
        if n <= max_dataset_size:
            return dataset
        return dataset.index(np.arange(max_dataset_size))

    def normalize(self, dataset):
        return scale_tensor_dataset(dataset, [self.scaler_x, self.scaler_y])

    def adjust_splits_size(self, n):
        # Convert ratios to number of elements in the dataset
        splits_size = torch.tensor(self.train_val_calib_test_split_ratio) * n
        log.debug(f'Total size: {n}')
        log.debug(f'Splits size before: {splits_size}')
        # Adjust the calibration set size to be at most 2048
        calib_index = 2
        to_remove_from_calib = max(0, splits_size[calib_index] - 2048)
        splits_size[calib_index] -= to_remove_from_calib
        # The mask indicates which splits should be adjusted
        # Splits of size 0 should remain empty
        mask = (splits_size != 0) & (torch.arange(len(splits_size)) != calib_index)
        # Redistribution of the removed elements
        splits_size[mask] += to_remove_from_calib / mask.sum()
        splits_size = splits_size.to(torch.int32)
        # Make sure that the sum of the splits size is equal to the total size
        splits_size[-1] = n - splits_size[:-1].sum()
        log.debug(f'Splits size after: {splits_size}')
        return splits_size.tolist()

    def load_datasets(self):
        seed_everything(self.rc.run_id + SeedOffset.DATAMODULE)
        x, y = self.get_data()
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        data = DimSlicedTensorDataset((x, y))

        # Shuffle
        shuffle_seed = self.rc.run_id + SeedOffset.DATA_SHUFFLING
        data = shuffle_tensor_dataset(data, generator=torch.Generator().manual_seed(shuffle_seed))

        # Subsample
        max_dataset_size = self.rc.config.max_dataset_size
        if self.rc.config.fast:
            max_dataset_size = 25
        data = self.subsample(data, max_dataset_size=max_dataset_size)
        self.total_size = len(data)

        # Split
        splits_size = self.adjust_splits_size(self.total_size)
        (
            self.data_train,
            self.data_val,
            self.data_calib,
            self.data_test,
        ) = split_tensor_dataset(data, splits_size)

        # Scale
        x_train, y_train = self.data_train[:]
        self.scaler_x = StandardScaler().fit(x_train)
        self.scaler_y = StandardScaler().fit(y_train)
        if self.rc.config.normalize:
            self.data_train = self.normalize(self.data_train)
            self.data_val = self.normalize(self.data_val)
            self.data_calib = self.normalize(self.data_calib)
            self.data_test = self.normalize(self.data_test)

        # Save inputs and outputs dimensions
        self.input_dim = x_train.shape[1]
        self.output_dim = y_train.shape[1]

    def get_dataloader(self, dataset, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.rc.config.default_batch_size
        return VectorizedDataLoader(dataset, batch_size=batch_size, **kwargs)

    def train_dataloader(self):
        seed = self.rc.run_id + SeedOffset.TRAIN_DATALOADER
        return self.get_dataloader(
            self.data_train, drop_last=True, shuffle=True, generator=torch.Generator().manual_seed(seed)
        )

    def val_dataloader(self):
        return self.get_dataloader(self.data_val)

    def calib_dataloader(self):
        return self.get_dataloader(self.data_calib)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)
