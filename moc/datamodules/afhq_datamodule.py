from pathlib import Path

import torch
import torchvision as tv
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from moc.utils.general import seed_everything, SeedOffset


class CustomImageFolder(tv.datasets.ImageFolder):
    def __init__(self, *args, gaussian_noise=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_noise = gaussian_noise

    def __getitem__(self, idx):
        img, class_idx = super().__getitem__(idx)
        # For compatibility with other functions, we treat the image as unidimensional and transform the image
        # to the correct shape inside image models.
        img = img.flatten(start_dim=-3)
        if self.gaussian_noise:
            img += self.gaussian_noise * torch.randn_like(img)
        return class_idx, img


class AFHQDataModule(LightningDataModule):
    def __init__(
        self,
        rc=None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore='rc', logger=False)
        self.rc = rc
        self.dataset_group = rc.dataset_group
        self.dataset = rc.dataset
        self.load_datasets()

    def subsample(self, dataset, max_dataset_size):
        n = len(dataset)
        if n <= max_dataset_size:
            return dataset
        return torch.utils.data.Subset(
            dataset,
            torch.randperm(n)[:max_dataset_size],
        )

    def load_datasets(self):
        seed_everything(self.rc.run_id + SeedOffset.DATAMODULE)
        img_size = 256

        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(img_size),
                tv.transforms.CenterCrop(img_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        def get_dataset(name):
            path = Path('data') / 'afhq' / name
            return CustomImageFolder(
                path,
                transform=transform,
                target_transform=lambda y: torch.tensor([y]),
                gaussian_noise=self.rc.config.afhq_noise,
            )

        # We actually use the conventional train set as validation set for recalibration.
        # The conventional validation set, which has not been seen, is used for testing.
        self.data_val = get_dataset('train')
        self.data_test = get_dataset('val')

        if self.rc.config.fast:
            self.data_val = self.subsample(self.data_val, 10)
            self.data_test = self.subsample(self.data_test, 10)

        # Save inputs and outputs dimensions
        x, y = self.data_val[0]
        self.input_dim = x.shape[0]
        self.output_dim = y.shape[0]
        self.total_size = len(self.data_val) + len(self.data_test)

    def get_dataloader(self, dataset, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.rc.config.default_batch_size
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)
