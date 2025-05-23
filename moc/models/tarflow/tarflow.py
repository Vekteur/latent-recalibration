import subprocess
from pathlib import Path

import torch
from torch.distributions import constraints, Independent, Normal, Transform

from moc.models.utils import CustomTransformedDistribution

from .transformer_flow import Model


class TarFlowTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, model, x):
        super().__init__(cache_size=1)
        self.model = model
        self.x = x
        self.image_shape = self.model.image_shape
        self.latent_shape = self.model.latent_shape
        self._cached_forward_logdet = None

    def _call(self, z):
        """
        `z` is of shape (..., batch_size, d)
        `self.x` is of shape (batch_size, 1)
        """
        batch_shape = self.x.shape[:-1]
        sample_shape = z.shape[:-2]
        # Repeat `self.x` to match the shape of `z`
        x_repeat = self.x.view((1,) * len(sample_shape) + self.x.shape).expand(sample_shape + self.x.shape)
        # Flatten `z` and `x_repeat` to pass them to the flow
        z_flat = z.reshape(-1, *self.image_shape)
        x_repeat_flat = x_repeat.reshape(-1)

        # Flow forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            z_flat, _, logdet_flat = self.model.forward(z_flat, x_repeat_flat)

        # Reshape back to original shape
        z = z_flat.reshape((*sample_shape, *batch_shape, self.latent_shape.numel()))
        logdet = logdet_flat.reshape(sample_shape + batch_shape)
        # Important: in transformer_flow, the logdet is scaled by the number of pixels
        # To be consistent with the rest of the code, we need to multiply by the number of pixels
        logdet *= self.image_shape.numel()
        self._cached_forward_logdet = logdet
        return z

    def _inverse(self, z):
        """
        `z` is of shape (..., batch_size, d)
        `self.x` is of shape (batch_size, 1)
        """
        batch_shape = self.x.shape[:-1]
        sample_shape = z.shape[:-2]
        # Repeat `self.x` to match the shape of `z`
        x_repeat = self.x.view((1,) * len(sample_shape) + self.x.shape).expand(sample_shape + self.x.shape)
        # Flatten `z` and `x_repeat` to pass them to the flow
        z_flat = z.reshape(-1, *self.latent_shape)
        x_repeat_flat = x_repeat.reshape(-1)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            z_flat = self.model.reverse(z_flat, x_repeat_flat)

        # Reshape back to original shape
        z = z_flat.reshape((*sample_shape, *batch_shape, self.image_shape.numel()))
        self._cached_forward_logdet = None
        return z

    def log_abs_det_jacobian(self, x, y):
        # Use cached forward logdet if available
        x_old, y_old = self._cached_x_y
        if x is x_old and y is y_old and self._cached_forward_logdet is not None:
            return self._cached_forward_logdet
        self._call(x)
        return self._cached_forward_logdet


dataset = 'afhq'
num_classes = 3
img_size = 256
channel_size = 3

patch_size = 8
channels = 768
blocks = 8
layers_per_block = 8
noise_std = 0.07

input_shape = torch.Size((1,))
image_shape = torch.Size((channel_size, img_size, img_size))
latent_shape = torch.Size(((img_size // patch_size) ** 2, channel_size * patch_size**2))


class TarFlowPretrained(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        assert input_dim == input_shape.numel()
        assert output_dim == image_shape.numel() == latent_shape.numel()

        model_name = f'{patch_size}_{channels}_{blocks}_{layers_per_block}_{noise_std:.2f}'
        ckpt_file = Path('models') / f'{dataset}_model_{model_name}.pth'
        if not ckpt_file.exists():
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            url = 'https://ml-site.cdn-apple.com/models/tarflow/afhq256/afhq_model_8_768_8_8_0.07.pth'
            subprocess.run(['wget', '-q', url, '-O', str(ckpt_file)], check=True)

        self.model = Model(
            in_channels=channel_size,
            img_size=img_size,
            patch_size=patch_size,
            channels=channels,
            num_blocks=blocks,
            layers_per_block=layers_per_block,
            num_classes=num_classes,
        )
        ckpt = torch.load(ckpt_file, map_location='cpu')
        self.model.load_state_dict(ckpt, strict=True)
        self.model.image_shape = image_shape
        self.model.latent_shape = latent_shape

    def fit(self, datamodule):
        assert datamodule.dataset == 'afhq', 'TarFlowPretrained only supports AFHQ dataset'

    def latent_dist(self):
        return Independent(
            Normal(
                torch.tensor(0.0, dtype=self.dtype, device=self.device),
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
            ).expand([latent_shape.numel()]),
            1,
        )

    def predict(self, x):
        batch_shape = x.shape[:-1]
        return CustomTransformedDistribution(
            self.latent_dist().expand(batch_shape), [TarFlowTransform(self.model, x).inv]
        )

    def forward(self, x):
        return self.predict(x)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def output_type(cls):
        return 'distribution'
