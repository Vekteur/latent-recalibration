import logging

import torch

from moc.datamodules.utils import DimSlicedTensorDataset, VectorizedDataLoader

log = logging.getLogger(__name__)


def generate_grid(grid_side, xlim, ylim, n, device='cpu'):
    y1, y2 = (
        torch.linspace(*xlim, grid_side, device=device),
        torch.linspace(*ylim, grid_side, device=device),
    )
    Y1, Y2 = torch.meshgrid(y1, y2, indexing='ij')
    y_grid = torch.dstack((Y1, Y2))
    y_grid = y_grid[:, :, None, :].expand(-1, -1, n, -1)
    assert y_grid.shape == (y1.shape[0], y1.shape[0], n, 2)
    return y_grid


def generate_dl_plot(grid_side, xlim, ylim, x_plot, device='cpu'):
    y_plot = generate_grid(grid_side, xlim, ylim, len(x_plot), device=device)
    data_plot = DimSlicedTensorDataset((x_plot, y_plot), (0, 2))
    return VectorizedDataLoader(data_plot, batch_size=1, shuffle=False)
