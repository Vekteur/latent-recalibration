import torch
from torch.distributions import AffineTransform
from torch.utils.data import Dataset


class DimSlicedTensorDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, tensors, dims=None) -> None:
        if dims is None:
            dims = [0] * len(tensors)
        self.len = tensors[0].size(dims[0])
        assert all(tensor.size(dim) == self.len for tensor, dim in zip(tensors, dims)), (
            'Size mismatch between tensors'
        )
        self.tensors = tensors
        self.dims = dims

    def __getitem__(self, index):
        item = []
        for tensor, dim in zip(self.tensors, self.dims):
            # Handle negative dimensions
            dim = dim % tensor.dim()
            item.append(tensor[(slice(None),) * dim + (index,)])
        return tuple(item)

    def index(self, index):
        return DimSlicedTensorDataset(self[index], self.dims)

    def __len__(self):
        return self.len


class StandardScaler:
    def __init__(self, mean=None, scale=None, epsilon=1e-7):
        """
        Standard Scaler for PyTorch tensors.
        """
        self.mean_ = mean
        self.scale_ = scale
        self.epsilon = epsilon
        if mean is not None and scale is not None:
            self.transformer = AffineTransform(loc=mean, scale=scale + epsilon).inv

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean_ = torch.mean(values, dim=dims)
        self.scale_ = torch.std(values, dim=dims)
        self.transformer = AffineTransform(loc=self.mean_, scale=self.scale_ + self.epsilon).inv
        return self

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    def transform(self, values):
        return self.transformer(values)

    def inverse_transform(self, values):
        return self.transformer.inv(values)

    def to(self, device):
        return StandardScaler(
            mean=self.mean_.to(device),
            scale=self.scale_.to(device),
            epsilon=self.epsilon,
        )


class VectorizedDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    the standard DataLoader. This implementation is vectorized and does not need
    to grab individual indices of the dataset.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, generator=None):
        self.dataset = dataset
        self.batch_size = min(batch_size, len(self.dataset))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        # Calculate number of batches
        n_batches, remainder = divmod(len(self.dataset), self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        return VectorizedIterator(self)

    def __len__(self):
        return self.n_batches


class VectorizedIterator:
    """
    An iterator for FastTensorDataLoader that allows parallel independent iterations.
    """

    def __init__(self, loader):
        self.loader = loader
        self.i = 0
        if loader.shuffle:
            self.dataset = shuffle_tensor_dataset(loader.dataset, generator=loader.generator)
        else:
            self.dataset = loader.dataset

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.loader.n_batches:
            raise StopIteration
        batch = self[self.i]
        self.i += 1
        return batch

    def __getitem__(self, index):
        return self.dataset[index * self.loader.batch_size : (index + 1) * self.loader.batch_size]


def shuffle_tensor_dataset(dataset, generator=None):
    """
    Shuffle a TensorDataset.
    """
    indices = torch.randperm(len(dataset), generator=generator)
    return dataset.index(indices)


def split_tensor_dataset(dataset, splits_size, shuffle=False, generator=None):
    """
    Split a TensorDataset into multiple datasets.
    """
    if shuffle:
        dataset = shuffle_tensor_dataset(dataset, generator=generator)
    splits = torch.split(torch.arange(len(dataset)), splits_size)
    return (dataset.index(split) for split in splits)


def scale_tensor_dataset(dataset, scalers):
    """
    Scale a TensorDataset using a list of scalers.
    """
    return DimSlicedTensorDataset(
        [scaler.transform(tensor) for tensor, scaler in zip(dataset.tensors, scalers)], dims=dataset.dims
    )
