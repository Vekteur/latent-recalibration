import torch
from torch.distributions import Distribution

from moc.metrics.calibration import hpd


class HDRRecalibratedDistribution(Distribution):
    has_log_prob = False

    def __init__(self, uncal_dist, R):
        self.uncal_dist = uncal_dist
        self.R = R  # Recalibration map
        self.device = R.device
        self._cached_sample = None
        self._cached_sample_log_prob = None
        super().__init__(uncal_dist.batch_shape, uncal_dist.event_shape, validate_args=False)

    def sample(self, sample_shape, oversampling=1):
        """
        Args:
            sample_shape (tuple): Shape of the sample to be generated. Must be of the form (m,)
                with m divisible by B
            oversampling (int): If > 1, it samples more points than needed, reducing the number of duplicates
                but increasing sampling time.
        """
        assert len(sample_shape) == 1, 'Only a sample shape of length 1 is supported'
        B = self.R.shape[0] - 1  # Number of bins
        # Sample from the uncalibrated distribution and evaluate the log densities
        base_sample = self.uncal_dist.sample((sample_shape[0] * oversampling,))
        m, b, d = base_sample.shape
        assert m % B == 0, 'Number of samples must be divisible by K'
        base_sample_log_prob = self.uncal_dist.log_prob(base_sample)
        # Sort samples in ascending order of densities
        indices = torch.argsort(base_sample_log_prob, dim=0)
        base_sample = base_sample.gather(0, indices.unsqueeze(-1).expand(-1, -1, d))
        base_sample_log_prob = base_sample_log_prob.gather(0, indices)
        # Divide samples into K bins
        B_i = m // B  # Number of samples per bin
        base_sample_log_prob = base_sample_log_prob.reshape(B, B_i, b)
        base_sample = base_sample.reshape(B, B_i, b, d)
        # Compute the number of samples in each bin
        assert m % oversampling == 0
        m //= oversampling
        n_resample_per_bin = (self.R * m).long().diff()
        assert n_resample_per_bin.shape == (B,)
        assert n_resample_per_bin.sum() == m
        # Resample from each bin
        sample_per_bin = []
        sample_log_prob_per_bin = []
        for i in range(B):
            K_i = n_resample_per_bin[i]
            if K_i == 0:
                continue
            # Sample from the bin with replacement
            indices_i = torch.randint(0, B_i, (K_i, b), device=base_sample.device)
            sample_i = base_sample[i].gather(0, indices_i.unsqueeze(-1).expand(-1, -1, d))
            base_sample_log_prob_i = base_sample_log_prob[i].gather(0, indices_i)
            assert sample_i.shape == (K_i, b, d)
            sample_per_bin.append(sample_i)
            sample_log_prob_per_bin.append(base_sample_log_prob_i)
        # Concatenate samples and log probs from all bins
        sample = torch.cat(sample_per_bin, dim=0)
        sample_log_prob = torch.cat(sample_log_prob_per_bin, dim=0)
        # Important: don't forget to shuffle samples
        perms = torch.rand(m, b, device=sample.device).argsort(dim=0)
        sample = sample.gather(0, perms.unsqueeze(-1).expand(-1, -1, d))
        sample_log_prob = sample_log_prob.gather(0, perms)
        # Keep in cache
        self._cached_sample = sample
        self._cached_sample_log_prob = sample_log_prob
        return sample

    def log_prob(self, value):
        # We evaluate log_prob as in the original paper. However, we should note that this log_prob does
        # not correspond to the log_prob of the HDRRecalibratedDistribution, but rather to the log_prob of
        # the uncalibrated distribution. The log_prob of the HDRRecalibratedDistribution would be very hard
        # to compute.
        # We cache the log_prob of the last sample to avoid recomputing it.
        if self._cached_sample is value:
            return self._cached_sample_log_prob
        return self.uncal_dist.log_prob(value)


class HDRRecalibrator(torch.nn.Module):
    def __init__(self, model, datamodule, B=20, n_samples_cal=100):
        super().__init__()
        self.model = model
        self.B = B
        self.n_samples_cal = n_samples_cal
        self.scores = self.get_scores(model, datamodule.val_dataloader())
        # Evaluate the empirical CDF of the scores at bounds_levels
        self.scores = torch.sort(self.scores).values
        bounds_levels = torch.linspace(0, 1, B + 1, device=self.scores.device)
        self.R = torch.searchsorted(self.scores, bounds_levels) / len(self.scores)
        assert self.R[0] == 0.0
        self.R[-1] = 1.0
        assert len(self.R) == B + 1

    def get_scores(self, model, dl):
        scores = []
        for x, y in dl:
            x, y = x.to(model.device), y.to(model.device)
            dist = model.predict(x)
            scores.append(1 - hpd(dist, y, n_samples=self.n_samples_cal))
        return torch.cat(scores)

    def predict(self, x):
        uncal_dist = self.model.predict(x)
        return HDRRecalibratedDistribution(uncal_dist, self.R)

    @classmethod
    def output_type(cls):
        return 'distribution'

    @property
    def device(self):
        return self.model.device
