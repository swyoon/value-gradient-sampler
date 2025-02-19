"""
Original code from https://github.com/noegroup/bgflow.git, modified for compatibility with dis-distribution
"""
import torch
from utils.particle_utils import compute_distances, remove_mean
import numpy as np

__all__ = ["MultiDoubleWellPotential"]


class MultiDoubleWellPotential(torch.nn.Module):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via

    .. math::
        E_{DW}(d) = a \cdot (d-d_{\text{offset})^4 + b \cdot (d-d_{\text{offset})^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    # Default values set identical to https://github.com/jarridrb/DEM
    def __init__(self, dim = 8, n_particles = 4, a = 0.9, b = -4, c = 0, offset = 4, data_path = None): 
        super().__init__()
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset
        self.stddevs = 1.8165502548217773 # Calculated from all_split_DW4-120k.npy

        if data_path is not None:
            data = np.load(data_path, allow_pickle=True)
            self.data = remove_mean(torch.tensor(data), self._n_particles, self._n_dimensions)
            self.n_data = data.shape[0]
            print(f"Ground truth sample shape: {data.shape}")
        else:
            self.data = None
            self.n_data = 0
            print("No Ground truth sample provided")

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._n_dimensions)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)

    def unnorm_log_prob(self, x):
        return -self._energy(x)
    
    def sample(self, shape: tuple):
        assert len(shape) == 1, "This implementation only supports sampling a single batch"
        assert self.data is not None, "No ground truth sample provided"
        n_samples = shape[0]
        index = np.random.choice(self.n_data, n_samples, replace=False)
        return self.data[index]
    
    def to(self, device):
        super().to(device)
        if self.data is not None:
            self.data = self.data.to(device)
        return self



