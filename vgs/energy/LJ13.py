import torch
from utils.particle_utils import distances_from_vectors, distance_vectors, remove_mean
import numpy as np

def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    p = 0.9
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential(torch.nn.Module):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        energy_factor=1.0,
        data_path=None,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        super().__init__()
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor
        self.stddevs = 0.6807141304016113 # Calculated from all_split_LJ13-120k.npy
        if data_path is not None:
            data = np.load(data_path, allow_pickle=True)
            self.data = remove_mean(torch.tensor(data), self._n_particles, self._n_dims)
            self.n_data = data.shape[0]
            print(f"Ground truth sample shape: {data.shape}") 
        else:
            self.data = None
            self.n_data = 0
            print("No Ground truth sample provided")


    def _energy(self, x):
        batch_shape = x.shape[0]
        x = x.view(batch_shape, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = lj_energies.view(batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(batch_shape)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

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