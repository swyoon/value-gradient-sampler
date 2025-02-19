from __future__ import annotations

import logging
import ot as pot
import math
import torch
import pykeops.torch as keops
import tqdm
import numpy as np
from vgs.energy.base import Distribution
from utils.particle_utils import remove_mean
from .sinkhorn import *


def evaluation(sampler, energy_model, energy, device, n_eval, is_particle_exp = False, final_noise = True):
    d_sample = sampler.sample(n_eval, device, energy=energy, final_noise=final_noise)
    samples=d_sample['sample']
    energy_samples = energy_model.sample((n_eval,))
    marginal_dims = [0,1]
    if is_particle_exp:
        n_particles, n_dim = sampler.value.n_particles, sampler.value.n_dim
        samples = remove_mean(samples, n_particles, n_dim)
        w2 = wasserstein(energy_samples, samples)
        print(f"Wasserstein distance : {w2}")
        tvd_e = Energy_TVD_particle(samples, energy_samples, energy)
        print(f"TVD-E : {tvd_e}")
        tvd_d = Atomic_TVD_particle(samples, energy_samples, n_particles, n_dim)
        print(f"TVD-D : {tvd_d}")
        d_eval = { "eval/wasserstein_dist_": w2, "eval/tvd_e_": tvd_e, "eval/tvd_d_": tvd_d}
        return d_eval
    else:
        delta_stdev = marginal_stddev(distr=energy_model,samples=samples,marginal_dims=marginal_dims)
        
        print(f"delta_stdev : {delta_stdev}")
        try:
            sinkhorn_dist = Sinkhorn().compute(samples, energy_samples)
        except:
            print("Sinkhorn failed, evaluating with Sinkhorn_pytorch")
            sinkhorn_dist = Sinkhorn_pytorch().compute(samples, energy_samples)
        print(f"Sinkhorn distance : {sinkhorn_dist[0].item()}")
        tvd_e = Energy_TVD(samples, energy_samples, energy)
        print(f"TVD-E : {tvd_e}")
        d_eval = {"eval/delta_stdev_": delta_stdev, "eval/sinkhorn_dist_": sinkhorn_dist[0].item(), "eval/tvd_e_": tvd_e}
        return d_eval
    

# Evaluation Metrics 

def marginal_stddev(
    distr: Distribution,
    samples: torch.Tensor,
    marginal_dims: list[int] | None = None,
) -> dict[str, float]:
    
    if not all(d < distr.dim for d in marginal_dims):
        logging.warning("Removing non-existent marginal dims for metrics.")
        marginal_dims = [d for d in marginal_dims if d < distr.dim]

    marginal_dims = marginal_dims or []

    stddevs = samples.std(dim=0)

    distr.compute_stats_sampling()

    if distr.stddevs is not None:
        assert distr.stddevs.shape == stddevs.shape
        avg_marginal_stddev = (stddevs - distr.stddevs).abs().mean().item()

    return avg_marginal_stddev


def Energy_TVD(gen_sample, gt_sample, energy):
    gt_energy = energy(gt_sample).detach().cpu()
    gen_energy = energy(gen_sample).detach().cpu()
    H_data_set, x_dataset = torch.histogram(gt_energy, bins=200)
    H_gen_samples, _ = torch.histogram(gen_energy, bins=(x_dataset))
    tv = 0.5*(torch.abs(H_data_set/H_data_set.sum() - H_gen_samples/H_gen_samples.sum())).sum()
    return tv


def wasserstein(x0, x1): # Used the same code as in the DEM repository "dem/models/components/optimal_transport.py"
    M = torch.cdist(x0, x1)
    M = M**2
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    ret = pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    return math.sqrt(ret)


def interatomic_dist(x, n_particles, n_dim):
    batchsize = x.shape[0]
    x = x.view(batchsize, n_particles, n_dim)
    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,]
    dist = torch.linalg.norm(distances, dim=-1)
    return dist


def Energy_TVD_particle(gen_sample, gt_sample, energy):
    gt_energy = energy(gt_sample).detach().cpu()
    gen_energy = energy(gen_sample).detach().cpu()
    return total_variation_distance(gen_energy, gt_energy)


def Atomic_TVD_particle(gen_sample, gt_sample, n_particles, n_dim):
    gt_interatomic = interatomic_dist(gt_sample, n_particles, n_dim).detach().cpu()
    gen_interatomic = interatomic_dist(gen_sample, n_particles, n_dim).detach().cpu()
    return total_variation_distance(gen_interatomic, gt_interatomic)


""" Evaluation code from https://github.com/jiajunhe98/DiKL, DiKL/train_utils.py"""
def total_variation_distance(samples1, samples2, bins=200):
    min_ = -100
    max_ = 100
    # Create histograms of the two sample sets
    hist1, bins = np.histogram(samples1, bins=bins, range=(min_, max_))
    hist2, _ = np.histogram(samples2, bins=bins, range=(min_, max_))

    if sum(hist1) / samples1.shape[0] < 0.6: #  in case that the samples are outside [min, max]
        return 1e10
    
    # Normalize histograms to get probability distributions
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate the Total Variation distance
    tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))
    
    return tv_distance