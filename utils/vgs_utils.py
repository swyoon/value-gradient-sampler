import torch
import numpy as np


def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":  # this is what is used in Ying Fan
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "constant":
        betas = torch.ones(n_timesteps) * start 
    elif schedule == "exp":
        betas = torch.exp(torch.linspace(np.log(start), np.log(end), n_timesteps))
    else:
        raise ValueError("Unknown schedule")
    return betas