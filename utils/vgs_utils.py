import torch


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
    elif schedule == "cum_geometric":
        cum_betas = start * (end / start) ** (torch.linspace(0, 1, n_timesteps))
        cum_betas = torch.cat([cum_betas, torch.zeros(1)])
        betas = cum_betas[:-1] - cum_betas[1:]
    return betas