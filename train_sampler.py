import sys
import argparse
from tqdm import trange
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import argparse
from vgs.eval.metrics import *
import utils.cmd_utils as cmd
from tensorboardX import SummaryWriter
from utils.logger import BaseLogger
from omegaconf import OmegaConf
from hydra.utils import instantiate
import wandb
from utils.particle_utils import remove_mean
from utils.module_utils import *


################################
# Example Usage: python train_sampler.py --config configs/gmm.yaml --device 1 --run test --exp_num 0 
# Example Usage(Funnel) : python train_sampler.py --config configs/funnel_temp.yaml --device 0 --run funnel_20 -exp_num 1
################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--device', type=str, default='0', help='gpu id')
    parser.add_argument('--run', type=str, required=True, help='name of the run')
    parser.add_argument('--save_training_visualization', type=bool, default=False, help='Whether to save value visualization while training')
    parser.add_argument('--exp_num', type=int, default=0, help='experiment index')
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = cmd.parse_unknown_args(unknown)
    d_cmd_cfg = cmd.parse_nested_args(d_cmd_cfg)
    print("Overriding", d_cmd_cfg)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(
        cfg, OmegaConf.create(d_cmd_cfg)
    )  # override with command line arguments
    print(OmegaConf.to_yaml(cfg))

    model_cfg_name = os.path.basename(args.config).split(".")[0]
    logdir = os.path.join(f"results/{model_cfg_name}", args.run)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    device = "cuda:{}".format(args.device)
  
    energy_model = instantiate(cfg.energy_model)
    print(energy_model)
    energy_model.to(device)
    energy = lambda x: -energy_model.unnorm_log_prob(x).float().to(device)
    sampler = instantiate(cfg.sampler).to(device)
    if "sampler_ckpt" in cfg.training and cfg.training.sampler_ckpt is not None:
        sampler.load_state_dict(torch.load(cfg.training.sampler_ckpt), strict = True) # Strict is easier to debug
    print('Value Network')
    print_size(sampler.value)

    noise_scale = cfg.off_policy.noise_scale

    n_step = sampler.n_step
    d_v_loss = {t:[] for t in range(n_step)}
    l_v_loss = []

    writer = SummaryWriter(logdir=logdir)
    OmegaConf.save(cfg, os.path.join(logdir, 'config.yaml'))
    print(f'Current config file saved to {os.path.join(logdir, "config.yaml")}')
    logger = BaseLogger(writer, use_wandb=True)
    temp_name = args.config.split('/')[1].split('.')[0]
    wandb.init(project=f'{temp_name}_exp', name=f'{model_cfg_name}_{args.run}', dir=logdir,
        config=OmegaConf.to_container(cfg))

    n_iter = cfg.training.n_iter
    batch_size = cfg.training.batch_size
    log_iter = cfg.training.log_iter
    save_iter = cfg.training.save_iter if 'save_iter' in cfg.training else log_iter
    val_iter = cfg.training.val_iter if 'val_iter' in cfg.training else log_iter
    is_particle_exp = cfg.is_particle_exp if 'is_particle_exp' in cfg else False
    final_noise = cfg.final_noise if 'final_noise' in cfg else True
    
    if is_particle_exp:
        best_tvd_e = np.inf

    for i_iter in trange(n_iter):
        # update value
        d_sample = sampler.sample(batch_size, device, energy=energy, noise_scale=noise_scale)

        d_train = sampler.value_update_step_TD_buffer(d_sample, energy=energy, n_update = cfg.buffer.n_update)
        
        if i_iter % log_iter == 0:
            d_sample = sampler.sample(batch_size, device, energy = energy)
            x = d_sample['sample']
            e_x = energy(x).detach()
            d_sample_energy = {"sample_energy/min_": e_x.min().item(), "sample_energy/median_": e_x.median().item(),
                                "sample_energy/mean_": e_x.mean().item(), "sample_energy/max_": e_x.max().item()}
            d_others = {"others/init_sigma_": sampler.init_sigma.item(), "others/weight_norm_": weight_norm(sampler.value).item()}

            logger.log({**d_train, **d_sample_energy, **d_others}, i_iter)
        
        if i_iter % val_iter == 0:
            d_eval = evaluation(sampler, energy_model, energy, device, int(cfg.training.n_eval), is_particle_exp, final_noise)
            if is_particle_exp:
                if d_eval["eval/tvd_e_"] < best_tvd_e:
                    best_tvd_e = d_eval["eval/tvd_e_"]
                    torch.save(sampler.state_dict(), os.path.join(logdir, f"exp{args.exp_num}_best_{best_tvd_e}.pth"))
                    print(f"Best TVD-E : {best_tvd_e}")
            logger.log({**d_eval}, i_iter)

        if i_iter % save_iter:
            """visualize (x, y) of the first particle"""
            fig, axs = plt.subplots(ncols=6, figsize=(4*6,4))
            for i_ax, ax in enumerate(axs):
                sample = d_sample['l_sample'][i_ax*(n_step//5)].detach().cpu().numpy()
                if is_particle_exp:
                    sample = remove_mean(sample, sampler.value.n_particles, sampler.value.n_dim)
                ax.plot(sample[:,0], sample[:,1], '.')
                ax.set_title(f't={i_ax*(n_step//5)}')
                if i_ax == 5:
                    real_samples = energy_model.sample((batch_size,)).detach().cpu().numpy()
                    ax.plot(real_samples[:,0], real_samples[:,1], '.')
            plt.tight_layout()
            wandb.log({"sample": wandb.Image(fig)}, step=i_iter)
            plt.close(fig)

    # Save the final results and model
    d_eval = evaluation(sampler, energy_model, energy, device, int(cfg.training.n_eval), is_particle_exp, final_noise)
    file_path = os.path.join(logdir, f"results_{args.exp_num}.txt")
    with open(file_path, 'a') as file:
        for key, value in d_eval.items():
            key = key[:-1] # remove the last underscore
            key = key.replace("eval/", "")
            file.write(f"{key} : {value}\n")
    ckpt = os.path.join(logdir, f"exp{args.exp_num}.pth")
    torch.save(sampler.state_dict(), ckpt)


