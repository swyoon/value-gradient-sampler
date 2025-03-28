import argparse
from tqdm import trange, tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import argparse
import os
from myutils.ebm_utils import evaluation_ebm, get_initial_samples
from myutils.module_utils import print_size
import myutils.cmd_utils as cmd
from tensorboardX import SummaryWriter
from myutils.logger import BaseLogger
from omegaconf import OmegaConf
from hydra.utils import instantiate
import random
import wandb
        

################################
# Example Usage: python train_ebm.py --config configs/mvtec.yaml --run test --device 0
################################



if __name__ == '__main__':
    """
    Example usage: python train_ebm.py --config configs/mvtec.yaml --run test --device 0
    """
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument('--run', type=str)

    args, unknown = parser.parse_known_args()
    d_cmd_cfg = cmd.parse_unknown_args(unknown)
    d_cmd_cfg = cmd.parse_nested_args(d_cmd_cfg)
    print("Overriding", d_cmd_cfg)

    # load configs
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(
        cfg, OmegaConf.create(d_cmd_cfg)
    )  # override with command line arguments
    print(OmegaConf.to_yaml(cfg))

    model_cfg_name = os.path.basename(args.config).split(".")[0]
    logdir = os.path.join(f"results/{model_cfg_name}", args.run)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    # setting seeds
    device = "cuda:{}".format(args.device)

    if 'seed' in cfg:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # instantiate networks
    energy = instantiate(cfg.energy).to(device)
    if cfg.training.get('energy_ckpt', None) is not None:
        energy_dict = torch.load(cfg.training.energy_ckpt)
        energy.load_state_dict(energy_dict, strict=False)
    print('Energy Network')
    print_size(energy)
    sampler = instantiate(cfg.sampler).to(device)
    print('Value Network')
    print_size(sampler.value)

    # Set optimizer
    opt_e = Adam(energy.parameters(), lr=cfg.training.e_lr)

    # load data
    train_data = torch.load('datasets/ebm_exp/train_mvtec.pth', weights_only=False)
    val_data = torch.load('datasets/ebm_exp/val_mvtec.pth', weights_only=False)
    X_train = train_data['feature_align']
    X_train = torch.tensor(X_train.reshape(len(X_train), 272, -1))
    X_train = X_train.permute(0, 2, 1).reshape(-1, 272)
    X_val = val_data['feature_align']
    X_val = torch.tensor(X_val.reshape(len(X_val), 272, -1))
    y_val = torch.tensor(val_data['label'])
    clsname_val = np.array(val_data['clsname'])
    mask_val = val_data['mask']

    # Project data to unit sphere
    X_train = X_train / X_train.norm(dim=1, keepdim=True)
    X_val = X_val / X_val.norm(dim=1, keepdim=True)

    # dataloader
    batchsize = cfg.training.batchsize
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize*196, # Effective batchsize is batchsize * 14 * 14
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle= False, num_workers=4, pin_memory=True)

    # logger
    writer = SummaryWriter(logdir=logdir)
    OmegaConf.save(cfg, os.path.join(logdir, 'config.yaml'))
    print(f'Current config file saved to {os.path.join(logdir, "config.yaml")}')
    logger = BaseLogger(writer, use_wandb=True)
    temp_name = args.config.split('/')[1].split('.')[0]
    wandb.init(project=f'{temp_name}_exp', name=f'{model_cfg_name}_{args.run}', dir=logdir,
        config=OmegaConf.to_container(cfg))

    i_iter = 0
    best_mean_auc = 0
    gamma = cfg.training.gamma if 'gamma' in cfg.training else None
    for epoch in trange(cfg.training.n_epochs):
        print('epoch', epoch)

        # Evaluation
        if cfg.training.ood_epoch is not None and epoch % cfg.training.ood_epoch == 0:
            print('evaluating OOD Detection AUROC')
            energy.eval()
            d_ood_result = evaluation_ebm(device, energy, y_val, clsname_val, val_loader)
            logger.log(d_ood_result, i_iter)
     
        ############################################
        # Main training loop
        ############################################
        for step, input in enumerate(tqdm(train_loader, ncols=80)):
            x = input[0].to(device)
            D = x.shape[1]
            # Sample positive samples
            x_pos = x
            # Sample negative samples
            initial = X_train[torch.randint(len(X_train), (len(x),))].to(device)
            initial = get_initial_samples(initial, sampler.s2)
            d_sample = sampler.sample(x.shape[0], device, energy=energy, initial=initial)
            x_neg = d_sample['sample']
            # update energy
            opt_e.zero_grad()
            energy.train()
            # Scale energy by D in loss calculation
            E_pos = energy(x_pos)/D
            E_neg = energy(x_neg)/D
            loss_e = E_pos.mean() - E_neg.mean()
            reg = (E_pos**2).mean() + (E_neg**2).mean()
            if gamma is not None:
                loss_e += gamma * reg
            loss_e.backward()
            opt_e.step()

            # update value
            d_value = sampler.value_update_step_TD(d_sample, energy=energy, n_update = cfg.buffer.n_update)

            # logging
            if i_iter% cfg.training.log_iter == 0:
                d_energy = {'energy/loss_': loss_e.item(), 'energy/pos_e_scaled_': E_pos.mean().item(), 
                            'energy/neg_e_scaled_': E_neg.mean().item(), 'energy/reg_': reg.item()}
                d_mu = {}
                for t, mu in zip(range(len(d_sample['l_mu'])), d_sample['l_mu']):
                    d_mu[f'mu_norm/mu_{t}_'] = mu.norm(dim=1).mean().item()
                logger.log({**d_value, **d_energy, **d_mu}, i_iter)
            i_iter += 1

    # Save model
    d_other_info = {'epoch': epoch, 'iter': i_iter}
    d_value = {'state_dict': sampler.value.state_dict()}
    d_value.update(d_other_info)
    torch.save(d_value, os.path.join(logdir, f'value.pth')) 
    torch.save({'state_dict': energy.state_dict()}, 
                os.path.join(logdir, f'energy.pth')) 
