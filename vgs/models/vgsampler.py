import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import copy
from myutils.ebm_utils import project_vel, move_on_sphere


class ValueGradientSampler(nn.Module):
    def __init__(self, value, n_step, sample_shape, s2_schedule, s2_start, s2_end, alpha_method = "vp",
                 tau = 1.0, ema=None, v_lr = 1e-4, i_lr = None, clip_energy = None, clip_grad = None, TD_loss = 'l2', D_effective = None,
                 scale_with_D = False, normalize = False):
        """
        value: nn.Module, a value network takes a point and a time step as input and returns a value.
        sample_shape: shape of the samples
        s2: torch.Tensor, the variances of the q_tilde distributions.
        s2_schedule: str, one of 'linear', 'quad', 'sigmoid', 'constant'
        s2_start: float, the initial value of s2
        s2_end: float, the final value of s2
        alpha_method: str, one of 'vp', 've'. 'vp' for variance preserving, 've' for variance exploding
        tau: float, entropy regularization coefficient
        ema: float, EMA parameter for target value update: param.data = param.data * self.ema + param_on.data * (1-self.ema). EMA is not used if None.
        v_lr: float, learning rate for value network
        i_lr: float, learning rate for initial sigma. If None, inital sigma is fixed
        clip_energy: float, clip the output of the energy function. If None, no clipping.
        clip_grad: float, clip the norm of the parameter gradient during sampling. If None, no clipping.
        TD_loss: str, one of 'l2', 'l1', 'smooth_l1'. Loss function for value network update.
        D_effective: This option is for the particle system experiment where the effective dimension of the system is (n_particles-1) * n_dimensions. 
                     If None, D = np.prod(sample_shape) as usual.
        scale_with_D: bool, if True, the value network learnes the 1/D of the theoretical value function. Useful for normalizing the neural network output.
        normalize: bool, whether to project the samples to the unit sphere.
        """
        super().__init__()
        self.value = value
        self.n_step = n_step
        self.sample_shape = sample_shape
        if D_effective is None:
            self.D = np.prod(sample_shape)
        else:
            self.D = D_effective
        s2 = make_beta_schedule(
            schedule=s2_schedule,
            n_timesteps=n_step,
            start=s2_start,
            end=s2_end,
        )
        self.register_buffer('s2', s2)
        if alpha_method == "vp":
            self.register_buffer('alpha', torch.sqrt(1/(1-self.s2)))
            self.register_parameter('init_sigma', nn.Parameter(torch.tensor([1.0]))) # q(\bx_0) when q(\bx_T) is N(0, I)
        elif alpha_method == "ve":
            self.register_buffer('alpha', torch.ones(n_step))
            self.register_parameter('init_sigma', nn.Parameter((1.0 + torch.sum(self.s2)).sqrt())) # q(\bx_0) when q(\bx_T) is N(0, I)
        self.sigma = torch.sqrt(s2) * self.alpha
        self.tau = tau

        if ema is None:
            self.ema = 0 # No EMA
        else:
            self.ema = ema
        self.value_on = copy.deepcopy(value) # Online-updated value network for EMA
        self.opt_v = torch.optim.Adam(self.value_on.parameters(), lr = v_lr)

        if i_lr is not None:
            self.opt_i = torch.optim.Adam([self.init_sigma], lr = i_lr)
        else:
            self.opt_i = None
        
        self.clip_energy = clip_energy
        self.clip_grad = clip_grad
        self.TD_loss = TD_loss
        self.scale_with_D = scale_with_D
        self.normalize = normalize


    def update_value(self):
        for param, param_on in zip(self.value.parameters(), self.value_on.parameters()):
            param.data = param.data * self.ema + param_on.data * (1 - self.ema)
        

    def update_init_sigma(self, n_sample, device):
        self.opt_i.zero_grad()
        z = torch.randn((n_sample, *self.sample_shape)).to(device) * self.init_sigma
        if not self.scale_with_D:
            loss = self.value(z, 0).mean() - self. D * self.tau * torch.log(self.init_sigma)
        else:
            loss = self.value(z, 0).mean() - self.tau * torch.log(self.init_sigma)
        loss.backward()
        self.opt_i.step()
        return None


    def get_loss(self, pred, target):
        if self.TD_loss == 'l2':
            return F.mse_loss(pred, target)
        elif self.TD_loss == 'l1':
            return F.l1_loss(pred, target)
        elif self.TD_loss == 'smooth_l1':
            return F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss function {self.TD_loss}")
              
    def sample(self, n_sample, device, energy=None, noise_scale=1.0, final_noise = True, initial = None):
        """generate samples using Value Gradient Sampler.
        Added noise_scale option for off policy learning
        """
        z = torch.randn((n_sample, *self.sample_shape)).to(device) * self.init_sigma * noise_scale if initial is None else initial
        if self.normalize:
            z = z / z.norm(dim=-1, keepdim=True)
        l_sample = [z.detach()]
        l_grad = []
        l_mu = []
        self.value.eval()
        for t in range(self.n_step):
            z_alpha = self.alpha[t]*z
            z_alpha.requires_grad_(True)
            if t == self.n_step - 1 and energy is not None:    
                E_z = torch.clamp(energy(z_alpha), max = self.clip_energy).sum() if (self.clip_energy is not None) and (self.clip_grad is None) else energy(z_alpha).sum()
            else:
                E_z = self.value(z_alpha, t+1).sum() if not self.scale_with_D else self.value(z_alpha, t+1).sum()  * self.D
            grad_E = grad(E_z, z_alpha)[0]
            step_size = self.s2[t]* (self.alpha[t]**2) * (1/self.tau)
            sigma = self.sigma[t]
            if t == self.n_step - 1 and not final_noise:
                sigma = 0.0
            if not self.normalize:
                mu = grad_E * step_size
                z = z_alpha - mu + torch.randn_like(z) * sigma * noise_scale
            else:
                mu = project_vel(z_alpha, grad_E * step_size)
                noise = project_vel(z_alpha, torch.randn_like(z) * sigma * noise_scale)
                z = move_on_sphere(z_alpha, - mu + noise)
            l_sample.append(z.detach())
            l_grad.append(grad_E.detach())
            l_mu.append(mu.detach())
            
        d_sample = {'sample': z, 'l_sample': l_sample, 'l_grad': l_grad, 'l_mu': l_mu}
        return d_sample
    

    def value_update_step_TD(self, d_sample, energy=None, n_update=1):
        """Temporal difference update of value network.
        EMA update + shuffled timestep update using buffer
        """
        x_seq = d_sample["l_sample"]
        mu_seq = d_sample["l_mu"]
        batchsize = len(x_seq[0])
        n_step = len(x_seq) - 1
        device = x_seq[0].device
        state_buffer = torch.FloatTensor().to(device)
        timestep_buffer = torch.LongTensor().to(device)
        mu_buffer = torch.FloatTensor().to(device)
        for t in range(n_step):
            state_buffer = torch.cat((state_buffer, x_seq[t].detach()))
            timestep_buffer = torch.cat((timestep_buffer, torch.tensor([t] * batchsize).to(device)))
            mu_buffer = torch.cat((mu_buffer, mu_seq[t].detach()))
        for _ in range(n_update):
            permutation = torch.randperm(state_buffer.shape[0])
            n_data = len(permutation)
            d_train = {}
            l_v_loss = []
            l_velocity = []
            for m in range(0, n_data, batchsize):
                self.opt_v.zero_grad()
                self.value.eval()
                indices = permutation[m : m + batchsize]
                state = state_buffer[indices] # (B, D)
                mu = mu_buffer[indices] # (B, D)
                t = timestep_buffer[indices] # (B)
                s2 = self.s2[t] # (B)
                alpha = self.alpha[t] # (B)
                sigma = torch.sqrt(s2) * alpha # (B)
                is_last = (t == self.n_step - 1).float() # (B)
                velocity = 0.5 * (mu **2).sum(dim=-1) / s2 / (alpha**2)  # (B)
                if not self.normalize:
                    next_state = alpha.view(-1,1) * state - mu + torch.randn_like(state) * sigma.view(-1,1) # Resample to break the correlation between samples
                else:
                    next_state = move_on_sphere(alpha.view(-1,1) * state, - mu + project_vel(alpha.view(-1,1) * state, torch.randn_like(state) * sigma.view(-1,1)))
                energy_term = torch.clamp(energy(next_state).squeeze(), max = self.clip_energy) if self.clip_energy is not None else energy(next_state).squeeze()
                value_term = self.value(next_state, t+1).squeeze() if not self.scale_with_D else self.value(next_state, t+1).squeeze() * self.D
                v_xtp1 = energy_term * is_last + value_term * (1 - is_last)
                target = v_xtp1 + self.tau * (velocity - self.D * torch.log(alpha)) # Included all constant terms, Used the deterministic running cost. 
                self.value_on.train()
                v_xt = self.value_on(state, t).squeeze() if not self.scale_with_D else self.value_on(state, t).squeeze() * self.D
                v_loss = self.get_loss(v_xt/self.D, target/self.D) # Normalized by D to prevent the loss from growing with D
                v_loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.value_on.parameters(), self.clip_grad)
                self.opt_v.step()
                l_v_loss.append(v_loss.item())
                l_velocity.append(velocity.mean().item())
        self.update_value() # Since the velocity is calculated based on a fixed target value, we also fix the target value during the for loop.

        if self.opt_i is not None:
            self.update_init_sigma(len(state), state.device)

        v_loss_mean, velocity_mean = np.mean(l_v_loss), np.mean(l_velocity)
        d_train['value/v_loss_avg_'] = v_loss_mean
        d_train['velocity/velocity_avg_'] = velocity_mean
        return d_train
    


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