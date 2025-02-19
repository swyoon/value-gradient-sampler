import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import FCNet

"""
Modules adopted from Diffusion Recovery Likelihood related experiments
"""

def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = timesteps.type(dtype=torch.float)[:, None] * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.pad(emb, [0, 1], value=0.0)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def process_single_t(x, t):
    """make single integer t into a vector of an appropriate size"""
    if isinstance(t, int) or len(t.shape) == 0 or len(t) == 1:
        t = torch.ones([x.shape[0]], dtype=torch.long, device=x.device) * t
    return t


class FCNet_temb(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim=128, t_emb_dim=32, activation="relu", spec_norm=False,
    ):
        super().__init__()
        self.net1 = FCNet(
            in_dim=in_dim,
            out_dim=hidden_dim,
            l_hidden=[],
            activation=activation,
            out_activation="linear",
            use_spectral_norm=spec_norm
        )
        self.net2 = FCNet(
            in_dim=t_emb_dim,
            out_dim=hidden_dim,
            l_hidden=(hidden_dim, hidden_dim),
            activation=activation,
            out_activation="linear",
            use_spectral_norm=spec_norm
        )
        self.net3 = FCNet(
            in_dim=2 * hidden_dim,
            out_dim=out_dim,
            l_hidden=(hidden_dim, hidden_dim),
            activation=activation,
            out_activation="linear",
        )
        self.t_emb_dim = t_emb_dim

    def forward(self, x, t, class_labels=None):
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        x_ = self.net1(x)
        t = process_single_t(x, t)
        t_emb = get_timestep_embedding(t, self.t_emb_dim).to(x.device)
        t_emb = self.net2(t_emb)
        x_ = torch.cat([x_, t_emb], dim=1)
        return self.net3(x_)
    
    
class FCNet_temb_deeper(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim=1024, t_emb_dim=128, t_feature_dim = 512,
        output_same_dim=False, residual = False 
    ):
        super().__init__()
        # fc layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

        # temb layers
        self.temb = nn.Linear(t_emb_dim, t_feature_dim)
        self.temb1 = nn.Linear(t_feature_dim, hidden_dim)
        self.temb2 = nn.Linear(t_feature_dim, hidden_dim)

        self.t_emb_dim = t_emb_dim
        self.output_same_dim = output_same_dim
        self.residual = residual

    def forward(self, x, t, class_labels=None):
        """x is either (batch, feature_dim) or (batch, feature_dim, 1, 1)"""
        if len(x.shape) == 4:
            is_flatten = False
            x = x.squeeze(2).squeeze(2)
        else:
            is_flatten = True

        if self.residual:
            x_init = x

        # get timestep embedding
        t = process_single_t(x, t)
        t_emb = get_timestep_embedding(t, self.t_emb_dim).to(x.device)
        t_emb = self.temb(t_emb)

        x_ = self.fc1(x)
        x_ = F.relu(x_) + F.relu(self.temb1(t_emb))
        x_ = self.fc2(x_)
        x_ = F.relu(x_)
        x_ = self.fc3(x_)
        x_ = F.relu(x_) + F.relu(self.temb2(t_emb))
        x_ = self.fc4(x_)
        x_ = x_ + x_init if self.residual else x_

        if not is_flatten and self.output_same_dim:
            x_ = x_.unsqueeze(2).unsqueeze(3)
        return x_


class FCNet_temb_deeperv2(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim=1024, t_emb_dim=128, 
        output_same_dim=False, residual = False 
    ):
        super().__init__()
        # fc layers
        self.fc1 = nn.Linear(in_dim + t_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

        # # temb layers
        # self.temb = nn.Linear(t_emb_dim, t_feature_dim)
        # self.temb1 = nn.Linear(t_feature_dim, hidden_dim)
        # self.temb2 = nn.Linear(t_feature_dim, hidden_dim)

        self.t_emb_dim = t_emb_dim
        self.output_same_dim = output_same_dim
        self.residual = residual

    def forward(self, x, t, class_labels=None):
        """x is either (batch, feature_dim) or (batch, feature_dim, 1, 1)"""
        if len(x.shape) == 4:
            is_flatten = False
            x = x.squeeze(2).squeeze(2)
        else:
            is_flatten = True

        if self.residual:
            x_init = x

        # get timestep embedding
        t = process_single_t(x, t)
        t_emb = get_timestep_embedding(t, self.t_emb_dim).to(x.device)
        x_ = torch.cat([x, t_emb], dim=1)
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        x_ = self.fc2(x_)
        x_ = F.relu(x_)
        x_ = self.fc3(x_)
        x_ = F.relu(x_)
        x_ = self.fc4(x_)
        x_ = x_ + x_init if self.residual else x_
        
        if not is_flatten and self.output_same_dim:
            x_ = x_.unsqueeze(2).unsqueeze(3)
        return x_


class FCNet_temb_deeperv3(nn.Module):
    """
    New FCNet_temb_deeper architecture inspired by models/wideresnet_te/wideresnet_te.py from GCD repo
    """
    def __init__(
        self, in_dim, out_dim, hidden_dim=1024, t_emb_dim=128, t_feature_dim = 512,
    ):
        super().__init__()
        # fc layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

        # temb layers
        self.temb = nn.Linear(t_emb_dim, t_feature_dim)
        self.temb1 = nn.Linear(t_feature_dim, hidden_dim)
        self.temb2 = nn.Linear(t_feature_dim, hidden_dim)

        self.t_emb_dim = t_emb_dim


    def forward(self, x, t):
        """x is (batch, feature_dim)"""
        # get timestep embedding
        t = process_single_t(x, t)
        t_emb = get_timestep_embedding(t, self.t_emb_dim).to(x.device)
        t_emb = self.temb(t_emb)

        x_ = self.fc1(x)
        x_ = torch.cat((x_, self.temb1(t_emb)), dim = 1)
        x_ = F.relu(x_)
        x_ = self.fc2(x_)
        x_ = F.relu(x_)
        x_ = self.fc3(x_)
        x_ *= self.temb2(t_emb)
        x_ = self.fc4(x_)

        return x_