import torch
import torch.nn as nn
from myutils.particle_utils import compute_distances


def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


class FCNet(nn.Module):
    """fully-connected network"""
    def __init__(self,
            in_dim,
            out_dim,
            l_hidden=(50,),
            activation='sigmoid',
            out_activation='linear',
            use_spectral_norm=False,
            flatten_input=False,
            batch_norm=False,
            out_batch_norm=False,
            learn_out_scale=False,
            bias=True):
        super().__init__()
        l_neurons = tuple(l_hidden) + (out_dim,)
        if isinstance(activation, str):
            activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            if use_spectral_norm and i_layer < len(l_neurons) - 1:  # don't apply SN to the last layer
                l_layer.append(P.spectral_norm(nn.Linear(prev_dim, n_hidden)))
            else:
                l_layer.append(nn.Linear(prev_dim, n_hidden, bias=bias))
            if batch_norm:
                if out_batch_norm:
                    l_layer.append(nn.BatchNorm1d(num_features=n_hidden))
                else:
                    if i_layer < len(l_neurons) - 1:  # don't apply BN to the last layer
                        l_layer.append(nn.BatchNorm1d(num_features=n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        # add learnable scaling operation at the end
        if learn_out_scale:
            l_layer.append(nn.Linear(1, 1, bias=True))

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim
        self.out_shape = (out_dim,)
        self.flatten_input = flatten_input

    def forward(self, x):
        if self.flatten_input and len(x.shape) == 4:
            x = x.view(len(x), -1)
        return self.net(x)

        
class invariant_wrapper(nn.Module):

    def __init__(self, n_particles, n_dim, net, dis_reciprocal=False, eps = 1.0):
        super().__init__()
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.net = net # input_dim should be n_particles * (n_particles-1) // 2
        self.dis_reciprocal = dis_reciprocal
        self.eps = eps

    def forward(self, x, t):
        """
        x: torch.Tensor of shape [Batch,  n_particles * n_dimensions]
        t: int
        """
        distances = compute_distances(x, self.n_particles, self.n_dim, remove_duplicates=True) # [Batch, n_particles * (n_particles-1) // 2]
        distances, _ = torch.sort(distances, dim=-1, descending=True)
        if self.dis_reciprocal:
            input = 1 / (distances ** 2 + self.eps).sqrt() # To avoid division by zero 
        else:
            input = (distances ** 2 + self.eps).sqrt() # to avoid near-zero distances
        return self.net(input, t)
    

class AE_energy(nn.Module):
    """Uses Autoencoder architecture for modeling energy functions"""

    def __init__(
        self,
        encoder,
        decoder,
        tau=1.0,
        learn_out_scale=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tau = tau  # Temperature constraint
        self.learn_out_scale = learn_out_scale
        if learn_out_scale:
            self.register_parameter("out_scale", nn.Parameter(torch.tensor(1.0)))

    def forward(self, x):
        z = self.encoder(x)
        # Normalize z
        z = z / torch.norm(z, dim=1, keepdim=True)
        recon = self.decoder(z)
        # Normalize recon
        recon = recon / torch.norm(recon, dim=1, keepdim=True)
        recon_error = ((x - recon) ** 2).view(len(x), -1).sum(dim=1)
        out = recon_error if not self.learn_out_scale else (self.out_scale ** 2) * recon_error
        out = out / self.tau
        return out