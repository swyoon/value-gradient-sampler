import numpy as np
import torch


def weight_norm(net):
    """computes L2 norm of weights of parameters"""
    norm = 0
    for param in net.parameters():
        norm += (param ** 2).sum()
    return norm

def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)