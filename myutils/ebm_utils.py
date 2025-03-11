import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F
from pprint import pprint


def get_initial_samples(data, s2):
    """
    Get initial samples from data samples
    The initial samples are obtained by diffusing the data samples on a unit sphere according to the schedule s2
    """
    for i in reversed(range(len(s2))):
        mu = torch.randn_like(data) * torch.sqrt(s2[i])
        data = move_on_sphere(data, mu)
    return data


def move_on_sphere(x: torch.Tensor, v: torch.Tensor):
    """
    Moves points x on the n-1 sphere in R^n by velocity v.
    Assumes x has shape (B, n) and v has shape (B, n).
    
    Args:
        x: (B, n) points on the n-1 sphere.
        v: (B, n) velocity vectors in R^n.
    
    Returns:
        x': (B, n) moved points on the n-1 sphere.
    """
    # Project v onto the tangent space of the sphere at x
    v_tan = v - (torch.sum(v * x, dim=-1, keepdim=True) * x)
    
    # Compute the norm of the tangential velocity
    v_tan_norm = torch.norm(v_tan, dim=-1, keepdim=True)
    
    # Avoid division by zero (if v_tan_norm is 0, no movement occurs)
    eps = 1e-8
    v_tan_norm = torch.clamp(v_tan_norm, min=eps)

    # Compute the new position using the exponential map
    x_new = torch.cos(v_tan_norm) * x + torch.sin(v_tan_norm) * (v_tan / v_tan_norm)

    return x_new


def evaluation_ebm(device, energy, y_val, clsname_val, val_loader):
    pred_y, pred_y_mask = predict(energy, val_loader, device)
    auc = roc_auc_score(y_val, pred_y)    
    in_pred = pred_y[y_val == 0].numpy().mean()
    out_pred = pred_y[y_val == 1].numpy().mean()
    d_ood_result = {'auc/total_auc_': auc, 'ood_score/in_': in_pred, 'ood_score/ood_': out_pred}
    pprint(d_ood_result)
    d_cls_auc = compute_classwise_auc(pred_y, y_val, clsname_val)
    for k, v in d_cls_auc.items():
        d_ood_result[f'auc/{k}_'] = v
    return d_ood_result


def predict(m, data_loader, device):
    l_pred = []
    for xx, _ in data_loader:
        xx = xx.permute(0, 2, 1).reshape(-1, 272)
        xx = xx.to(device)
        with torch.no_grad():
            pred = m(xx).detach().cpu()
        l_pred.append(pred)
    pred_y_ = torch.cat(l_pred, dim=0)
    pred_y = pred_y_.reshape(-1, 14*14)
    pred_y = pred_y.max(axis=1).values

    pred_y_ = pred_y_.reshape(-1, 1, 14, 14)
    pred_y_map = F.interpolate(pred_y_, size=(224, 224), mode='bilinear', align_corners=False).numpy()
    return pred_y, pred_y_map


def compute_classwise_auc(pred_y, y, clsname):
    l_cls = np.unique(clsname)
    d_auc = {}
    l_auc = []
    print('=============')
    for c in l_cls:
        idx = clsname == c
        auc = roc_auc_score(y[idx], pred_y[idx])
        print(f'{c}: {auc:.4f}')
        d_auc[c] = auc
        l_auc.append(auc)
    print(f'mean: {np.mean(l_auc):.4f}')
    print('=============')
    d_auc['mean'] = np.mean(l_auc)
    return d_auc


def compute_classwise_localization_auc(pred_y, mask, clsname):
    l_cls = np.unique(clsname)
    d_auc = {}
    l_auc = []
    print('=============')
    for c in l_cls:
        idx = clsname == c
        auc = roc_auc_score(mask[idx].reshape(-1), pred_y[idx].reshape(-1))
        print(f'{c}: {auc:.4f}')
        d_auc[c] = auc
        l_auc.append(auc)
    print(f'mean: {np.mean(l_auc):.4f}')
    print('=============')
    d_auc['mean'] = np.mean(l_auc)
    return d_auc