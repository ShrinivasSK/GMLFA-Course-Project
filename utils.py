import random

import dgl
import numpy as np
import torch
import torch.nn.functional as F

eps = 1e-9

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def outer_distance(y1, y2, train_mask):
    y1[y1 == 0] = eps
    y2[y2 == 0] = eps
    y1 = F.normalize(y1, p=1, dim=1)
    y2 = F.normalize(y2, p=1, dim=1)
    d = (y1[train_mask] * torch.log(y1[train_mask]) - y1[train_mask] * torch.log(y2[train_mask])).sum(dim=-1).mean(0)
    return d

def inner_distance(y, train_mask):
    y[y == 0] = eps
    y = F.normalize(y, p=1, dim=1)
    d = (y[train_mask] * torch.log(y[train_mask])).sum(dim=-1).mean(0) 
    if (~train_mask).sum()>0:
        d = d - (y[~train_mask] * torch.log(y[~train_mask])).sum(dim=-1).mean(0)
    return d