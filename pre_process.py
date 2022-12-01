""" Adopted and Modified from https://github.com/skepsun/SAGN_with_SLE repo """
import dgl.function as fn
import numpy as np
import torch

from dataset import load_dataset
from utils import (inner_distance, outer_distance)

""" Adopted from the Github Repo """
def neighbor_average_features(g, feat, args):
    ##############################################
    ## Compute multi-hop neighbor-averaged node features
    ##############################################
    
    aggr_device = torch.device("cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu))
    g = g.to(aggr_device)
    feat = feat.to(aggr_device)
    idx = torch.arange(len(feat)).to(aggr_device)
        
    feat_0 = feat.clone()
    train_mask = g.ndata["train_mask"]
    print(f"hop 0: outer distance {outer_distance(feat_0, feat_0, train_mask):.4f}, inner distance {inner_distance(feat_0, train_mask):.4f}")
    init_feat = feat
    
    #################### Average over the hops #########################
    for hop in range(1, args.label_K+1):         
        g.ndata['f'] = feat
        g.update_all(fn.copy_src(src='f', out='msg'),
                    fn.mean(msg='msg', out='f'))
        feat = g.ndata.pop('f')
        feat = 0.5 * feat + 0.5 * init_feat
        
        print(f"hop {hop}: outer distance {outer_distance(feat_0, feat, train_mask):.4f}, inner distance {inner_distance(feat, train_mask):.4f}")

        res = feat[idx].clone()
    
    return res

""" Modified by us to remove the SLE component """
def prepare_data(device, args):
    ####################################################
    ## Load dataset and compute neighbor-averaged node features used by scalable GNN model
    ## Note that we select only one integrated representation as node feature input for mlp 
    ####################################################

    aggr_device = torch.device("cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu))

    ################ Load Dataset ###############
    ## Might take some time
    data = load_dataset(aggr_device, args)
    
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data

    train_mask = torch.BoolTensor(np.isin(np.arange(len(labels)), train_nid))
    g.ndata["train_mask"] = train_mask.to(aggr_device)
    
    in_feats = g.ndata['feat'].shape[1]
    feat = g.ndata.pop('feat')

    ############# Calculate Neighbour Averaged Features #########################
    feats = neighbor_average_features(g, feat, args)

    ## Bring to Device
    labels = labels.to(device)
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)

    return feats, labels, in_feats, n_classes, \
        train_nid, val_nid, test_nid, evaluator
