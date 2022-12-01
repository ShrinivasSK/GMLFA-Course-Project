""" Addition by us to incorporate the RLU component """
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############# Train Loop for one Epoch ######################
def train(model, feats, labels, loss_fcn, optimizer, train_loader,predict_prob, args):
    ## Set model to train
    model.train()
    device = labels.device
    outs=[]
    for idx,batch in enumerate(train_loader):
        if len(batch) == 1:
            continue
        
        ## Get features and label embs
        batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        batch_labels = labels[batch].to(device)

        ## Forward pass
        out, _ = model(batch_feats)

        ## Save predictions
        outs.append(out)
        
        ## Cross Entropy Loss
        L1 = loss_fcn(out, batch_labels)

        L3=0
        if predict_prob is not None:
            ########### Reliable Label Utilisation #####################

            ############ Reliable Label Propagation ###################
            confident_idx=torch.arange(len(predict_prob[idx,:]))[
                            predict_prob[idx,:].max(1)[0] > args.threshold]
            confident_prob=predict_prob[idx,:][confident_idx]

            ########### Reliable Label Distillation ###################
            teacher_soft = confident_prob.to(device)
            teacher_prob = torch.max(teacher_soft, dim=1, keepdim=True)[0]
            L3 = (teacher_prob*(teacher_soft*(torch.log(teacher_soft+1e-8)-torch.log_softmax(out, dim=1)))).sum(1).mean()
            
        ## Combined Loss
        loss = L1 + L3*args.gama

        ## Optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    outs = torch.stack(outs)
    
    return outs


############ Evaluation Loop after one epoch #############################
def test(model, feats, labels, loss_fcn, val_loader, test_loader, evaluator,
         train_nid, val_nid, test_nid):
    ## Set model to eval
    model.eval()

    device = labels.device
    
    ## Saving metrics
    loss_list = []
    count_list = []
    preds = []

    ############# Validation ###################
    for batch in val_loader:

        ## get features and labels
        batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        batch_labels = labels[batch].to(device)

        ## Forward pass
        out, _ = model(batch_feats)

        ## Save metrics
        loss_list.append(loss_fcn(out, batch_labels).cpu().item())
        count_list.append(len(batch))

    loss_list = np.array(loss_list)
    count_list = np.array(count_list)
    val_loss = (loss_list * count_list).sum() / count_list.sum()

    ############## Testing ######################
    for batch in test_loader:

        ## get features
        batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
        
        ## Forward pass
        out, _ = model(batch_feats)

        ## Get Predictions
        preds.append(torch.argmax(out, dim=-1))

    ## Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)

    ########### Evaluate #######################
    train_res = evaluator(preds[:len(train_nid)], labels[train_nid])
    val_res = evaluator(preds[len(train_nid):(len(train_nid)+len(val_nid))], labels[val_nid])
    test_res = evaluator(preds[(len(train_nid)+len(val_nid)):], labels[test_nid])

    return train_res, val_res, test_res, val_loss
