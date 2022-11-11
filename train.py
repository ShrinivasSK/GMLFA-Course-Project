import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SAGN
from pre_process import prepare_data
from train_utils import test, train
from utils import get_n_params, seed

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run(args, data, device):
    ############### Saving Metrics #################
    best_epoch=0
    best_val = 0
    best_val_loss = 1e9
    best_test = 0
    val_accs = []
    val_loss = []
    predict_probs=None
    
    ############# Initialize Model ####################
    model = SAGN(in_feats, args.num_hidden, n_classes, args.num_hops,
                            args.mlp_layer, args.num_heads, 
                            weight_style=args.weight_style,
                            dropout=args.dropout, 
                            input_drop=args.input_drop, 
                            attn_drop=args.attn_drop,
                            zero_inits=args.zero_inits,
                            position_emb=args.position_emb,
                            focal=args.focal)

    model = model.to(device)
    print("# Params:", get_n_params(model))

    ############### Loss Function ##################
    loss_fcn = nn.CrossEntropyLoss()

    ############### Optimizer ######################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)

    feats, labels, in_feats, n_classes, \
        train_nid, val_nid, test_nid, evaluator, _ = data

    ############# Dataloaders #####################
    train_loader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_nid, batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        torch.cat([train_nid, val_nid, test_nid], dim=0), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)

    ############ Running for multiple stages #################
    for stage in range(1, args.num_stages +1):
        print("Stage ",stage)
        ################## Main Train Loop ###############
        for epoch in range(1, args.num_epochs + 1):
            print("Epoch ",epoch)

            ## Running one train epoch
            predict_probs = train(model, feats, labels, loss_fcn, optimizer, train_loader,predict_probs,args)

            ## Evaluate every K epochs
            if epoch % args.eval_every == 0:
                with torch.no_grad():
                    acc = test(model, feats, labels, loss_fcn, val_loader, test_loader, evaluator,
                            train_nid, val_nid, test_nid)

                ## Save the best metrics
                if (acc[1] > best_val) or (acc[3] < best_val_loss):
                    best_epoch = epoch
                    best_val = acc[1]
                    best_test = acc[2]
                    best_val_loss = acc[3]

                ## Saving all vall accuracies and losses
                val_accs.append(acc[1])
                val_loss.append(acc[-2])
                ## Print Logs
                log += "Best Val loss: {:.4f}, Accs: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best Val: {:.4f}, Best Test: {:.4f}".format(best_val_loss, acc[0], acc[1], acc[2], best_val, best_test)
                print(log)
    
    return best_val, best_test, val_accs, val_loss, best_epoch


def main(args):
    device = torch.device("cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu))

    print("-" * 100)
    print(f"Starting Training")
    seed(seed=args.seed)

    with torch.no_grad():
        data = prepare_data(device, args)

    best_val, best_test, val_acc, val_loss, best_epoch = run(args, data, device)

    print("Best Val Results : "+str(best_val)+" at Epoch "+str(best_epoch))
    print("Best Test Results: ",best_test)
    print("Val Accuracy: ",val_acc)
    print("Val Loss: ",val_loss)


def define_parser():
    parser = argparse.ArgumentParser(description="Scalable Adaptive Graph neural Networks with Self-Label-Enhance")

    ## Data Arguments
    parser.add_argument("--dataset", type=str, default="ppi")
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ssd/dataset")

    ## Model Arguments
    parser.add_argument("--zero-inits", action="store_true", 
                        help="Whether to initialize hop attention vector as zeros")
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--weight-style", type=str, default="attention")
    parser.add_argument("--focal", type=str, default="first")
    parser.add_argument("--mag-emb", action="store_true")
    parser.add_argument("--position-emb", action="store_true")
    parser.add_argument("--label-residual", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--input-drop", type=float, default=0.2,
                        help="dropout on input features")
    parser.add_argument("--attn-drop", type=float, default=0.4,
                        help="dropout on hop-wise attention scores")
    parser.add_argument("--label-drop", type=float, default=0.5)
    parser.add_argument("--mlp-layer", type=int, default=2,
                        help="number of MLP layers")
    parser.add_argument("--label-mlp-layer", type=int, default=4,
                        help="number of label MLP layers")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.9, 0.9],
                        help="threshold used to generate pseudo hard labels")

    ## Preprocessing Arguments
    parser.add_argument("--label-K", type=int, default=9,
                        help="number of label propagation hops")

    ## Training Arguments
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_epochs", nargs='+',type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="evaluation batch size")

    return parser

if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    print(args)
    main(args)


