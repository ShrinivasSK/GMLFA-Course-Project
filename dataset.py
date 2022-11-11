from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

############## Load Evaluator from OGB ###############
def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]

############ Load dataset from OGB ##################
def load_dataset(device, args):
    """
    Load dataset and move graph and features to device
    """
    dataset = DglNodePropPredDataset(name=args.dataset, root=args.data_dir)
    splitted_idx = dataset.get_idx_split()
    ## Using subset of the dataset
    train_nid = splitted_idx["train"][:int(0.5*len(splitted_idx["train"]))]
    val_nid = splitted_idx["valid"][:int(0.5*len(splitted_idx["valid"]))]
    test_nid = splitted_idx["test"][:int(0.5*len(splitted_idx["test"]))]
    g, labels = dataset[0]
    n_classes = dataset.num_classes
    g = g.to(device)
    
    g.ndata['feat'] = g.ndata['feat'].float()

    labels = labels.squeeze()

    evaluator = get_ogb_evaluator('ogbn-products')

    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator