# Graph ML Course Project : Group 06

This is our project for the course GMLFA: Graph Machine Learning Foundations and Applications.

## Problem Statement
Predicting the category of a product in a multi-class classification setup among 47 top-level categories by using an undirected and unweighted graph which represents an Amazon product co-purchasing network.

## Dataset
**ogbn-products [1]**: The ogbn-products dataset is an undirected and unweighted graph, representing an Amazon product co-purchasing network. Nodes represent products sold on Amazon, and edges between two products indicate that the products are purchased together.

## Why is this project interesting
The ogbn-products dataset is an ideal benchmark dataset for the field to move beyond the extremely small graph datasets and to catalyze the development of scalable mini-batch-based graph models. It also uses a realistic split based on the sales ranking of the product rather than a random split which offers an opportunity to improve out-of-distribution generalization. The project is specifically interesting because a distinct correlation can be established between which category of products is usually bought together by customers. This information can help increase sales of companies and also increase the ease of customer access to correlated products.

## Core Architecture
SAGN [2] (Scalable and Adaptive Graph Neural Networks) is our base Graph Neural Network architecture. This is because it works well for large datasets like the one we have used. It is a very expressive classifier as it uses an inception-like module and uses learnable attention weights. 
On top of this architecture, we plan to implement the Reliable Label Utilization (RLU) module used with the GAMLP [3] ( Graph Attention Multi-Layer Perceptron) architecture to better utilize the predicted soft labels from the classifier. 

## File Structure

- [dataset.py](dataset.py)
    This includes the code for loading the dataset and the evaluator using the OGBN framework
- [layers.py](layers.py)
    This file has the implementation of the layers in the model. 
- [models.py](models.py)
    Here is where we have defined the SAGN architecture
- [pre_process.py](pre_process.py)
    This file contains the pre-processing functions that include the neighbourhood aggregation function for the initial set of features for the nodes
- [train_utils.py](trainn_utils.py)
    Here we have defined the main train and evalutaion loops for one epoch. The loss function has been calculated here
- [train.py](train.py)
    This file has the main train loop of the architecture. The complete pipeline for training is defined here including the multiple stages
- [utils.py](utils.py)
    This file contains some utility functions used for the whole project. 

## Note
This code is loosely based on two github repos [SAGN](https://github.com/skepsun/SAGN_with_SLE.git) and [GAMLP](https://github.com/PKU-DAIR/GAMLP.git). Our work has been coming up a design to incorporate bothe the designs and both extensive code commenting and cleaning of the code. 

## References
1. Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. arXiv. https://doi.org/10.48550/arXiv.2005.00687
2. Chuxiong Sun, & Guoshi Wu (2021). Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training. CoRR, abs/2104.09376.
3. Wentao Zhang, Ziqi Yin, Zeang Sheng, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, & Bin Cui (2021). Graph Attention Multi-Layer Perceptron. CoRR, abs/2108.10097.  (https://arxiv.org/abs/2206.04355)