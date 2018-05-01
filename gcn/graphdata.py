import numpy as np
import random
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def get_data(split='train'):
    
    assert split in ['train','test']    
    gsize = 10
    sgsize = 3
    
    features = []
    adj = np.zeros((gsize,gsize))
    labels = []
    if split == 'train':
        sample_size = 100
    else:
        sample_size = 20

    mu1, sigma1 = 1, 2.0/gsize
    mu2, sigma2 = 5, 2.0/sgsize

    for i in range(sample_size):
        s1 = np.random.normal(mu1,sigma1,gsize)
        idxs = random.sample(range(gsize),sgsize)
        s2 = np.zeros(gsize)
        for j in idxs:
            s1[j] = np.random.normal(mu2,sigma2)
            s2[j] = 1
        features.append(s1)
        labels.append(s2)
    
    features = np.array(features,dtype=np.float)
    labels = np.array(labels,dtype=np.int)
    
    adj[0][1] = adj[1][0] = 1
    adj[0][2] = adj[2][0] = 1
    adj[0][9] = adj[9][0] = 1
    
    adj[1][8] = adj[8][1] = 1
    
    adj[2][6] = adj[6][2] = 1

    adj[3][9] = adj[9][3] = 1
    adj[3][4] = adj[4][3] = 1
    adj[3][5] = adj[5][3] = 1
    
    adj[4][9] = adj[9][4] = 1
    
    adj[5][6] = adj[6][5] = 1
    adj[5][8] = adj[8][5] = 1
        
    adj[7][8] = adj[8][7] = 1
    
    # features = sp.vstack(features).tolil()

    if split == 'train':
           return labels, features, adj
    else:
        return labels, features, adj

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


labels, features, adj = get_data('train')
features = preprocess_features(features)
# f = features.tolist()
print(features[0])