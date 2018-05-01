import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
import arff  # pip install liac-arff
import torchfile  # pip install torchfile
import os
import random

from gcn.utils import *

__author__ = "Michael Gygli, ETH Zurich"
dir_path = os.path.dirname(os.path.realpath(__file__))

def evaluate_f1(predictor, features, labels):
    """Compute the F1 performance of a predictor on the given data."""
    mean_f = []
    fp = open("f1-data.txt","w")
    for idx, (feature, lbl) in enumerate(zip(features, labels)):
        pred_lbl = predictor(feature)
#        print lbl
#        print pred_lbl
        f1 = f1_score(lbl, pred_lbl)
        mean_f.append(f1)
        if idx % 100 == 0:
            print ("%.3f (%d of %d)" % (np.mean(mean_f), idx, len(features)))
            fp.write(str(np.mean(mean_f)) + '\n')
    print ("%.3f" % (np.mean(mean_f)))
    fp.write(str(np.mean(mean_f)) + '\n')    
    fp.close()
    return np.mean(mean_f)

def evaluate_IOU(predictor, features, labels):
    """Compute the IOU performance of a predictor on the given data."""
    mean_iou = []
    fp = open("iou-data.txt","w")
    for idx, (feature, lbl) in enumerate(zip(features, labels)):
        pred_lbl = predictor(feature)
#        print lbl
#        print pred_lbl
        pred_lbl = np.array(pred_lbl, np.int)        
        intersect = np.sum(np.min([pred_lbl, lbl], axis=0))
        union = np.sum(np.max([pred_lbl, lbl], axis=0))
        iou = intersect / float(max(10 ** -8, union))
        mean_iou.append(iou)
        if idx % 100 == 0:
            print ("%.3f (%d of %d)" % (np.mean(mean_iou), idx, len(features)))
            fp.write(str(np.mean(mean_iou)) + '\n')
    print ("%.3f" % (np.mean(mean_iou)))
    fp.write(str(np.mean(mean_iou)) + '\n')    
    fp.close()
    return np.mean(mean_iou)

def get_APDM():
    #APDM Input Graph
    #subgraph size: 301
    #features size: 12527
    #labels size: 12527
    #dataSource = WaterData
    f = open('water_03.txt')
    lineStr = f.readline().strip()
    
    while lineStr:
        lineStr = f.readline().strip()
        
        if lineStr.startswith('#'):
            continue
    
        if lineStr.startswith('numNodes'):
            tokens = lineStr.split(' ')
            numNodes = int(tokens[2])
            features = np.zeros((1,numNodes))
            labels = np.zeros((1,numNodes),dtype = int)
            continue
    
        if lineStr.startswith('SECTION2'):
            lineStr = f.readline().strip() # skip colums header (NodeID PValue Counts)
            lineStr = f.readline().strip()
            idx = 0
            while lineStr and not lineStr.startswith('END'):
                pvalue = lineStr.split(' ')
                features[0][idx] = float(pvalue[1])
                idx += 1
                lineStr = f.readline().strip()
    
        if lineStr.startswith('SECTION4'):
            lineStr = f.readline().strip() # skip colums header (EndPoint0 EndPoint1 Weight)
            lineStr = f.readline().strip()
            while lineStr and not lineStr.startswith('END'):
                idx = lineStr.split(' ')
                idx = int(idx[0])
                labels[0][idx] = 1
                lineStr = f.readline().strip()
    
    txt_labels = labels
#    print 'subgraph size:', np.shape(np.where(labels == 1))[1]  
#    print 'features size:', np.shape(features)[1]
#    print 'labels size:', np.shape(labels)[1]
    
    f.close
    
    return labels, features, txt_labels
    
#def generateGraph(split='train'): # incomplete
#
#    assert split in ['train','test']
#    N = 10    
#    k = 0    
#    Nodes = np.zeros((N,N),dtype=np.int)
#    for i in xrange(N):
#        for j in xrange(N):
#            Nodes[i][j] = k
#            k = k + 1
#
#    mu1, sigma1 = 5, 1
#    mu2, sigma2 = 1, 1
#    
#    features = []
#    labels = []
#    if split == 'train':
#        sample_size = 5000
#    else:
#        
#        sample_size = 2000    
#    graph = np.array([[-1 for i in xrange(N*N)] for i in xrange(N*N)], dtype=np.float)
#    
#    count = 0
#    i = 1
#    for k in xrange(1, (N*N)+1):
#        count = count + 1
#        if (count > N):
#            count = 1
#            i = i + 1
#        j = count
#        if not ((i - 1) < 1):
#            graph[k - 1][Nodes[i - 2][j - 1]] = weight
#        if not ((j - 1) < 1):
#            graph[k - 1][Nodes[i - 1][j - 2]] = weight
#        if not ((j + 1) > N):
#            graph[k - 1][Nodes[i - 1][j]] = weight
#
#    print graph.flatten()
    
def get_data(split='train'):
    
    assert split in ['train','test']    
    gsize = 100
    sgsize = 20
    
    features = []
    labels = []
    if split == 'train':
        sample_size = 5000
    else:
        sample_size = 2000

    mu1, sigma1 = 1, 2.0/gsize
    mu2, sigma2 = 5, 2.0/sgsize

    for i in range(sample_size):
        s1 = np.random.normal(mu1,sigma1,gsize)
        idxs = random.sample(range(gsize),np.random.randint(sgsize - 1) + 1)
        s2 = np.zeros(gsize)
        for j in idxs:
            s1[j] = np.random.normal(mu2,sigma2)
            s2[j] = 1
        features.append(s1)
        labels.append(s2)
    
    features = np.array(features,dtype=np.float)
    labels = np.array(labels,dtype=np.int)
    txt_labels = labels
    txt_inputs = features
    
    if split == 'train':
           return labels, features, txt_labels # labels: n x 159, features: n x 1836, txt_labels: 159 x 1
    else:
        return labels, features, txt_labels, txt_inputs

def gcn_adv(split='train'):

    assert split in ['train','test']
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')

    # Some preprocessing
    features = preprocess_features(features)

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

def get_bibtex(split='train'):
    """Load the bibtex dataset."""
    assert split in ['train', 'test']
    feature_idx = 1836
    if split == 'test':
        dataset = arff.load(open('%s/bibtex/bibtex-test.arff' % dir_path, 'rb'))
    else:
        dataset = arff.load(open('%s/bibtex/bibtex-train.arff' % dir_path, 'rb'))

    data = np.array(dataset['data'], np.int)

    labels = data[:, feature_idx:] # all rows, 159 labels (columns: 1836-1995)
    features = data[:, 0:feature_idx] # all rows, 1836 features (columns: 0-1835)
    
#    print 'labels[0,:]:', labels[0,:]
#    print 'features[0,0] & [0,5]', features[0,0],features[0,5]
    
    txt_labels = [t[0] for t in dataset['attributes'][1836:]]
    txt_inputs = [t[0] for t in dataset['attributes'][:1836]]

    if split == 'train':
        return labels, features, txt_labels # labels: n x 159, features: n x 1836, txt_labels: 159 x 1
    else:
        return labels, features, txt_labels, txt_inputs


def get_bookmarks(split='train'):
    """Load the bookmarks dataset"""
    assert split in ['train', 'test']
    feature_dim = 2150
    label_dim = 208

    features = np.zeros((0, feature_dim))
    labels = np.zeros((0, label_dim))

    if split == "train":
        # Load train data
        for nr in range(1, 6):
            data = torchfile.load("%s/icml_mlc_data/data/bookmarks/bookmarks-train-%d.torch" % (dir_path,nr))
            labels = np.concatenate((labels, data['labels']), axis=0)
            features = np.concatenate((features, data['data'][:, 0:feature_dim]), axis=0)

        # Load dev data
        data = torchfile.load("%s/icml_mlc_data/data/bookmarks/bookmarks-dev.torch" % dir_path)
        labels = np.concatenate((labels, data['labels']), axis=0)
        features = np.concatenate((features, data['data'][:, 0:feature_dim]), axis=0)
    else:
        # Load train data
        for nr in range(1, 4):
            data = torchfile.load("%s/icml_mlc_data/data/bookmarks/bookmarks-test-%d.torch" % (dir_path, nr))
            labels = np.concatenate((labels, data['labels']), axis=0)
            features = np.concatenate((features, data['data'][:, 0:feature_dim]), axis=0)

    return labels, features, None
