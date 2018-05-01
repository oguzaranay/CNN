# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:22:27 2018

@author: oguz
"""
import numpy as np

def generateGraph(N, weight):
#    graph = np.ones((N * N, N * N))
    graph = np.array([[np.random.normal(5,1,1) for i in range(N*N)] for i in range(N*N)], dtype=np.float)
    k = 0    
    Nodes = np.zeros((N,N),dtype=np.int)
    for i in range(N):
        for j in range(N):
            Nodes[i][j] = k
            k = k + 1
#    print Nodes
    
    count = 0
    i = 1
    for k in range(1, (N*N)+1):
        count = count + 1
        if (count > N):
            count = 1
            i = i + 1
        j = count
        if not ((i - 1) < 1):
#            graph[k - 1][Nodes[N - 1][j - 1]] = weight # [0][6], [1][7], [2][8]
#        else:
            graph[k - 1][Nodes[i - 2][j - 1]] = weight
#        if ((i + 1) > N):
#            graph[k - 1][Nodes[0][j - 1]] = weight
#        else:
#            graph[k - 1][Nodes[i][j - 1]] = weight 
        if not ((j - 1) < 1):
#            graph[k - 1][Nodes[i - 1][N - 1]] = weight # [0][2], [3][5]
#        else:
            graph[k - 1][Nodes[i - 1][j - 2]] = weight
        if not ((j + 1) > N):
#            graph[k - 1][Nodes[i - 1][0]] = weight # [2][0]
#        else:
            graph[k - 1][Nodes[i - 1][j]] = weight
#    print k 
#    print graph
#    graph = np.array(graph,dtype=np.float)
    graph = graph.flatten()
    print(graph)
    
     
if __name__=='__main__':
    generateGraph(2,4)
    
#    count = 0
#    i = 1
#    for k in xrange(1, N * N):
#        count = count + 1
#        if (count > N):
#            count = 1
#            i = i + 1
#        j = count
#        if ((i - 1) < 1):
#            graph[k - 1][Nodes[N - 1][j - 1]] = weight # [0][6], [1][7], [2][8]
#        else:
#            graph[k - 1][Nodes[i - 2][j - 1]] = weight
#        if ((i + 1) > N):
#            graph[k - 1][Nodes[0][j - 1]] = weight
#        else:
#            graph[k - 1][Nodes[i][j - 1]] = weight 
#        if ((j - 1) < 1):
#            graph[k - 1][Nodes[i - 1][N - 1]] = weight # [0][2], [3][5]
#        else:
#            graph[k - 1][Nodes[i - 1][j - 2]] = weight
#        if ((j + 1) > N):
#            graph[k - 1][Nodes[i - 1][0]] = weight # [2][0]
#        else:
#            graph[k - 1][Nodes[i - 1][j]] = weight
