from sklearn import neighbors
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

def radius_adj(X, radius, mode='distance'):
    A = neighbors.radius_neighbors_graph(X, radius, mode=mode)
    return A

def kneighbors_adj(X, n_neighbors, mode='distance'):
    A = neighbors.kneighbors_graph(X, n_neighbors, mode=mode)
    return A

def heat_kernel_weights(dists, param):
    W = -dists**2/param
    np.exp(W.data, W.data)
    return W

def compute_mapping(X, W, l):
    D = np.diagflat(W.sum(axis=0))
    L = D - W
    eigvals, eigvecs = eigh(X.T.dot(L).dot(X), X.T.dot(D).dot(X), eigvals=(0,l-1))
    return eigvecs

class LocalityPreservingProjection(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, adjacency='kneighbors',
         adjacency_param=3, weights='heatkernel',
         kernel_param=0.1):
        self.n_components = n_components
        self.adjacency = adjacency
        self.adjacency_param = adjacency_param
        self.weights = weights
        self.kernel_param = kernel_param

        
    def fit(self, X, y=None):
        if self.adjacency == 'kneighbors':
            adj_func = kneighbors_adj
        else:
            adj_func = radius_adj
            
        if self.weights == 'heatkernel':
            mode = 'distance'
        else:
            mode = 'connectivity'
            
        W = adj_func(X, self.adjacency_param, mode=mode)
        
        if self.weights == 'heatkernel':
            W = heat_kernel_weights(W, self.kernel_param)
        
        self.components_ = compute_mapping(X, W, self.n_components)
        
        return self
            
    def transform(self, X):
        return X.dot(self.components_)






