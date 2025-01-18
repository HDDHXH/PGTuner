import sys
sys.path.append('./utils')
import os
# from sklearn.cluster import Birch, DBSCAN
# from hdbscan import HDBSCAN
import numpy as np
import cupy as cp
from cuml import UMAP, PCA
from cuml.decomposition import IncrementalPCA
from cuml.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA as PCA_CPU
import random
import time
from tqdm import tqdm


def umap_dr(data, num_components):
    umap = UMAP(n_components=num_components, n_neighbors=15, random_state=42)
    reduced_data = umap.fit_transform(data)

    return reduced_data
 
def randomprojection_dr(data, num_components):
    rp = GaussianRandomProjection(n_components=num_components, random_state=42)
    reduced_data = rp.fit_transform(data)

    return reduced_data

def pca_dr(data, num_components):
    rp = PCA(n_components=num_components, random_state=42)
    reduced_data = rp.fit_transform(data)

    del rp

    return reduced_data

def incre_pca_dr(data, num_components, batchsize):
    rp = IncrementalPCA(n_components=num_components, batch_size =batchsize)
    reduced_data = rp.fit_transform(data)

    del rp

    return reduced_data

def pca_dr_cpu(data, num_components):
    rp = PCA_CPU(n_components=num_components, random_state=42)
    reduced_data = rp.fit_transform(data)

    del rp

    return reduced_data