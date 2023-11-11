
import sys
sys.path.insert(0, '/home/zhuoyan/vision/pmf_cvpr22')

import os
import time
import json
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn.functional as F
from collections import defaultdict

def load_test_features(irun = 0):
    data_root = "/datadrive2/datasets/meta_dataset_taskEmb/testSet/clip"
    dataset_name = f"subset{irun}_8domains_features.json"
    cached_file_test = os.path.join(
        data_root,
        dataset_name,
    )

    if os.path.exists(cached_file_test):
        start_load = time.time()
        with open(cached_file_test) as f:
            features = json.load(f)
        
        print(
            "Loading image features from cached file {} [took {:.3f} s]".format(cached_file_test, time.time() - start_load)
        )
    else:
        print("didn't load image feature | test")
    
    for key, val in features.items():
        # print(key)
        fea = torch.tensor(val)
        # print(fea.shape)
        features[key] = fea
    return features

def load_train_task_features(task_path = "/datadrive2/datasets/meta_dataset_taskEmb/100_per_domain/clip", task_id = 0):
    with open(f'{task_path}/task_{task_id}.json', 'r') as json_file:
        task = json.load(json_file)
    # source = task['source']
    fea = torch.tensor(task['features'])
    return fea

def MeanEmbeddingSimilarity(embeddings1, embeddings2):
    mean_embedding1 = embeddings1.mean(dim=0)
    mean_embedding2 = embeddings2.mean(dim=0)
    similarity = torch.nn.functional.cosine_similarity(mean_embedding1, mean_embedding2, dim=0)
    return similarity.item()

def PairwiseSimilarity(embeddings1, embeddings2):
    similarities = torch.mm(embeddings1, embeddings2.type(torch.float32).t())
    # print(similarities)
    mean_similarity = similarities.mean()
    return mean_similarity.item()

import numpy as np
from scipy.stats import chisquare

def histogram_distance(embeddings1, embeddings2, num_bins=30):
    # Compute pairwise cosine similarities
    similarities = np.matmul(embeddings1, embeddings2.T)
    hist1, bin_edges = np.histogram(similarities.flatten(), bins=num_bins, density=True)  # density=True for normalized histogram

    # Compute pairwise cosine similarities (in the opposite order)
    similarities = np.matmul(embeddings2, embeddings1.T)
    hist2, _ = np.histogram(similarities.flatten(), bins=bin_edges, density=True)  # density=True for normalized histogram

    # Chi-squared distance
    chi2_dist, _ = chisquare(hist1, hist2)

    return -chi2_dist

import ot

def wasserstein_distance(embeddings1, embeddings2):
    # Convert to float32 if they're not
    if embeddings1.dtype == torch.float16:
        embeddings1 = embeddings1.to(torch.float32)
    if embeddings2.dtype == torch.float16:
        embeddings2 = embeddings2.to(torch.float32)
    
    # Convert to numpy for computation with the POT library
    embeddings1 = embeddings1.cpu().numpy()
    embeddings2 = embeddings2.cpu().numpy()

    # Normalize embeddings to make sure they're in same scale
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # Pairwise Euclidean distance
    distance_matrix = ot.dist(embeddings1, embeddings2)

    # Uniform distribution over embeddings
    p = np.ones(embeddings1.shape[0]) / embeddings1.shape[0]
    q = np.ones(embeddings2.shape[0]) / embeddings2.shape[0]

    # Compute Wasserstein distance
    w_dist = ot.emd2(p, q, distance_matrix)

    return -w_dist

simfunc = {"MeanEmbeddingSimilarity": MeanEmbeddingSimilarity,
           "PairwiseSimilarity": PairwiseSimilarity,
           "histogram_distance": histogram_distance,
           "wasserstein_distance": wasserstein_distance
           }




def compute_mahalanobis_distance(target_emb, task_emb):
    """
    Compute the Mahalanobis distance between target embeddings and task embeddings.
    
    Parameters:
    - target_emb: Tensor of shape (N, D), where N is the number of samples and D is the embedding size.
    - task_emb: Tensor of shape (M, D), where M is the number of samples and D is the embedding size.
    
    Returns:
    - Mahalanobis distance scalar.
    """
    target_mean = torch.mean(target_emb, dim=0)
    task_mean = torch.mean(task_emb, dim=0)

    # Compute the covariance matrix and its inverse
    task_covariance = torch_cov(task_emb)
    task_covariance_inv = torch.inverse(task_covariance)

    # Compute the Mahalanobis distance
    diff = target_mean - task_mean
    mahalanobis_distance = torch.matmul(diff, torch.matmul(task_covariance_inv, diff))

    return mahalanobis_distance

def torch_cov(m, rowvar=False):
    """
    Estimate a covariance matrix given data.
    
    Parameters:
    - m: A 2D Tensor of shape (N, D), where N is the number of samples and D is the embedding size.
    - rowvar: If True, treat rows as variables (default is to treat columns as variables).
    
    Returns:
    - Covariance matrix of shape (D, D).
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m = m - torch.mean(m, dim=1, keepdim=True)
    return (m @ m.t()) / (m.size(1) - 1)