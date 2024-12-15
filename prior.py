import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons, make_circles
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import torch
import random
device = torch.device("cuda")


def sample_clusters(batch_size=100, seq_len=100, num_features=2, type_='make_blobs', noise=False):
    clusters_x = []
    clusters_y = []
    clusters_y_noisy = []
    for i in range(batch_size):
        std = random.random()
        centers = random.randint(2, 8)
        x, y = make_blobs(n_samples=seq_len, n_features=num_features, centers=3, cluster_std=std,
                          shuffle=True)
        x = preprocessing.MinMaxScaler().fit_transform(x)
        # y = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1,1))
        clusters_x.append(x)
        clusters_y.append(y)

        if noise:
            # todo add noise to the target
            pass
        else:
            clusters_y_noisy.append(y)

    clusters_x = torch.tensor(np.array(clusters_x), dtype=torch.float32)
    clusters_y = torch.tensor(np.array(clusters_y) , dtype=torch.float32)
    clusters_x = clusters_x.permute(1, 0, 2)
    clusters_y = clusters_y.permute(1, 0)

    return clusters_x.to(device), clusters_y.to(device), clusters_y.to(device)


