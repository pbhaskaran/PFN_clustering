import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons, make_circles
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import torch
import random
device = torch.device("cuda")


def sample_clusters(batch_size=100, num_features=2, noise=False, num_classes=3,kmeans=False,
                    std_variation=False):
    batch_classes = []
    # generate sequences from 100 to 200 data points
    seq_len = random.randint(100, 200)
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_y_noisy = []
    for i in range(batch_size):
        centers = random.randint(2, num_classes)
        if std_variation:
            std = [random.random() for _ in range(centers)]
        else:
            std = random.random()
        batch_classes.append(centers)
        x, y = make_blobs(n_samples=seq_len, n_features=num_features, centers=centers, cluster_std=std,
                          shuffle=True)
        x = preprocessing.MinMaxScaler().fit_transform(x)
        x, y = sort(x, y, centers)
        clusters_x[i] = x
        clusters_y[i] = y

        if noise:
            # todo add noise to the target
            pass
        else:
            clusters_y_noisy.append(y)

    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_x = clusters_x.permute(1, 0, 2)
    clusters_y = clusters_y.permute(1, 0)
    if kmeans:
        return clusters_x, clusters_y, clusters_y, batch_classes

    return clusters_x.to(device), clusters_y.to(device), clusters_y.to(device)

def sort(x, y, centers):
    distances = np.linalg.norm(x, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_x = x[sorted_indices]
    sorted_y = y[sorted_indices]

    mapping = {}
    storage = set()
    curr = 0
    for i in range(len(sorted_y)):
        if len(mapping) == centers:
            break
        if sorted_y[i] not in storage:
            mapping[sorted_y[i]] = curr
            storage.add(sorted_y[i])
            curr += 1

    y_mapped = np.array([mapping[number] for number in sorted_y])
    indices = np.random.permutation(len(sorted_x))
    shuffled_x = sorted_x[indices]
    shuffled_y = y_mapped[indices]
    return shuffled_x, shuffled_y



def sample_clusters2(random_state=10, batch_size=100, num_features=2, noise=False, num_classes=3,kmeans=False,
                    std_variation=False):
    batch_classes = []
    # generate sequences from 100 to 200 data points
    rng = np.random.default_rng(random_state)
    seq_len = rng.integers(1, 5)
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_y_noisy = []
    for i in range(batch_size):
        random_state += 1
        rng = np.random.default_rng(random_state)
        centers = rng.integers(2, num_classes + 1)
        if std_variation:
            std = rng.random(centers).tolist()
        else:
            std = rng.integers(1, 2)
        batch_classes.append(centers)
        x, y = make_blobs(n_samples=seq_len, n_features=num_features, centers=centers, cluster_std=std,
                          shuffle=True, random_state=random_state)
        x = preprocessing.MinMaxScaler().fit_transform(x)
        x, y = sort(x, y, centers)
        clusters_x[i] = x
        clusters_y[i] = y
        if noise:
            # todo add noise to the target
            pass
        else:
            clusters_y_noisy.append(y)

    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_x = clusters_x.permute(1, 0, 2)
    clusters_y = clusters_y.permute(1, 0)
    if kmeans:
        return clusters_x.to(device), clusters_y.to(device), clusters_y.to(device), batch_classes

    return clusters_x.to(device), clusters_y.to(device), clusters_y.to(device)