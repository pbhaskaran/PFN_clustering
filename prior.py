import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons, make_circles
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import torch
import random
from scipy.stats import dirichlet, multivariate_normal
device = torch.device("cuda")
random_state = 0
def sample_clusters(batch_size=100, num_features=2,seq_len=200, noise=False, num_classes=10,random_seed=0, kmeans=False,
                    std_variation=True):
    global random_state
    generator = np.random.default_rng(random_seed)
    batch_classes = []
    # generate sequences from 100 to 200 data points
    seq_len = generator.integers(low=100, high=seq_len, size=1)[0]
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_y_noisy = []
    for i in range(batch_size):
        centers = generator.integers(2, high=num_classes , size=1)[0]
        if std_variation:
            std = [generator.random() for _ in range(centers)]
        batch_classes.append(centers)
        x, y = make_blobs(n_samples=seq_len, n_features=num_features, centers=centers, cluster_std=std,
                          shuffle=True, random_state=random_state)
        random_state += 1
        x = preprocessing.MinMaxScaler().fit_transform(x)
        x, y = sort(x, y, centers)
        # y -= 1
        clusters_x[i] = x
        clusters_y[i] = y
        clusters_y_noisy.append(y)

    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_x = clusters_x.permute(1, 0, 2)
    clusters_y = clusters_y.permute(1, 0)
    batch_classes = torch.tensor(batch_classes, dtype=torch.long).unsqueeze(0)
    return clusters_x.to(device), clusters_y.to(device), None,  None, batch_classes.to(device)

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

def sample_dirichlet_clusters(batch_size=100,seq_len=200, num_features=2, num_classes=10,random_seed=0):
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_x_true = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_y_true = np.zeros((batch_size, seq_len))
    batch_classes = np.zeros(batch_size)
    for i in range(batch_size):
        X, y = sample_dirichlet_process_gaussians(n_samples=seq_len,num_features=num_features,num_classes=num_classes, alpha=1, base_mean=0, base_cov=10)
        num_clusters = len(np.unique(y))
        # y -= 1
        clusters_x_true[i] = X
        clusters_y_true[i] = y
        X = preprocessing.MinMaxScaler().fit_transform(X)
        X, y = sort(X, y, num_clusters)
        clusters_x[i] = X
        clusters_y[i] = y
        batch_classes[i] = num_clusters
    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_x_true = torch.tensor(clusters_x_true, dtype=torch.float32)

    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_y_true = torch.tensor(clusters_y_true, dtype=torch.float32)
    clusters_x =  clusters_x.permute(1, 0, 2)
    clusters_x_true =  clusters_x_true.permute(1, 0, 2)
    clusters_y = clusters_y.permute(1, 0)
    clusters_y_true = clusters_y_true.permute(1,0)
    batch_classes = torch.tensor(batch_classes, dtype=torch.long).unsqueeze(0)
    return clusters_x.to(device) , clusters_y.to(device), clusters_x_true.to(device),clusters_y_true.to(device), batch_classes.to(device)


def sample_dirichlet_process_gaussians(n_samples=500,num_features=2,num_classes=10, alpha=1.0, base_mean=0, base_cov=10000):
    max_clusters = num_classes
    # Base measure (Gaussian prior for means of clusters)
    base_mean = np.zeros(num_features) if isinstance(base_mean, (int, float)) else np.array(base_mean)
    base_cov = np.eye(num_features) * base_cov if isinstance(base_cov, (int, float)) else np.array(base_cov)
    correlation_strength= 0
    # Stick-breaking process to determine cluster weights
    betas = np.random.beta(1, alpha, size=n_samples)
    pis = betas * np.cumprod(np.hstack([[1], 1 - betas[:-1]]))  # Stick-breaking probabilities

    # Assign samples to clusters
    cluster_indices = np.random.choice(min(len(pis), max_clusters), size=n_samples, p=pis[:max_clusters]/np.sum(pis[:max_clusters]))
    unique_clusters = np.unique(cluster_indices)
    mapping = {unique_clusters[i] : i for i in range(len(unique_clusters))}
    unique_clusters_mapped = [mapping.get(x) for x in unique_clusters]
    cluster_indices_mapped =  np.array([mapping.get(x) for x in cluster_indices])
    # Sample cluster means and covariances
    cluster_means = {k: np.random.multivariate_normal(base_mean, base_cov) for k in unique_clusters_mapped}
    cluster_covs = {k: np.eye(num_features) * (0.5 + np.random.rand(num_features,num_features)) for k in unique_clusters_mapped}  # Random covariances
    #cluster_covs = {}
    # for k in unique_clusters_mapped:
    #     random_cov = np.random.rand(dim, dim)
    #     correlated_cov = (random_cov + random_cov.T) / 2  # Make symmetric
    #     np.fill_diagonal(correlated_cov, 1)
    #     correlated_cov *= correlation_strength # Scale correlation strength
    #     cluster_covs[k] = np.dot(correlated_cov, correlated_cov.T)  # Ensure positive semi-definiteness

    # Generate data points
    X = np.array([multivariate_normal.rvs(mean=cluster_means[k], cov=cluster_covs[k]) for k in cluster_indices_mapped])
    return X, cluster_indices_mapped

def sample_clusters2(batch_size=100, num_features=2, noise=False, num_classes=3,random_seed=0, kmeans=False,
                    std_variation=False):
    generator = np.random.default_rng(random_seed)
    batch_classes = []
    # generate sequences from 100 to 200 data points
    seq_len = generator.integers(low=100, high=201, size=1)[0]
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_y_noisy = []
    for i in range(batch_size):
        centers = generator.integers(2, high=num_classes , size=1)[0]
        if std_variation:
            std = [generator.random() for _ in range(centers)]
        batch_classes.append(centers)
        x, y = make_blobs(n_samples=seq_len, n_features=num_features, centers=centers, cluster_std=std,
                          shuffle=True, random_state=random_seed)
        random_seed += 1
        x = preprocessing.MinMaxScaler().fit_transform(x)
        x, y = sort(x, y, centers)
        clusters_x[i] = x
        clusters_y[i] = y
        clusters_y_noisy.append(y)

    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_x = clusters_x.permute(1, 0, 2)
    clusters_y = clusters_y.permute(1, 0)
    batch_classes = torch.tensor(batch_classes).unsqueeze(0)
    return clusters_x.to(device), clusters_y.to(device), clusters_y.to(device), batch_classes.to(device)