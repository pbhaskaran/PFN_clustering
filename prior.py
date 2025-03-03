import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons, make_circles
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import torch
import random
from scipy.stats import dirichlet, multivariate_normal
import random
device = torch.device("cuda")
random_state = 0

def sample_clusters(batch_size=100, num_features=2, seq_len=200, num_classes=10, cluster_type='make_blobs', **kwargs):
    if cluster_type == 'make_blobs':
        return sample_make_blob_clusters(batch_size=batch_size,seq_len=seq_len,
                                         num_features=num_features, num_classes=num_classes, **kwargs)
    elif cluster_type == 'dirichlet':
        return sample_dirichlet_clusters(batch_size=batch_size,seq_len=seq_len,
                                         num_features=num_features, num_classes=num_classes, **kwargs)
    elif cluster_type == 'make_circles':
        return sample_circles(batch_size=batch_size, seq_len=seq_len)
    else:
        print("cluster type not found!")

def sample_make_blob_clusters(batch_size=100, seq_len=300, num_features=2,min_classes=1, num_classes=10, **kwargs):
    batch_classes = []
    # generate sequences from 100 to 200 data points
    seq_len = random.randint(100, seq_len)
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_x_true = np.zeros((batch_size, seq_len, num_features))
    print(clusters_x.shape)
    for i in range(batch_size):
        centers = random.randint(min_classes, num_classes)
        n_samples = fill_buckets_dirichlet(seq_len, centers, None)
        std = [random.uniform(0.5, 2.0) for _ in range(centers)]
        centers_arr = np.random.uniform(-10, 10, size=(centers, 2))
        batch_classes.append(centers)
        print(num_features)
        X_true, y = make_blobs(n_samples=n_samples, n_features=5, centers=centers_arr, cluster_std=std,
                          shuffle=True)
        print(y.shape, X_true.shape)
        x = preprocessing.MinMaxScaler().fit_transform(X_true)
        x, y, X_true = sort(x, y,X_true, centers)
        print(x.shape, y.shape, X_true.shape)
        clusters_x_true[i] = X_true
        clusters_x[i] = x
        clusters_y[i] = y

    clusters_x_true = torch.tensor(clusters_x_true, dtype=torch.float32)
    clusters_x_true = clusters_x_true.permute(1, 0, 2)
    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_x = clusters_x.permute(1, 0, 2)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_y = clusters_y.permute(1, 0)
    batch_classes = torch.tensor(batch_classes).unsqueeze(0)
    return clusters_x.to(device), clusters_y.to(device), clusters_x_true.to(device), batch_classes.to(device)


def sort( x, y,X_true, centers):
    distances = np.linalg.norm(x, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_x = x[sorted_indices]
    sorted_y = y[sorted_indices]
    sorted_X_true = X_true[sorted_indices]
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
    shuffled_X_true = sorted_X_true[indices]
    return shuffled_x, shuffled_y, shuffled_X_true

def sample_dirichlet_clusters(batch_size=100,seq_len=200, num_features=2, num_classes=10, **kwargs):
    clusters_x = np.zeros((batch_size, seq_len, num_features))
    clusters_x_true = np.zeros((batch_size, seq_len, num_features))
    clusters_y = np.zeros((batch_size, seq_len))
    batch_classes = np.zeros(batch_size)
    for i in range(batch_size):
        X_true, y = sample_dirichlet_process_gaussians(n_samples=seq_len,num_features=num_features,num_classes=num_classes, alpha=1, base_mean=0, base_cov=10)
        num_clusters = len(np.unique(y))
        X = preprocessing.MinMaxScaler().fit_transform(X_true)
        X, y, X_true = sort( X, y,X_true, num_clusters)
        clusters_x_true[i] = X_true
        clusters_x[i] = X
        clusters_y[i] = y
        batch_classes[i] = num_clusters

    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_x_true = torch.tensor(clusters_x_true, dtype=torch.float32)
    clusters_x =  clusters_x.permute(1, 0, 2)
    clusters_x_true =  clusters_x_true.permute(1, 0, 2)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_y = clusters_y.permute(1, 0)
    batch_classes = torch.tensor(batch_classes, dtype=torch.long).unsqueeze(0)
    return clusters_x.to(device), clusters_y.to(device), clusters_x_true.to(device), batch_classes.to(device)


def sample_dirichlet_process_gaussians(n_samples=500,num_features=2,num_classes=10, alpha=1.0, base_mean=0, base_cov=10):
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


def fill_buckets_dirichlet(total=150, n_buckets=5, alpha=None):
    if alpha is None:
        alpha = np.ones(n_buckets)

    min_count = 10
    remaining_total = total - (n_buckets * min_count)
    proportions = np.random.dirichlet(alpha)
    additional_counts = np.round(proportions * remaining_total).astype(int)
    diff = remaining_total - additional_counts.sum()
    additional_counts[:abs(diff)] += np.sign(diff)

    bucket_counts = additional_counts + min_count
    return bucket_counts


def sample_circles(batch_size= 10, seq_len = 200):
    batch_classes = []
    # generate sequences from 100 to 200 data points
    seq_len = random.randint(100, seq_len)
    clusters_x = np.zeros((batch_size, seq_len, 2))
    clusters_y = np.zeros((batch_size, seq_len))
    clusters_x_true = np.zeros((batch_size, seq_len, 2))

    for i in range(batch_size):
        X_true, y = make_circles(n_samples=seq_len, factor=random.uniform(0, 1))
        x = preprocessing.MinMaxScaler().fit_transform(X_true)
        x, y, X_true = sort(x, y,X_true, 2)
        clusters_x_true[i] = X_true
        clusters_x[i] = x
        clusters_y[i] = y
        batch_classes.append(2)

    clusters_x_true = torch.tensor(clusters_x_true, dtype=torch.float32)
    clusters_x_true = clusters_x_true.permute(1, 0, 2)
    clusters_x = torch.tensor(clusters_x, dtype=torch.float32)
    clusters_x = clusters_x.permute(1, 0, 2)
    clusters_y = torch.tensor(clusters_y, dtype=torch.float32)
    clusters_y = clusters_y.permute(1, 0)
    batch_classes = torch.tensor(batch_classes).unsqueeze(0)
    return clusters_x.to(device), clusters_y.to(device), clusters_x_true.to(device), batch_classes.to(device)