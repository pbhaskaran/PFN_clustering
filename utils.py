import random
import math
from torch.optim.lr_scheduler import LambdaLR
import itertools
import numpy as np
import torch
device = torch.device("cuda")
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import prior
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


def compute_split(X, y, y_noisy):
    S, B, _ = X.shape
    single_eval_pos = random.randint(int(1 * S / 4), int(3 * S / 4))
    return X, y_noisy[single_eval_pos:], single_eval_pos


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def permute(num_classes):
    numbers = np.arange(num_classes)
    permutation = list(itertools.permutations(numbers, num_classes))
    return permutation


def map_labels(permutation, y): # y is of shape S, B and needs to be in shape S*B
    mapping = {key: value for key, value in enumerate(permutation)}
    mapping_tensor = torch.empty(max(mapping.keys()) + 1, dtype=torch.long, device=device)
    for key, value in mapping.items():
        mapping_tensor[key] = value
    y= y.long()
    targets = mapping_tensor[y]
    targets = targets.reshape(-1).type(torch.LongTensor).to(device)
    return targets


def k_means(X_train, num_classes):
    X_train = X_train.cpu().numpy()
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(X_train)
    return kmeans.labels_




def compute_accuracy_metric(model,batch_size,seq_len, num_features, num_classes,std_variation=True, random_state=42):
    X,y, _, num_classes_array = prior.sample_dirichlet_clusters(batch_size=batch_size, num_features=num_features,
                                        num_classes=num_classes)


    accuracy_buckets = np.zeros(11)
    permutations = permute(num_classes=num_classes)

    # logits of shape (S,B,num_classes)
    logits = model(X.to(device))
    logits = logits.cpu()
    for batch_index, num_class in enumerate(num_classes_array):
        prediction = logits[:,batch_index, :].argmax(dim=-1)  # logits of shape S,1
        targets = y[:, batch_index]
        accuracy = accuracy_score(prediction.cpu().numpy(), targets.cpu().numpy()) * 100
        # if accuracy < 90:
        #     for permutation in permutations:
        #         target = map_labels(permutation, targets)
        #         curr_accuracy = accuracy_score(prediction.cpu().numpy(), target.cpu().numpy()) * 100
        #         if curr_accuracy > accuracy:
        #             accuracy = curr_accuracy

        acc_bucket = int(accuracy / 10)
        accuracy_buckets[acc_bucket] += 1

    return accuracy_buckets


def plot_accuracy_metric(accuracy_buckets):
    buckets = [f"{i * 10}-{(i + 1) * 10}%" for i in range(10)]
    plt.figure(figsize=(8, 5))
    plt.bar(buckets, accuracy_buckets, color='skyblue', edgecolor='black')
    plt.xlabel("Accuracy Ranges (%)")
    plt.ylabel("Frequency")
    plt.title("Accuracy Distribution Across Buckets")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




def compute_accuracy_metric2(model,batch_size, num_features, num_classes,std_variation=True, random_state=42):
    X,y, _, num_classes_array = prior.sample_clusters(batch_size=batch_size, num_features=num_features,
                                        num_classes=num_classes,kmeans=True,std_variation=std_variation)


    accuracy_buckets = np.zeros(11)
    accuracy_buckets_gmm = np.zeros(11)
    accuracy_buckets_kmeans = np.zeros(11)

    # logits of shape (S,B,num_classes)
    logits = model(X.to(device))
    logits = logits.cpu()
    for batch_index, num_class in enumerate(num_classes_array):
        if num_class < 5:
            prediction = logits[:,batch_index, :].argmax(dim=-1)  # logits of shape S,1
            gmm = GaussianMixture(n_components=num_class, random_state=42)
            gmm.fit(X[: , batch_index, :])
            gmm_labels = gmm.predict(X[: , batch_index, :])
            k_labels = k_means(X[: , batch_index, :], num_class)
            targets = y[:, batch_index]
            accuracy = accuracy_score(prediction.cpu().numpy(), targets.cpu().numpy()) * 100

            if num_class < 5 and accuracy < 80:
                permutations = permute(num_classes=num_class)
                for permutation in permutations:
                    target = map_labels(permutation, targets)
                    curr_accuracy = accuracy_score(target.cpu().numpy(), prediction.cpu().numpy()) * 100
                    if curr_accuracy > accuracy:
                        accuracy = curr_accuracy
            acc_bucket = int(accuracy / 10)
            accuracy_buckets[acc_bucket] += 1


            accuracy_gmm = 0
            permutations = permute(num_classes=num_class)
            for permutation in permutations:
                target = map_labels(permutation, targets)
                curr_accuracy = accuracy_score(target.cpu().numpy(), gmm_labels) * 100
                if curr_accuracy > accuracy_gmm:
                    accuracy_gmm = curr_accuracy
            acc_bucket_gmm = int(accuracy_gmm / 10)
            accuracy_buckets_gmm[acc_bucket_gmm] += 1

            accuracy_k_labels = 0
            permutations = permute(num_classes=num_class)
            for permutation in permutations:
                target = map_labels(permutation, targets)
                curr_accuracy = accuracy_score(target.cpu().numpy(), k_labels) * 100
                if curr_accuracy > accuracy_k_labels:
                    accuracy_k_labels = curr_accuracy
            acc_bucket_kmeans = int(accuracy_k_labels / 10)
            accuracy_buckets_kmeans[acc_bucket_kmeans] += 1

    return accuracy_buckets, accuracy_buckets_gmm, accuracy_buckets_kmeans


def predict_num_classes(model,batch_size, num_features, num_classes,std_variation=True,exact=False, random_state=42):
    X,y, _, num_classes_array = prior.sample_clusters(batch_size=batch_size, num_features=num_features,
                                        num_classes=num_classes,kmeans=True,std_variation=std_variation)

    logits = model(X.to(device))
    logits = logits.cpu()
    counts_transformer = 0
    counts_gmm_bayes = 0

    for batch_index, num_class in enumerate(num_classes_array):
        if exact:
            n_components = num_class
        else:
            n_components = num_classes
        prediction = logits[:, batch_index, :].argmax(dim=-1)
        gmm_bayes = BayesianGaussianMixture(n_components=n_components, random_state=42)
        gmm_bayes.fit(X[:, batch_index, :])
        gmm_bayes_labels = gmm_bayes.predict(X[:, batch_index, :])
        transformer_labels = len(np.unique(prediction))
        gmm_bayes_labels = len(np.unique(gmm_bayes_labels))
        if transformer_labels == num_class:
            counts_transformer += 1
        if gmm_bayes_labels == num_class:
            counts_gmm_bayes += 1

    return f"Proportion of dataset classes predicted correctly: Transformer: {counts_transformer / batch_size:.2f}, BGM: {counts_gmm_bayes / batch_size:.2f}"


    def num_classes_correctly_predicted():
        pass


    def num_classe():
        pass
