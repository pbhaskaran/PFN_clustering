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



def compute_accuracy_distribution(X, y, batch_classes,model=None,model_type=None, **kwargs):
    if model_type == 'transformer':
        return compute_accuracy_distribution_transformer(X, y,batch_classes,model, ** kwargs)
    elif model_type == 'gmm' or 'kmeans':
        return compute_accuracy_distribution_sklearn(X, y,batch_classes, model_type, **kwargs)
    else:
        print(f"Model type {model_type} does not exist")


def compute_accuracy_distribution_transformer(X, y, batch_classes, model, **kwargs):
    accuracy_buckets = np.zeros(11)
    # logits of shape (S,B,num_classes)
    logits, cluster_count_output = model(X.to(device),batch_classes)
    logits = logits.cpu()
    batch_classes = batch_classes.permute(1, 0)
    for batch_index, num_class in enumerate(batch_classes):
        prediction = logits[:, batch_index, :].argmax(dim=-1)  # logits of shape S,1
        permutations = permute(num_classes=num_class.cpu().item()) # Need to change this
        targets = y[:, batch_index]
        accuracy = accuracy_score(prediction.cpu().numpy(), targets.cpu().numpy()) * 100

        for permutation in permutations:
            target = map_labels(permutation, targets)
            curr_accuracy = accuracy_score(target.cpu().numpy(), prediction.cpu().numpy()) * 100
            if curr_accuracy > accuracy:
                accuracy = curr_accuracy
        acc_bucket = int(accuracy / 10)
        accuracy_buckets[acc_bucket] += 1
    return accuracy_buckets

def compute_accuracy_distribution_sklearn(X, y, batch_classes, model_type, **kwargs):
    accuracy_buckets = np.zeros(11)
    batch_classes = batch_classes.permute(1, 0)

    for batch_index, num_class in enumerate(batch_classes):
        num_class = num_class.cpu().item()
        if model_type == 'gmm':
            model =  GaussianMixture(n_components=num_class, random_state=42)
        elif model_type == 'kmeans':
            model = KMeans(n_clusters=num_class, random_state=42)
        else:
            print("Model not found")
        input_fit = X[: , batch_index, :].cpu()
        model.fit(input_fit)
        targets = y[:, batch_index]
        labels = model.predict(input_fit)
        accuracy = 0
        permutations =permute(num_classes=num_class)
        for permutation in permutations:
            target = map_labels(permutation, targets)
            curr_accuracy = accuracy_score(target.cpu().numpy(), labels) * 100
            if curr_accuracy > accuracy:
                accuracy = curr_accuracy
        acc_bucket = int(accuracy / 10)
        accuracy_buckets[acc_bucket] += 1

    return accuracy_buckets



def plot_accuracy_metric(accuracy_buckets, model_name):
    buckets = [f"{i * 10}-{(i + 1) * 10 -1}%" for i in range(10)]
    buckets.append("100%")
    plt.figure(figsize=(8, 5))
    plt.bar(buckets, accuracy_buckets, color='skyblue', edgecolor='black')
    plt.xlabel("Accuracy Ranges (%)")
    plt.ylabel("Frequency")
    plt.title(f"Accuracy Distribution Across Buckets {model_name}")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# def predict_num_classes(model,batch_size, num_features, num_classes,std_variation=True,exact=False, random_state=42):
#     X,y, _, num_classes_array = prior.sample_clusters(batch_size=batch_size, num_features=num_features,
#                                         num_classes=num_classes,kmeans=True,std_variation=std_variation)
#
#     logits = model(X.to(device))
#     logits = logits.cpu()
#     counts_transformer = 0
#     counts_gmm_bayes = 0
#
#     for batch_index, num_class in enumerate(num_classes_array):
#         if exact:
#             n_components = num_class
#         else:
#             n_components = num_classes
#         prediction = logits[:, batch_index, :].argmax(dim=-1)
#         gmm_bayes = BayesianGaussianMixture(n_components=n_components, random_state=42)
#         gmm_bayes.fit(X[:, batch_index, :])
#         gmm_bayes_labels = gmm_bayes.predict(X[:, batch_index, :])
#         transformer_labels = len(np.unique(prediction))
#         gmm_bayes_labels = len(np.unique(gmm_bayes_labels))
#         if transformer_labels == num_class:
#             counts_transformer += 1
#         if gmm_bayes_labels == num_class:
#             counts_gmm_bayes += 1
#
#     return (f"Proportion of dataset classes predicted correctly: Transformer:"
#             f" {counts_transformer / batch_size:.2f}, BGM: {counts_gmm_bayes / batch_size:.2f}")