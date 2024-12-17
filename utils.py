import random
import math
from torch.optim.lr_scheduler import LambdaLR
import itertools
import numpy as np
import torch
device = torch.device("cuda")

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

