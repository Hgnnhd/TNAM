import random

import numpy as np
import torch
from sklearn.utils import check_random_state


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CustomRandomUnderSampler:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        target_count = np.min(class_counts)

        resampled_data = []
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            sampled_indices = random_state.choice(class_indices, target_count, replace=False)
            resampled_data.append(X[sampled_indices])

        X_resampled = np.concatenate(resampled_data, axis=0)
        y_resampled = np.repeat(unique_classes, target_count)
        return X_resampled, y_resampled
