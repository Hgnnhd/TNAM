"""
Expected input format
---------------------
The training script expects two pickle files under:
    data/train{dataset_name}/data.pickle
    data/train{dataset_name}/label.pickle
where:
- data.pickle: numpy array with shape (N, T, F)
- label.pickle: numpy array with shape (N,)
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from evaluate_score import evaluate_sepsis_score
from MODEL import TNAM
from utils import set_seed


def build_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(model, X_train, y_train, device, epochs, batch_size, learning_rate):
    loader = build_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


def evaluate_model(model, X_val, y_val, device, batch_size):
    loader = build_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    criterion = nn.BCELoss()
    total_loss = 0.0
    total_size = 0
    scores = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * len(y_batch)
            total_size += len(y_batch)
            scores.extend(outputs.cpu().numpy())

    scores = np.asarray(scores)
    predictions = scores > 0.5
    metrics = evaluate_sepsis_score(y_val, predictions, scores)
    return total_loss / total_size, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/A.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_config = config["train"]
    model_config = config["models"]["TNAM"]

    set_seed(int(train_config.get("seed", 42)))

    device = torch.device(f"cuda:{train_config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")
    dataset_name = str(train_config["dataset"])
    batch_size = int(train_config.get("batch_size", 256))
    learning_rate = float(train_config.get("learning_rate", 1e-3))
    epochs = int(train_config.get("epochs", 30))
    nfold = int(train_config.get("nfold", 5))
    hours = train_config.get("hours")

    data_dir = root / "data" / f"train{dataset_name}"
    with open(data_dir / "data.pickle", "rb") as f:
        features = np.asarray(pickle.load(f), dtype=np.float32)
    with open(data_dir / "label.pickle", "rb") as f:
        labels = np.asarray(pickle.load(f), dtype=np.float32).reshape(-1)

    if hours not in (None, "None"):
        features = features[:, : int(hours), :]

    kf = KFold(n_splits=nfold, shuffle=True, random_state=42)
    fold_losses = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_val = features[val_idx]
        y_val = labels[val_idx]

        model = TNAM(
            X_train.shape[-1],
            hidden_size=model_config.get("hidden_size", 64),
            num_layers=model_config.get("num_layers", 1),
            dropout_rate=model_config.get("dropout_rate", 0.0),
        ).to(device)

        train_one_fold(model, X_train, y_train, device, epochs, batch_size, learning_rate)
        val_loss, metrics = evaluate_model(model, X_val, y_val, device, batch_size)
        fold_losses.append(val_loss)
        fold_metrics.append(metrics)

        auroc, auprc, accuracy, sensitivity, specificity, f_measure = metrics
        print(
            f"Fold {fold}: val_loss={val_loss:.4f}, "
            f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}, ACC={accuracy:.4f}, "
            f"SEN={sensitivity:.4f}, SPE={specificity:.4f}, F1={f_measure:.4f}"
        )

    fold_losses = np.asarray(fold_losses)
    fold_metrics = np.asarray(fold_metrics)
    means = fold_metrics.mean(axis=0)
    stds = fold_metrics.std(axis=0)

    print(f"Average val_loss: {fold_losses.mean():.4f} ± {fold_losses.std():.4f}")
    print(
        "Average metrics: "
        f"AUROC={means[0]:.4f}±{stds[0]:.4f}, "
        f"AUPRC={means[1]:.4f}±{stds[1]:.4f}, "
        f"ACC={means[2]:.4f}±{stds[2]:.4f}, "
        f"SEN={means[3]:.4f}±{stds[3]:.4f}, "
        f"SPE={means[4]:.4f}±{stds[4]:.4f}, "
        f"F1={means[5]:.4f}±{stds[5]:.4f}"
    )


if __name__ == "__main__":
    main()
