import argparse
import datetime
import logging
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from evaluate_sepsis_score import evaluate_sepsis_score
from MODEL import TNAM
from utils import CustomRandomUnderSampler, set_seed


def select_model(input_dim, input_length, model_config, dataset_name, device):
    hidden = model_config.get("hidden_size", 64)
    layers = model_config.get("num_layers", 1)
    dropout = model_config.get("dropout_rate", 0.0)
    return TNAM(
        input_dim,
        hidden_size=hidden,
        num_layers=layers,
        dropout_rate=dropout,
        time=input_length,
        dataset_name=dataset_name,
    ).to(device)


def build_loader(
    X,
    y,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
):
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def train_model(
    model,
    criterion,
    optimizer,
    Xtrain,
    ytrain,
    device,
    epochs,
    batch_size,
    amp_enabled,
    scaler,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
):
    train_loader = build_loader(
        Xtrain,
        ytrain,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext

    for _ in tqdm(range(epochs)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=pin_memory)
            y_batch = y_batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                outputs = model(X_batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                outputs = outputs.view(-1)

            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(outputs.float(), y_batch.float())

            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


def evaluate_model(
    model,
    Xtest,
    ytest,
    device,
    batch_size,
    amp_enabled,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
):
    model.eval()
    test_loader = build_loader(
        Xtest,
        ytest,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else nullcontext

    with torch.no_grad():
        predictions = []
        with autocast_ctx():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device, non_blocking=pin_memory)
                outputs = model(X_batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                predictions.extend(outputs.view(-1).float().cpu().numpy())

    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description="Train TNAM for sepsis prediction.")
    parser.add_argument("--config", type=str, default="config/B.yaml", help="Path to YAML config")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_config = config["train"]
    model_name = train_config.get("model", "TNAM")
    if model_name != "TNAM":
        raise ValueError(f"This minimal package only supports TNAM, got: {model_name}")

    model_config = config["models"]["TNAM"]

    device = torch.device(f"cuda:{train_config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = train_config.get("cudnn_benchmark", True)
        torch.backends.cuda.matmul.allow_tf32 = train_config.get("allow_tf32", True)

    matmul_precision = train_config.get("matmul_precision")
    if matmul_precision:
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except AttributeError:
            pass

    amp_enabled = bool(train_config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    compile_enabled = bool(train_config.get("compile", False))

    num_workers = int(train_config.get("num_workers", 0))
    pin_memory = bool(train_config.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(train_config.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(train_config.get("prefetch_factor", 2))

    set_seed(int(train_config.get("seed", 42)))

    dataset_name = str(train_config["dataset"])
    batch_size = int(train_config.get("batch_size", 1024))
    eval_batch_size = int(train_config.get("eval_batch_size", batch_size))
    learning_rate = float(train_config.get("learning_rate", 1e-3))
    epochs = int(train_config.get("epochs", 30))

    data_dir = root / "data" / f"train{dataset_name}"
    feature_path = data_dir / "data.pickle"
    label_path = data_dir / "label.pickle"
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing data files in {data_dir}. Need data.pickle and label.pickle."
        )

    with open(feature_path, "rb") as f:
        features = pickle.load(f)
    with open(label_path, "rb") as f:
        classes = pickle.load(f)

    log_dir = root / "log"
    save_dir = root / "save" / "model"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = log_dir / (
        f"training_{dataset_name}_{model_name}_bs{batch_size}_lr{learning_rate}_ep{epochs}_{now}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler()],
    )

    logging.info(
        "Training %s on %s with batch_size=%d, lr=%s, epochs=%d",
        model_name,
        dataset_name,
        batch_size,
        learning_rate,
        epochs,
    )

    skf = StratifiedKFold(n_splits=int(train_config.get("nfold", 5)), shuffle=True, random_state=42)
    sampler = CustomRandomUnderSampler(random_state=42)

    fold_performances = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, classes)):
        Xtrain = features[train_idx]
        ytrain = classes[train_idx]
        Xval = features[val_idx]
        yval = classes[val_idx]

        Xtrain_resampled, ytrain_resampled = sampler.fit_resample(Xtrain, ytrain)
        Xval_resampled, yval_resampled = sampler.fit_resample(Xval, yval)

        Xtrain_resampled = Xtrain_resampled.astype(np.float32)
        ytrain_resampled = ytrain_resampled.astype(np.float32)
        Xval_resampled = Xval_resampled.astype(np.float32)
        yval_resampled = yval_resampled.astype(np.float32)

        hours = train_config.get("hours")
        if hours not in (None, "None"):
            hours = int(hours)
            if hours != Xtrain_resampled.shape[1]:
                Xtrain_resampled = Xtrain_resampled[:, :hours, :]
                Xval_resampled = Xval_resampled[:, :hours, :]

        input_length = Xtrain_resampled.shape[1]
        input_dim = Xtrain_resampled.shape[-1]

        logging.info("Fold %d: input_length=%d, input_dim=%d", fold, input_length, input_dim)

        model = select_model(input_dim, input_length, model_config, dataset_name, device)
        if compile_enabled and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
            except Exception as exc:
                logging.warning("torch.compile failed, fallback to eager. Error: %s", exc)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_model(
            model,
            criterion,
            optimizer,
            Xtrain_resampled,
            ytrain_resampled,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            amp_enabled=amp_enabled,
            scaler=scaler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        scores = evaluate_model(
            model,
            Xval_resampled,
            yval_resampled,
            device=device,
            batch_size=eval_batch_size,
            amp_enabled=amp_enabled,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        labels = scores > 0.5

        auroc, auprc, accuracy, sensitivity, specificity, f_measure = evaluate_sepsis_score(
            yval_resampled,
            labels,
            scores,
        )

        fold_performances.append((auroc, auprc, accuracy, sensitivity, specificity, f_measure))

        model_path = save_dir / f"model{dataset_name}_{model_name}_{fold}_best_model.pth"
        torch.save(model.state_dict(), model_path)

        logging.info(
            "Fold %d Results | AUROC=%.4f AUPRC=%.4f ACC=%.4f SEN=%.4f SPE=%.4f F1=%.4f",
            fold,
            auroc,
            auprc,
            accuracy,
            sensitivity,
            specificity,
            f_measure,
        )

    metrics = np.array(fold_performances)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)

    summary = (
        f"Dataset: {dataset_name} - Model: {model_name} - "
        f"AUROC: {means[0]:.4f}±{stds[0]:.4f}, "
        f"AUPRC: {means[1]:.4f}±{stds[1]:.4f}, "
        f"Accuracy: {means[2]:.4f}±{stds[2]:.4f}, "
        f"F-Measure: {means[5]:.4f}±{stds[5]:.4f}, "
        f"Sensitivity: {means[3]:.4f}±{stds[3]:.4f}, "
        f"Specificity: {means[4]:.4f}±{stds[4]:.4f}"
    )

    logging.info("Average Results: %s", summary)
    with open(root / "results_tnam.txt", "a", encoding="utf-8") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    main()
