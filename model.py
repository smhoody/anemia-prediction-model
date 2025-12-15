"""Tabular PyTorch model for anemia prediction.

This module provides:
- `TabularDataset` : a simple Dataset wrapper for (X, y) arrays
- `AnemiaModel` : a small MLP for tabular data
- `train`, `evaluate`, `predict` : helpers to train/eval/predict
- `load_model_and_predict` : load a saved model and make predictions on new data

Example usage:
                py model.py   (show help menu)
        py model.py --train   (train model)
py model.py --test [sample]   (test existing model on sample)

"""

from __future__ import annotations

import sys, os
from typing import Optional, Tuple, Dict, Any

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import optuna
from torch import nn
from torch.utils.data import Dataset, DataLoader
import util

MODEL_OUTPUT_PATH = util.MODEL_OUTPUT_PATH

class HyperParams():
    def __init__(self, epochs=20, lr=1e-3, batch_size=64):
        self.epochs = epochs 
        self.lr = lr
        self.batch_size = batch_size
    
    def to_dict(self):
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size
        }

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None
        if y is not None:
            y = np.asarray(y)
            self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class AnemiaModel(nn.Module):
    def __init__(
        self, input_dim: int = 5, hidden_dims=(64, 32), dropout: float = 0.1
    ):
        """
        Small MLP for tabular data.

        Args:
            input_dim: number of input features (default 5)
            hidden_dims: tuple of hidden layer sizes
            dropout: float of dropout fraction
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    params: Dict[str, Any],
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"]) # type: ignore
    final_loss = 0

    # Log params to the currently active MLflow run (do not start a new run here).
    mlflow.log_params(params)
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)

        train_loss = running / max(1, count)
        final_loss = train_loss
        
        mlflow.log_metrics(
            {"train_loss": train_loss}, step=epoch
        )

    mlflow.pytorch.log_model(model, name=model.__class__.__name__)
    
    if mean is not None and std is not None:
        for i in range(mean.shape[1]):
            mlflow.log_param(f"mean_{i}", float(mean[0, i]))
            mlflow.log_param(f"std_{i}", float(std[0, i]))

    return final_loss


def evaluate(
        model: nn.Module, 
        loader: DataLoader, 
        device: Optional[torch.device] = None, 
        problem_type: str = "binary"
) -> Dict[str, Any]:
    """
    Evaluates loss & accuracy
    
    :returns: {"loss": float, "accuracy", float}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if problem_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    running = 0.0
    count = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            running += loss.item() * xb.size(0)
            count += xb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    loss = running / max(1, count)
    preds = np.vstack(all_preds) if all_preds else np.empty((0, 1))
    targets = np.vstack(all_targets) if all_targets else np.empty((0, 1))
    metrics: Dict[str, Any] = {"loss": loss}

    if problem_type == "binary" and len(preds) > 0:
        probs = 1 / (1 + np.exp(-preds))
        preds_bin = (probs >= 0.5).astype(int)
        acc = (preds_bin == targets).mean()
        metrics["accuracy"] = float(acc)

    return metrics


def predict(
        model: nn.Module, 
        X: np.ndarray, 
        device: Optional[torch.device] = None, 
        batch_size: int = 256
) -> np.ndarray:
    ds = TabularDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds = model(xb)
            out.append(preds.cpu().numpy())
    if out:
        return np.vstack(out)
    return np.empty((0, 1))


def _train_val_split(
        X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1 - val_frac))
    train_idx = idx[:cut]
    val_idx = idx[cut:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def get_loaders(hyper_params: Dict):
    df = util.get_data()

    EXPECTED_COLS = util.verify_data_labels(df.columns)
    data = df[EXPECTED_COLS].to_numpy()

    X = data[:, :5]
    y = data[:, 5]

    # Simple scaling using training stats
    X_train, X_val, y_train, y_val = _train_val_split(X, y, val_frac=0.2, seed=1)
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std

    train_ds = TabularDataset(X_train_s, y_train)
    val_ds = TabularDataset(X_val_s, y_val)
    train_loader = DataLoader(train_ds, batch_size=hyper_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    return train_loader, val_loader, mean, std


def load_mlflow_model(
    mode: str = "latest", device: Optional[torch.device] = None
) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Load an MLflow model by mode.

    :param mode: "latest" or "best". "latest" loads the most recent run with a logged model.
                 "best" loads the run with the lowest `loss` metric.
    :returns: (model, metadata_dict) where metadata may contain 'mean' and 'std'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = None
    if mode == "best":
        runs = mlflow.search_runs(max_results=1000)
        if runs.empty:
            raise ValueError("No MLflow runs found. Please train a model first.")
        runs_with_loss = runs[runs["metrics.loss"].notna()].copy()
        if runs_with_loss.empty:
            raise ValueError("No runs with loss metric found. Please train a model first.")
        runs_with_loss = runs_with_loss.sort_values("metrics.loss")
        best_run = runs_with_loss.iloc[0]
        run_id = best_run["run_id"]
        best_loss = best_run["metrics.loss"]
        print(f"Loading best model from run {run_id} with loss={best_loss:.6f}")
    else:
        runs = mlflow.search_runs(max_results=100, order_by=["start_time DESC"])
        if runs.empty:
            raise ValueError("No MLflow runs found. Please train a model first.")
        # take the first run that successfully loads a model
        for _, run in runs.iterrows():
            candidate_run_id = run["run_id"]
            try:
                model_uri = f"runs:/{candidate_run_id}/AnemiaModel"
                loaded = mlflow.pytorch.load_model(model_uri, map_location=device)
                run_id = candidate_run_id
                break
            except Exception:
                continue
        if run_id is None:
            raise ValueError("No valid logged PyTorch model found in recent MLflow runs.")

    # Load the model if not already loaded (for 'best' mode)
    if mode == "best":
        model_uri = f"runs:/{run_id}/AnemiaModel"
        loaded = mlflow.pytorch.load_model(model_uri, map_location=device)

    # Try to load metadata (mean/std) from logged parameters
    metadata = None
    try:
        # Get the run object to access params
        run_info = mlflow.tracking.MlflowClient().get_run(run_id)
        params = run_info.data.params
        
        # Try to extract mean and std arrays from params
        mean_vals = []
        std_vals = []
        for i in range(10):  # Check up to 10 features
            mean_key = f"mean_{i}"
            std_key = f"std_{i}"
            if mean_key in params and std_key in params:
                mean_vals.append(float(params[mean_key]))
                std_vals.append(float(params[std_key]))
            else:
                break
        
        if mean_vals and std_vals:
            metadata = {
                "mean": np.array(mean_vals, dtype=np.float32).reshape(1, -1),
                "std": np.array(std_vals, dtype=np.float32).reshape(1, -1)
            }
    except Exception:
        metadata = None

    return loaded, metadata


def predict_with_mlflow_model(
    X_new: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    mode: str = "latest",
) -> np.ndarray:
    """
    Load the latest MLflow model and predict on new data.
    Scales input using saved training statistics if available.
    
    :param X_new: input features (n_samples, n_features) or (n_features,) for single sample
    :returns: predictions (n_samples, 1)
    """
    model, metadata = load_mlflow_model(mode=mode, device=device)
    
    X_new = np.asarray(X_new)
    
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    
    # Scale if we have mean/std from metadata
    if metadata and "mean" in metadata and "std" in metadata:
        mean = metadata["mean"]
        std = metadata["std"]
        X_new = (X_new - mean) / std
    
    preds = predict(model, X_new, device=device, batch_size=batch_size)
    return preds


def objective(trial):
    #Default params
    hyper_params = HyperParams(
        epochs = 10,
        lr = 1e-3,
        batch_size = 64
    )

    #Get any arguments passed from terminal
    util.set_train_args_from_sys(hyper_params)


    with mlflow.start_run(nested=True):
        n_layers = trial.suggest_int("n_layers", 1, 5)
        hidden_dims = tuple(
            trial.suggest_int(f"hidden_dim_{i}", 16, 256)
            for i in range(n_layers)
        )

        hyper_params = {
            "epochs": trial.suggest_int("epochs", 10, 100),
            "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4)
        }
        mlflow.log_params(hyper_params)
        mlflow.log_param("n_layers", n_layers)
        for i, h in enumerate(hidden_dims):
            mlflow.log_param(f"hidden_dim_{i}", h)

        train_loader, val_loader, mean, std = get_loaders(hyper_params)

        model = AnemiaModel(
            input_dim=5, hidden_dims=hidden_dims, dropout=hyper_params["dropout"]
        )

        loss = train(
            model,
            train_loader,
            params=hyper_params,
            mean=mean,
            std=std
        )
        
        # Log scaling stats from training
        for i in range(mean.shape[1]):
            mlflow.log_param(f"mean_{i}", float(mean[0, i]))
            mlflow.log_param(f"std_{i}", float(std[0, i]))
        
        mlflow.log_metric("loss", loss)

        return loss


def train_model():
    with mlflow.start_run(run_name="Anemia Model HP Optimization"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)

        mlflow.log_params({f"best_{k}": v for k,v in study.best_params.items()})
        mlflow.log_metric("best_loss", study.best_value)


def test_model(mode: str = "latest"):
    """
    Test the model on a new sample.
    
    :param mode: "latest" to load the most recent model, or "best" to load the model with lowest loss
    """
    sample = []
    try: 
        sample = np.array([float(n) for n in sys.argv[2].split(',')])
        assert(len(sample) == 5)
    except Exception as e:
        raise ValueError(
            f"Expected sample usage: py {os.path.basename(__file__)} --test 0,16,29,35.4,82"
        )

    print(f"\n--- Making prediction on new sample using {mode} MLflow model ---")
    
    try:
        model, metadata = load_mlflow_model(mode=mode)
        
        # Reshape and scale the input
        X_new = sample.reshape(1, -1)
        if metadata and "mean" in metadata and "std" in metadata:
            X_new = (X_new - metadata["mean"]) / metadata["std"]
        
        logits = predict(model, X_new)
    except (ValueError, FileNotFoundError) as e:
        raise FileNotFoundError(f"Could not load model from MLflow. Make sure to train first ({e})")
   
    probs = 1 / (1 + np.exp(-logits))
    pred_label = (probs >= 0.5).astype(int)[0, 0]
    print(f"Input features: {sample}")
    print(f"Predicted probability: {probs[0, 0]:0.10f}")
    print(f"Predicted label (binary): {pred_label}")



if __name__ == "__main__":
    arg_funcs = {
        "--train": train_model,
        "--test": lambda: test_model("latest"),
        "--test-best": lambda: test_model("best")
    }

    if len(sys.argv) == 1: 
        help()
        exit()

    result = arg_funcs.get(sys.argv[1], None) 
    
    if not result:
        help()
    else:
        result()