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
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


MODEL_OUTPUT_PATH = "anemia_model.pt"
EXPECTED_COLS = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV", "Result"]

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
        self, input_dim: int = 5, hidden_dims=(64, 32), problem_type: str = "binary"
    ):
        """
        Small MLP for tabular data.

        Args:
            input_dim: number of input features (default 5)
            hidden_dims: tuple of hidden layer sizes
            problem_type: 'binary' or 'regression'
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.problem_type = problem_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 300,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    problem_type: str = "binary",
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if problem_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # type: ignore

    history = {"train_loss": [], "val_loss": []}
    for _ in range(1, epochs + 1):
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
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate(
                model, val_loader, device=device, problem_type=problem_type
            )["loss"]
            history["val_loss"].append(val_loss)
        else:
            history["val_loss"].append(None)

    return history


def evaluate(
        model: nn.Module, 
        loader: DataLoader, 
        device: Optional[torch.device] = None, 
        problem_type: str = "binary"
) -> Dict[str, Any]:
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


def load_model_and_predict(
    model_path: str,
    X_new: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Load a saved model and make predictions on new data.

    Args:
        model_path: path to the saved model checkpoint
        X_new: input data, shape (n_samples, 5) with features [Gender, Hemoglobin, MCH,
        MCHC, MCV]
        device: torch device (default: cuda if available, else cpu)
        batch_size: batch size for inference (default 256)

    Returns:
        predictions, shape (n_samples, 1). For binary classification, values are logits
        (use sigmoid to get probabilities).

    Example:
        >>> preds = load_model_and_predict('anemia_model.pt', X_new_scaled)
        >>> probs = 1 / (1 + np.exp(-preds))  # sigmoid for binary classification
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state = checkpoint["model_state"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    # Instantiate model and load weights
    model = AnemiaModel(input_dim=5, hidden_dims=(64, 32), problem_type="binary")
    model.load_state_dict(model_state)
    model.to(device)

    # Scale input using saved training statistics
    X_new = np.asarray(X_new)
    X_new_scaled = (X_new - mean) / std

    # Run prediction
    preds = predict(model, X_new_scaled, device=device, batch_size=batch_size)
    return preds


def verify_data_labels(columns) -> List[str]:
    """Ensure column order and extract arrays"""
    for c in EXPECTED_COLS:
        if c not in columns:
            raise KeyError(f"Expected column {c} in dataframe")
    return EXPECTED_COLS


def train_model():
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        FILE_PATH = "anemia.csv"
        df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, 
                                    "biswaranjanrao/anemia-dataset", FILE_PATH)
    except Exception:
        raise FileNotFoundError("kagglehub not available or failed")

    epochs = 10
    lr = 1e-3
    try:
        if "-epochs" in sys.argv:
            epoch_idx = sys.argv.index("-epochs")
            epochs = int(sys.argv[epoch_idx+1])
        if "-lr" in sys.argv:
            lr_idx = sys.argv.index("-lr")
            lr = float(sys.argv[lr_idx+1])
    except Exception as e:
        raise ValueError(f"Expected format as: py {os.path.basename(__file__)} \
                         -epochs [int] -lr [float]")

    EXPECTED_COLS = verify_data_labels(df.columns)
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
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = AnemiaModel(input_dim=5, hidden_dims=(64, 32), problem_type="binary")
    history = train(
        model, 
        train_loader, 
        val_loader=val_loader, 
        epochs=epochs, lr=lr, 
        problem_type="binary"
    )
    print("Loss history:", [round(l, 5) for l in history["train_loss"]])

    metrics = evaluate(model, val_loader, problem_type="binary")
    print("Validation metrics:", metrics)

    # Save the trained model
    torch.save(
        {
            "model_state": model.state_dict(), 
            "mean": mean, 
            "std": std
        }, 
        MODEL_OUTPUT_PATH
    )
    print(f"Saved model to {MODEL_OUTPUT_PATH}")


def test_model():
    if not os.path.exists(MODEL_OUTPUT_PATH):
        raise FileNotFoundError(
            "Model checkpoint not found. Make sure to train before testing"
        )
    
    sample = []
    try: 
        sample = np.array([float(n) for n in sys.argv[2].split(',')])
        assert(len(sample) == 5)
    except Exception as e:
        raise ValueError(
            f"Expected sample usage: py {os.path.basename(__file__)} --test 0,16,29,35.4,82"
        )

    print("\n--- Making prediction on new sample ---")
    logits = load_model_and_predict(MODEL_OUTPUT_PATH, sample)
    probs = 1 / (1 + np.exp(-logits))
    pred_label = (probs >= 0.5).astype(int)[0, 0]
    print(f"Input features: {sample}")
    print(f"Predicted probability: {probs[0, 0]:0.10f}")
    print(f"Predicted label (binary): {pred_label}")


def help():
    file = os.path.basename(__file__)
    print("\n\t--- Help ---")
    print("\nTrain argument examples:")
    print(f" Default (epochs=10): py {file} --train")
    print(f"       Custom epochs: py {file} --train -epochs 50")
    print(f"Custom learning rate: py {file} --train -lr 0.01")
    print(f"  Custom epochs & lr: py {file} --train -epochs 100 -lr 0.0001")

    print("\nTest argument examples:")
    print(f"          Test model: py {file} --test {",".join(EXPECTED_COLS[:-1])}")
    print(f"          Test model: py {file} --test 0,16,29,35.4,82")
    print("note: gender field has 2 values -- male=0 female=1")

    print("\nTrain Output:")
    print("\tLoss history: list of model losses during each epoch")
    print(
        "\tValidation metrics: 'loss' is the final loss, 'accuracy' \
        is the final model accuracy")
    print(f"\tSaved model to {MODEL_OUTPUT_PATH}: pt model file save location")

    print("\nTest Output:")
    print("\tInput features: list of features given")
    print("\tPredicted probability: probability of patient having anemia")
    print("\tPredicted label (binary): 0 = not anemic, 1 = anemic")


if __name__ == "__main__":
    
    arg_funcs = {
        "--train": train_model,
        "--test": test_model
    }

    if len(sys.argv) == 1: 
        help()
        exit()

    result = arg_funcs.get(sys.argv[1], None) 
    
    if not result:
        help()
    else:
        result()