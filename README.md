# Anemia Prediction Model

Simple PyTorch project to predict anemia `Result` from clinical features.

Project layout
- `model.py` — dataset wrapper (`TabularDataset`), `AnemiaModel` (MLP), training/evaluation helpers (`train`, `evaluate`, `predict`), and `load_model_and_predict` for inference.
- `requirements.txt` — library requirements.

Dataset
- Expected input columns: `Gender` (int), `Hemoglobin` (float), `MCH` (float), `MCHC` (float), `MCV` (float), `Result` (target — binary or continuous depending on dataset).
- The repository example uses the Kaggle dataset [https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset](`biswaranjanrao/anemia-dataset`) when `kagglehub` is available; otherwise a synthetic dataset is generated for demonstration.

Installation

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Add Kaggle API key in model directory
```bash
echo "KAGGLE_API_TOKEN=[token_here]" > .env
```

Quick start — train and save model

Run the example training flow which will attempt to load the dataset via `kagglehub` and otherwise raise an error if not available:


This script will:
- Load the dataset (via `kagglehub`) or expect the dataset to be present
- Split into training/validation sets
- Train a small MLP for binary classification
- Save the checkpoint to `anemia_model.pt` (includes `model_state`, `mean`, `std`)

### Terminal usage example:

```powershell
python model.py                    (show help menu)
python model.py --train            (train model and save model checkpoint)
python model.py --test [values]    (e.g. --test 0,14,34,37,82)
```

Programmatic usage — predict on new samples

Use the provided `load_model_and_predict` helper to load a saved checkpoint and run inference. The helper applies the saved training mean/std scaling automatically.

### Code module example:

```python
from model import load_model_and_predict
import numpy as np

# New sample(s) with features [Gender, Hemoglobin, MCH, MCHC, MCV]
X_new = np.array([[1, 13.5, 28.2, 31.8, 92.1]])

# Returns logits (shape (n_samples, 1)); for probabilities apply sigmoid
logits = load_model_and_predict('anemia_model.pt', X_new)
probs = 1 / (1 + np.exp(-logits))
pred_label = (probs >= 0.5).astype(int)
print(f"Probability: {probs[0,0]:.4f}")
print(f"Predicted class: {pred_label[0,0]}")
```


## Current Results

The highest model accuracy acheived with the current dataset is **`98.947%`**