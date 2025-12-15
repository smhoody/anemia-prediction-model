import sys
import os

from typing import List


MODEL_OUTPUT_PATH = "anemia_model.pt"
EXPECTED_COLS = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV", "Result"]
DATA_FILE_PATH = "anemia.csv"

def verify_data_labels(columns) -> List[str]:
    """Ensure column order and extract arrays"""
    for c in EXPECTED_COLS:
        if c not in columns:
            raise KeyError(f"Expected column {c} in dataframe")
    return EXPECTED_COLS


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


def get_data():
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        FILE_PATH = "anemia.csv"
        return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, 
                                    "biswaranjanrao/anemia-dataset", FILE_PATH)
    except Exception:
        raise FileNotFoundError("kagglehub not available or failed")

def set_train_args_from_sys(hyper_params):
    """ Hyper_params fields are changed by reference ! """
    try:
        if "-epochs" in sys.argv:
            epoch_idx = sys.argv.index("-epochs")
            hyper_params.epochs = int(sys.argv[epoch_idx+1])
        if "-lr" in sys.argv:
            lr_idx = sys.argv.index("-lr")
            hyper_params.lr = float(sys.argv[lr_idx+1])
    except Exception:
        raise ValueError(f"Expected format as: py {os.path.basename(__file__)} \
                         -epochs [int] -lr [float]")
