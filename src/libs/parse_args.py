import argparse
from typing import Optional


def parse_args() -> Optional[argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--data_name", type=str, required=True, choices=["criteo", "criteo_kaggle"]
    )
    parser.add_argument("--model", type=str, required=True, choices=["lr", "fm"])
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--is_test", action="store_true")
    return parser.parse_args()
