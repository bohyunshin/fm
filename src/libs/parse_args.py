import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteo_data_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["lr"])
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--is_test", action="store_true")
    return parser.parse_args()
