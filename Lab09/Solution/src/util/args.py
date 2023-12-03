import argparse


def parse_arguments(require_weights = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weights",
        required=require_weights,
        help="Path to an already existing weights file",
    )
    args = parser.parse_args()

    return args
