import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--weights",
        required=False,
        help="Path to an already existing weights file",
    )
    args = parser.parse_args()

    return args
