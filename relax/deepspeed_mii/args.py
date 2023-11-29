from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("")
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default="8000",
    )
    parser.add_argument(
        "--model-repository",
        type=str,
        required=False,
        default="/models"
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
