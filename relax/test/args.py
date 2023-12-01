from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Corvo deepspeed mii service")
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        help="Port to run the service on",
        default="8000",
    )
    parser.add_argument(
        "--model-repository",
        type=str,
        required=False,
        help="Path to the model repository where models should be loaded from",
        default="/models"
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()