import argparse
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("positional", type=str, default=None, help="Positional...")
parser.add_argument("--testing", type=int, default=69, help="Idk man...")
parser.add_argument("--boolean", action="store_true", help="Some boolean lol")

args = parser.parse_args(["butter", "--boolean"])

if __name__ == "__main__":
    print(args.positional)
    print(args.testing)
    print(args.boolean)

    print(np.__version__)
