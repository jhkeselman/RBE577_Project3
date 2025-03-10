import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on Kuka pick and place task"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["a3c", "ddpg"],
        required=True,
        help="RL algorithm to use (a3c or ddpg)",
    )
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    if args.algorithm == "a3c":
        from lib.a3c.train import train_a3c

        train_a3c()
    elif args.algorithm == "ddpg":
        from lib.ddpg.train import train_ddpg

        train_ddpg()
    else:
        print(f"Unknown algorithm: {args.algorithm}")
        sys.exit(1)


if __name__ == "__main__":
    main()
