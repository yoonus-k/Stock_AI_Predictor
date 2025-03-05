import argparse
import sys
from cli import main as cli_main

def main():
    parser = argparse.ArgumentParser(description="AI Trading CLI")
    parser.add_argument("--task", choices=["classify", "mine", "scale", "sentiment"], required=True, help="Select a task to perform")
    
    args = parser.parse_args()

    if args.task == "classify":
        print("Running classifier...")
        cli_main.classify()
    elif args.task == "mine":
        print("Running pattern mining...")
        cli_main.mine()
    elif args.task == "scale":
        print("Running data scaling...")
        cli_main.scale()
    elif args.task == "sentiment":
        print("Running sentiment analysis...")
        cli_main.sentiment()

if __name__ == "__main__":
    main()
