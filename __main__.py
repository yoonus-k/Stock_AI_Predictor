import argparse
import sys
from cli import main as cli_main

def main():
    parser = argparse.ArgumentParser(description="AI Trading CLI")
    parser.add_argument('--task', choices=["Predict", "Exit", "Help"], default="Predict")

    
    args = parser.parse_args()

    if args.task == "Predict":
        cli_main()
    elif args.task == "Exit":
        sys.exit(0)
    elif args.task == "Help":
        parser.print_help()
    else:
        print("Invalid task. Use --task to specify a valid task.")

if __name__ == "__main__":
    main()
