import argparse
import sys
import subprocess
import cli

def main():
    parser = argparse.ArgumentParser(description="AI Trading CLI/GUI Launcher")
    parser.add_argument('--mode', choices=["cli", "gui", "exit", "help"], default="gui", help="Choose interface mode")

    args = parser.parse_args()

    if args.mode == "cli":
        cli.main()
    elif args.mode == "gui":
        # Run Streamlit GUI
        try:
            subprocess.run(["streamlit", "run", "gui.py"])
        except FileNotFoundError:
            print("❌ Error: Streamlit not found. Please install it using 'pip install streamlit'")
    elif args.mode == "exit":
        sys.exit(0)
    elif args.mode == "help":
        parser.print_help()
    else:
        print("Invalid mode. Use --mode to specify 'cli' or 'gui'.")

if __name__ == "__main__":
    main()
