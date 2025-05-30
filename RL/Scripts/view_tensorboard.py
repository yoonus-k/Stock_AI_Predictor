import argparse
import subprocess
from pathlib import Path

def launch_tensorboard(log_dir=None, port=6006):
    """
    Launch TensorBoard to visualize training logs
    
    Parameters:
        log_dir: Directory containing TensorBoard logs. Defaults to RL/Logs/tensorboard
        port: Port to run TensorBoard on
    """
    # Set default log directory if not specified
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "Logs" / "tensorboard"
    
    log_dir = Path(log_dir)
    
    # Ensure log directory exists
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        print("Please ensure that you have trained a model with TensorBoard logging enabled.")
        return False
    
    print(f"Launching TensorBoard with logs from: {log_dir}")
    print(f"TensorBoard will be available at: http://localhost:{port}")
    print("\nPress Ctrl+C to stop TensorBoard")
    
    # Launch TensorBoard
    try:
        subprocess.run(
            ["tensorboard", "--logdir", str(log_dir), "--port", str(port)],
            check=True
        )
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except subprocess.CalledProcessError:
        print("❌ Error launching TensorBoard. Please ensure TensorBoard is installed:")
        print("Run: pip install tensorboard")
    except FileNotFoundError:
        print("❌ TensorBoard not found. Please install it:")
        print("Run: pip install tensorboard")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch TensorBoard to visualize training logs")
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=None,
        help="Directory containing TensorBoard logs"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006,
        help="Port to run TensorBoard on"
    )
    
    args = parser.parse_args()
    launch_tensorboard(args.log_dir, args.port)
