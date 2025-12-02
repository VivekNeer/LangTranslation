#!/usr/bin/env python
"""
TensorBoard Dashboard Launcher
Visualizes training metrics for the English->Tulu translation model
"""

import subprocess
import sys
import os
from pathlib import Path

# Configuration
RUNS_DIR = "runs"
PORT = 6006

def check_tensorboard_installed():
    """Check if TensorBoard is installed"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "tensorboard", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def launch_tensorboard(logdir=RUNS_DIR, port=PORT):
    """Launch TensorBoard server"""
    if not os.path.exists(logdir):
        print(f"❌ Error: Log directory '{logdir}' not found!")
        print("Make sure you have trained the model and logs are in the 'runs/' directory.")
        return
    
    # Count available runs
    runs = list(Path(logdir).glob("*"))
    if not runs:
        print(f"❌ Error: No training runs found in '{logdir}'")
        return
    
    print("=" * 60)
    print("🚀 TensorBoard Dashboard Launcher")
    print("=" * 60)
    print()
    print(f"📊 Found {len(runs)} training run(s)")
    print()
    print("The dashboard will show:")
    print("  ✓ Training Loss over time")
    print("  ✓ Learning Rate schedule")
    print("  ✓ Step-by-step metrics")
    print("  ✓ Compare multiple runs")
    print()
    print(f"🌐 Dashboard URL: http://localhost:{port}")
    print()
    print("💡 Tips:")
    print("  - Refresh your browser if data doesn't appear immediately")
    print("  - Use the 'Scalars' tab to see loss graphs")
    print("  - Click on runs to compare them")
    print()
    print("⏸️  Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Launch TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard",
            "--logdir", logdir,
            "--port", str(port),
            "--bind_all"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ TensorBoard server stopped")
    except Exception as e:
        print(f"\n❌ Error launching TensorBoard: {e}")

def main():
    if not check_tensorboard_installed():
        print("❌ TensorBoard is not installed!")
        print("\nInstall it with:")
        print("  pip install tensorboard")
        sys.exit(1)
    
    launch_tensorboard()

if __name__ == "__main__":
    main()
