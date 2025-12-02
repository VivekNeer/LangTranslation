#!/usr/bin/env python3
"""
Master script to regenerate all visualizations from available data sources
Run this after training or evaluation to update all plots
"""

import os
import sys
from pathlib import Path

def main():
    """Run all plotting scripts in sequence"""
    graphs_dir = Path(__file__).parent / "graphs"
    
    print("=" * 70)
    print("🎨 VISUALIZATION MASTER SCRIPT")
    print("=" * 70)
    print()
    
    scripts = [
        ("plot_tensorboard_csv.py", "TensorBoard CSV plots"),
        ("plot_training_loss.py", "Training loss plot"),
        ("plot_bleu_score.py", "BLEU score comparison"),
    ]
    
    success_count = 0
    fail_count = 0
    
    for script, description in scripts:
        script_path = graphs_dir / script
        
        if not script_path.exists():
            print(f"⚠️  {script} not found, skipping...")
            continue
        
        print(f"📊 Generating: {description}")
        print(f"   Running: {script}")
        print("-" * 70)
        
        # Change to graphs directory and run script
        original_dir = os.getcwd()
        try:
            os.chdir(graphs_dir)
            
            # Import and run the script
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_module", script)
            module = importlib.util.module_from_spec(spec)
            
            try:
                spec.loader.exec_module(module)
                print(f"✅ {description} - SUCCESS")
                success_count += 1
            except Exception as e:
                print(f"❌ {description} - FAILED: {e}")
                fail_count += 1
        finally:
            os.chdir(original_dir)
        
        print()
    
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📁 Output directory: {graphs_dir}")
    print()
    
    # List generated files
    print("Generated files:")
    for file in sorted(graphs_dir.glob("*.png")):
        size = file.stat().st_size
        size_str = f"{size / 1024:.0f}KB" if size < 1024*1024 else f"{size / (1024*1024):.1f}MB"
        print(f"  • {file.name} ({size_str})")
    
    print()
    print("=" * 70)
    print("✅ Visualization generation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
