#!/usr/bin/env python3
"""
Generate a professional loss visualization from TensorBoard CSV export
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

def plot_tensorboard_csv(csv_path, output_path=None):
    """
    Create a professional plot from TensorBoard CSV data
    
    Args:
        csv_path: Path to the CSV file exported from TensorBoard
        output_path: Path to save the output image (optional)
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract run name from filename
    csv_filename = os.path.basename(csv_path)
    run_name = csv_filename.replace('.csv', '')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot the loss curve
    ax.plot(df['Step'], df['Value'], linewidth=2, color='#2E86AB', alpha=0.8, label='Training Loss')
    
    # Add smoothed line
    window_size = min(50, len(df) // 10)
    if window_size > 1:
        smoothed = df['Value'].rolling(window=window_size, center=True).mean()
        ax.plot(df['Step'], smoothed, linewidth=3, color='#A23B72', alpha=0.9, label=f'Smoothed (window={window_size})')
    
    # Annotations for key points
    start_loss = df['Value'].iloc[0]
    end_loss = df['Value'].iloc[-1]
    min_loss = df['Value'].min()
    min_step = df.loc[df['Value'].idxmin(), 'Step']
    
    # Annotate start
    ax.annotate(f'Start: {start_loss:.2f}',
                xy=(df['Step'].iloc[0], start_loss),
                xytext=(10, 20),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Annotate minimum
    ax.annotate(f'Min: {min_loss:.2f}',
                xy=(min_step, min_loss),
                xytext=(10, -30),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Annotate end
    ax.annotate(f'End: {end_loss:.2f}',
                xy=(df['Step'].iloc[-1], end_loss),
                xytext=(-80, 20),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Calculate improvement
    improvement = ((start_loss - end_loss) / start_loss) * 100
    
    # Labels and title
    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Training Loss Over Time - {run_name}\nImprovement: {improvement:.1f}% (from {start_loss:.2f} to {end_loss:.2f})',
                 fontsize=15, fontweight='bold', pad=20)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add statistics box
    stats_text = (
        f'Statistics:\n'
        f'Total Steps: {df["Step"].iloc[-1]:,}\n'
        f'Initial Loss: {start_loss:.2f}\n'
        f'Final Loss: {end_loss:.2f}\n'
        f'Min Loss: {min_loss:.2f} (step {min_step:,})\n'
        f'Improvement: {improvement:.1f}%'
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = csv_path.replace('.csv', '_plot.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Also save as high-res version
    high_res_path = output_path.replace('.png', '_highres.png')
    plt.savefig(high_res_path, dpi=600, bbox_inches='tight')
    print(f"✅ High-res plot saved to: {high_res_path}")
    
    plt.close()
    
    return output_path

def main():
    """Main function to process CSV files in the graphs directory"""
    graphs_dir = Path(__file__).parent
    
    # Find all CSV files
    csv_files = list(graphs_dir.glob('*.csv'))
    
    if not csv_files:
        print("❌ No CSV files found in the graphs directory")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    for csv_file in csv_files:
        print(f"\n📊 Processing: {csv_file.name}")
        try:
            output_path = str(csv_file).replace('.csv', '_plot.png')
            plot_tensorboard_csv(str(csv_file), output_path)
        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")

if __name__ == "__main__":
    main()
