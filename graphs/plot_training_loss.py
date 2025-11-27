import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- Configuration ---
RUNS_DIR = "../runs"  # Directory where TensorBoard logs are stored
OUTPUT_DIR = "../outputs/mt5-english-tulu"  # Check for training_args.json here too

# --- Styling ---
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def find_latest_event_file(runs_dir):
    """Find the most recent TensorBoard event file"""
    pattern = os.path.join(runs_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(pattern, recursive=True)
    if not event_files:
        return None
    # Sort by modification time and return the latest
    return max(event_files, key=os.path.getmtime)

def extract_loss_from_tensorboard(event_file):
    """Extract training loss data from TensorBoard event file"""
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Get available tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available scalar tags: {scalar_tags}")
    
    # Common loss tag names in simpletransformers/transformers
    loss_tags = [tag for tag in scalar_tags if 'loss' in tag.lower()]
    
    if not loss_tags:
        print("No loss data found in TensorBoard logs")
        return None, None
    
    # Use the first loss tag found
    loss_tag = loss_tags[0]
    print(f"Using loss tag: {loss_tag}")
    
    loss_data = ea.Scalars(loss_tag)
    steps = [entry.step for entry in loss_data]
    losses = [entry.value for entry in loss_data]
    
    return steps, losses

def extract_loss_from_log_file(log_file):
    """Extract epoch and running loss from training log file"""
    import re
    epochs = []
    losses = []
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None, None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match lines like: "Epochs 1/10. Running Loss:    5.1570:"
            match = re.search(r'Epochs?\s+(\d+)/\d+\.\s+Running Loss:\s+([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    
    return epochs, losses

def plot_training_loss():
    """Generate training loss plot"""
    # Try to get data from TensorBoard logs first
    event_file = find_latest_event_file(RUNS_DIR)
    steps, losses = None, None
    
    if event_file:
        print(f"Found TensorBoard event file: {event_file}")
        steps, losses = extract_loss_from_tensorboard(event_file)
    
    # If TensorBoard logs not available, try log file
    if steps is None or losses is None:
        print("Trying to extract from training log file...")
        log_files = ['training_output.log', 'training_english_tulu.log']
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"Found log file: {log_file}")
                steps, losses = extract_loss_from_log_file(log_file)
                if steps and losses:
                    break
    
    if not steps or not losses:
        print("Error: No training loss data found!")
        print("Please ensure:")
        print(f"  1. TensorBoard logs exist in '{RUNS_DIR}/' directory, OR")
        print("  2. Training log file (training_output.log or training_english_tulu.log) exists")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Epoch' if len(steps) <= 20 else 'Training Steps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Time - English to Tulu Translation Model', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations for each point
    for i, (step, loss) in enumerate(zip(steps, losses)):
        plt.annotate(f'{loss:.2f}', 
                    xy=(step, loss), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    output_file = 'training_loss_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Training loss plot saved to graphs/{output_file}")
    
    # Also create a table with the data
    print("\n" + "="*50)
    print("Training Loss Summary:")
    print("="*50)
    for step, loss in zip(steps, losses):
        label = 'Epoch' if len(steps) <= 20 else 'Step'
        print(f"{label} {step:3d}: Loss = {loss:.4f}")
    print("="*50)
    
    # Calculate improvement
    if len(losses) > 1:
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"\nImprovement: {improvement:.1f}% reduction in loss")
        print(f"Initial Loss: {initial_loss:.4f}")
        print(f"Final Loss: {final_loss:.4f}")

if __name__ == '__main__':
    print("Generating training loss plot...")
    plot_training_loss()
    print("\nDone!")
