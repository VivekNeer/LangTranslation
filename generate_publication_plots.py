"""
Generate publication-quality BLEU score plots for research paper
Follows IEEE and ACL publication standards with professional styling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Publication parameters
FIGURE_WIDTH = 7  # inches (IEEE standard single column width: 3.5", double: 7")
FIGURE_HEIGHT = 4.5
DPI = 300  # High resolution for publication
FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_SIZE = 10

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': FONT_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.titlesize': TITLE_SIZE,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def create_bleu_comparison_bar_chart():
    """
    Create a professional bar chart comparing BLEU scores across different models/approaches
    Perfect for research paper comparison sections
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Example data - Replace with your actual model comparisons
    models = ['Baseline\nMT5', 'Fine-tuned\nMT5-Small\n(Our Model)', 'Google\nTranslate', 'Rule-based\nTranslator', 'Previous\nWork [1]']
    bleu_scores = [3.2, 8.40, 12.5, 2.8, 6.1]  # Replace with actual scores
    
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#95a5a6', '#95a5a6']
    colors[1] = '#2ecc71'  # Highlight our model in green
    
    bars = ax.bar(models, bleu_scores, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=FONT_SIZE, fontweight='bold')
    
    # Styling
    ax.set_ylabel('BLEU Score', fontweight='bold')
    ax.set_xlabel('Translation Model', fontweight='bold')
    ax.set_title('English-Tulu Translation Performance Comparison', fontweight='bold', pad=15)
    ax.set_ylim(0, max(bleu_scores) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line for baseline
    ax.axhline(y=np.mean(bleu_scores), color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Average BLEU')
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('publication_bleu_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('publication_bleu_comparison.pdf', bbox_inches='tight')  # Vector format for LaTeX
    print("✓ Generated: publication_bleu_comparison.png & .pdf")
    plt.close()


def create_training_progress_plot():
    """
    Create training progress plot showing BLEU score improvement over epochs
    Essential for demonstrating model convergence
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Example training data - Replace with your actual training logs
    epochs = np.arange(1, 11)
    train_bleu = [2.1, 3.8, 5.2, 6.4, 7.1, 7.8, 8.2, 8.3, 8.4, 8.4]
    val_bleu = [2.0, 3.5, 4.9, 6.0, 6.8, 7.3, 7.7, 7.9, 8.1, 8.0]
    
    # Plot with confidence intervals (if available)
    ax.plot(epochs, train_bleu, marker='o', label='Training BLEU', color='#3498db', linewidth=2.5)
    ax.plot(epochs, val_bleu, marker='s', label='Validation BLEU', color='#e74c3c', linewidth=2.5)
    
    # Add shaded regions for standard deviation (if available)
    # train_std = np.array([0.3, 0.4, 0.3, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05])
    # ax.fill_between(epochs, np.array(train_bleu) - train_std, np.array(train_bleu) + train_std, 
    #                  alpha=0.2, color='#3498db')
    
    # Mark best performance
    best_epoch = np.argmax(val_bleu) + 1
    best_score = max(val_bleu)
    ax.plot(best_epoch, best_score, marker='*', markersize=15, color='#f39c12', 
            label=f'Best Model (Epoch {best_epoch})', zorder=5)
    
    # Styling
    ax.set_xlabel('Training Epoch', fontweight='bold')
    ax.set_ylabel('BLEU Score', fontweight='bold')
    ax.set_title('Model Training Progress on English-Tulu Translation', fontweight='bold', pad=15)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, max(train_bleu + val_bleu) * 1.1)
    
    plt.tight_layout()
    plt.savefig('publication_training_progress.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('publication_training_progress.pdf', bbox_inches='tight')
    print("✓ Generated: publication_training_progress.png & .pdf")
    plt.close()


def create_metrics_comparison_radar():
    """
    Create radar chart comparing multiple translation quality metrics
    Shows comprehensive model performance
    """
    from matplotlib.patches import Circle
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), subplot_kw=dict(projection='polar'))
    
    # Metrics to compare
    metrics = ['BLEU\nScore', 'METEOR', 'ChrF++', 'Exact\nMatch', 'Char\nAccuracy']
    num_vars = len(metrics)
    
    # Our model scores (normalized to 0-100 scale)
    our_scores = [42.0, 35.2, 45.8, 20.2, 83.3]  # Replace with actual normalized scores
    baseline_scores = [16.0, 22.1, 30.5, 8.5, 65.2]  # Baseline comparison
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    our_scores += our_scores[:1]
    baseline_scores += baseline_scores[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, our_scores, 'o-', linewidth=2.5, label='Our Model (MT5-Small Fine-tuned)', 
            color='#2ecc71', markersize=8)
    ax.fill(angles, our_scores, alpha=0.25, color='#2ecc71')
    
    ax.plot(angles, baseline_scores, 's-', linewidth=2.5, label='Baseline MT5', 
            color='#e74c3c', markersize=8)
    ax.fill(angles, baseline_scores, alpha=0.15, color='#e74c3c')
    
    # Fix axis to go from 0-100
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=FONT_SIZE)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=FONT_SIZE - 2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Title and legend
    ax.set_title('Comprehensive Translation Quality Metrics', fontweight='bold', pad=20, 
                 fontsize=TITLE_SIZE)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('publication_metrics_radar.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('publication_metrics_radar.pdf', bbox_inches='tight')
    print("✓ Generated: publication_metrics_radar.png & .pdf")
    plt.close()


def create_performance_by_length():
    """
    Create scatter plot showing BLEU score vs input sentence length
    Important analysis for understanding model behavior
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Example data - Replace with actual analysis
    np.random.seed(42)
    sentence_lengths = np.random.randint(3, 50, 150)
    bleu_scores = 12 - 0.1 * sentence_lengths + np.random.normal(0, 2, 150)
    bleu_scores = np.clip(bleu_scores, 0, 30)
    
    # Create scatter with density coloring
    scatter = ax.scatter(sentence_lengths, bleu_scores, c=bleu_scores, 
                        cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(sentence_lengths, bleu_scores, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(sentence_lengths), max(sentence_lengths), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Polynomial Trend')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('BLEU Score', rotation=270, labelpad=20, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Input Sentence Length (words)', fontweight='bold')
    ax.set_ylabel('BLEU Score', fontweight='bold')
    ax.set_title('Translation Quality vs Input Length', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('publication_performance_by_length.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('publication_performance_by_length.pdf', bbox_inches='tight')
    print("✓ Generated: publication_performance_by_length.png & .pdf")
    plt.close()


def create_dataset_statistics():
    """
    Create visualization of dataset statistics
    Essential for reproducibility section
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH * 1.4, FIGURE_HEIGHT))
    
    # Dataset split distribution
    splits = ['Training', 'Validation', 'Test']
    sizes = [4150, 500, 500]  # Replace with actual numbers
    colors_split = ['#3498db', '#e74c3c', '#f39c12']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=splits, colors=colors_split, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontweight': 'bold'})
    ax1.set_title('Dataset Split Distribution', fontweight='bold', pad=15)
    
    # Sentence length distribution
    np.random.seed(42)
    lengths_train = np.random.gamma(5, 3, 4150)
    lengths_val = np.random.gamma(5, 3, 500)
    lengths_test = np.random.gamma(5, 3, 500)
    
    ax2.hist([lengths_train, lengths_val, lengths_test], bins=30, label=splits, 
            color=colors_split, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Sentence Length (words)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Sentence Length Distribution', fontweight='bold', pad=15)
    ax2.legend(framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('publication_dataset_stats.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('publication_dataset_stats.pdf', bbox_inches='tight')
    print("✓ Generated: publication_dataset_stats.png & .pdf")
    plt.close()


def create_combined_publication_figure():
    """
    Create a comprehensive multi-panel figure combining all key metrics
    Perfect as a main figure in the results section
    """
    fig = plt.figure(figsize=(FIGURE_WIDTH * 1.8, FIGURE_HEIGHT * 2))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel A: Model Comparison Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    models = ['Baseline\nMT5', 'Our Model\n(Fine-tuned)', 'Google\nTranslate', 'Previous\nWork']
    bleu_scores = [3.2, 8.40, 12.5, 6.1]
    colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#95a5a6']
    bars = ax1.bar(models, bleu_scores, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=FONT_SIZE, fontweight='bold')
    ax1.set_ylabel('BLEU Score', fontweight='bold')
    ax1.set_title('(a) Model Performance Comparison', fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Panel B: Training Progress
    ax2 = fig.add_subplot(gs[1, 0])
    epochs = np.arange(1, 11)
    train_bleu = [2.1, 3.8, 5.2, 6.4, 7.1, 7.8, 8.2, 8.3, 8.4, 8.4]
    val_bleu = [2.0, 3.5, 4.9, 6.0, 6.8, 7.3, 7.7, 7.9, 8.1, 8.0]
    ax2.plot(epochs, train_bleu, marker='o', label='Train', color='#3498db', linewidth=2)
    ax2.plot(epochs, val_bleu, marker='s', label='Validation', color='#e74c3c', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('BLEU Score', fontweight='bold')
    ax2.set_title('(b) Training Progress', fontweight='bold', loc='left')
    ax2.legend(framealpha=0.9, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Panel C: Quality Metrics
    ax3 = fig.add_subplot(gs[1, 1])
    metrics = ['BLEU', 'METEOR', 'ChrF++', 'Exact\nMatch', 'Char\nAcc']
    our_scores = [8.40, 15.2, 25.8, 20.2, 83.3]
    baseline = [3.2, 8.1, 15.5, 8.5, 65.2]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, our_scores, width, label='Our Model', color='#2ecc71', 
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax3.bar(x + width/2, baseline, width, label='Baseline', color='#95a5a6',
           edgecolor='black', linewidth=0.8, alpha=0.85)
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.set_title('(c) Quality Metrics Comparison', fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=FONT_SIZE - 1)
    ax3.legend(framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    # Panel D: Performance by Length
    ax4 = fig.add_subplot(gs[2, 0])
    np.random.seed(42)
    lengths = np.random.randint(3, 50, 100)
    scores = 12 - 0.1 * lengths + np.random.normal(0, 1.5, 100)
    scores = np.clip(scores, 0, 20)
    scatter = ax4.scatter(lengths, scores, c=scores, cmap='RdYlGn', alpha=0.6, 
                         s=40, edgecolors='black', linewidth=0.5)
    z = np.polyfit(lengths, scores, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(lengths), max(lengths), 100)
    ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    ax4.set_xlabel('Input Length (words)', fontweight='bold')
    ax4.set_ylabel('BLEU Score', fontweight='bold')
    ax4.set_title('(d) Performance vs Input Length', fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Panel E: Dataset Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    categories = ['Train', 'Val', 'Test']
    samples = [4150, 500, 500]
    colors_cat = ['#3498db', '#e74c3c', '#f39c12']
    bars = ax5.bar(categories, samples, color=colors_cat, edgecolor='black', 
                   linewidth=0.8, alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=FONT_SIZE, fontweight='bold')
    ax5.set_ylabel('Number of Samples', fontweight='bold')
    ax5.set_title('(e) Dataset Distribution', fontweight='bold', loc='left')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.set_axisbelow(True)
    
    # Add overall title
    fig.suptitle('English-Tulu Neural Machine Translation: Comprehensive Results', 
                fontsize=TITLE_SIZE + 2, fontweight='bold', y=0.995)
    
    plt.savefig('publication_combined_figure.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('publication_combined_figure.pdf', bbox_inches='tight')
    print("✓ Generated: publication_combined_figure.png & .pdf (MAIN FIGURE)")
    plt.close()


def create_attention_visualization():
    """
    Create attention heatmap visualization for model interpretability
    Shows which source words the model focuses on
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Example attention weights - Replace with actual model attention
    source_words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    target_words = ['ಬೆಕ್ಕು', 'ಚಾಪೆಯ', 'ಮೇಲೆ', 'ಕುಳಿತಿದೆ']
    
    attention = np.random.rand(len(target_words), len(source_words))
    attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize
    
    # Create heatmap
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    
    # Set ticks
    ax.set_xticks(np.arange(len(source_words)))
    ax.set_yticks(np.arange(len(target_words)))
    ax.set_xticklabels(source_words)
    ax.set_yticklabels(target_words)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add attention scores as text
    for i in range(len(target_words)):
        for j in range(len(source_words)):
            text = ax.text(j, i, f'{attention[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=FONT_SIZE - 2)
    
    # Labels and title
    ax.set_xlabel('Source (English)', fontweight='bold')
    ax.set_ylabel('Target (Tulu)', fontweight='bold')
    ax.set_title('Attention Weights Visualization', fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('publication_attention_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('publication_attention_heatmap.pdf', bbox_inches='tight')
    print("✓ Generated: publication_attention_heatmap.png & .pdf")
    plt.close()


def main():
    """Generate all publication-quality plots"""
    print("=" * 60)
    print("Generating Publication-Quality Plots for Research Paper")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path('publication_figures')
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    create_bleu_comparison_bar_chart()
    create_training_progress_plot()
    create_metrics_comparison_radar()
    create_performance_by_length()
    create_dataset_statistics()
    create_combined_publication_figure()
    create_attention_visualization()
    
    print()
    print("=" * 60)
    print("✓ All plots generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  • publication_bleu_comparison.png/.pdf")
    print("  • publication_training_progress.png/.pdf")
    print("  • publication_metrics_radar.png/.pdf")
    print("  • publication_performance_by_length.png/.pdf")
    print("  • publication_dataset_stats.png/.pdf")
    print("  • publication_combined_figure.png/.pdf (RECOMMENDED MAIN FIGURE)")
    print("  • publication_attention_heatmap.png/.pdf")
    print()
    print("LaTeX Usage Example:")
    print("  \\begin{figure}[t]")
    print("    \\centering")
    print("    \\includegraphics[width=\\linewidth]{publication_combined_figure.pdf}")
    print("    \\caption{Comprehensive evaluation of our English-Tulu NMT system.}")
    print("    \\label{fig:main_results}")
    print("  \\end{figure}")
    print()
    print("Note: Replace example data with your actual results!")


if __name__ == "__main__":
    main()
