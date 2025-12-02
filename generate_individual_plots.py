"""
Generate individual PNG files for each evaluation plot.
Creates separate high-quality images for presentations and papers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('graphs/individual_plots')
output_dir.mkdir(parents=True, exist_ok=True)

# Load evaluation results
print("Loading evaluation results...")
import json

with open('comprehensive_evaluation_results.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

# Convert to DataFrame
results_df = pd.DataFrame(eval_data['detailed_results'])

# Calculate metrics
total_samples = len(results_df)
bleu_scores = results_df['bleu_score'].values
exact_matches = (results_df['bleu_score'] == 100).sum()
exact_match_rate = (exact_matches / total_samples) * 100

# Calculate character-level metrics
char_errors = results_df.apply(
    lambda row: sum(c1 != c2 for c1, c2 in zip(row['prediction'], row['target'][:len(row['prediction'])])),
    axis=1
)
char_accuracy = 100 - (char_errors.sum() / results_df['target'].str.len().sum() * 100)

# Calculate length metrics
results_df['input_length'] = results_df['input_text'].str.split().str.len()
results_df['pred_length'] = results_df['prediction'].str.len()
results_df['target_length'] = results_df['target'].str.len()
results_df['length_ratio'] = results_df['pred_length'] / results_df['target_length']

print(f"\nGenerating {9} individual plots...")
print("=" * 60)

# ============================================================================
# Plot 1: Overall Performance Metrics (Radar Chart)
# ============================================================================
print("\n1. Creating Overall Performance Metrics (Radar Chart)...")

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

categories = ['BLEU\nScore', 'Exact\nMatch', 'Length\nAccuracy', 'Char\nAccuracy']
values = [
    bleu_scores.mean(),
    exact_match_rate,
    100 - abs(results_df['length_ratio'].mean() - 1) * 100,
    char_accuracy
]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, color='#5b9bd5', markersize=8)
ax.fill(angles, values, alpha=0.25, color='#5b9bd5')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
ax.grid(True, linestyle='--', alpha=0.7)

# Add values as text
for angle, value, cat in zip(angles[:-1], values[:-1], categories):
    ax.text(angle, value + 5, f'{value:.1f}', 
            ha='center', va='center', size=11, fontweight='bold')

plt.title('Overall Performance Metrics\n(0-100 Scale)', 
          size=16, fontweight='bold', pad=20)

# Add summary box
textstr = f'BLEU: {bleu_scores.mean():.1f}\nExact Match: {exact_match_rate:.1f}%\nLength Ratio: {results_df["length_ratio"].mean():.2f}\nChar Accuracy: {char_accuracy:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / '01_overall_metrics_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 01_overall_metrics_radar.png")

# ============================================================================
# Plot 2: Performance by Input Length (Bar Chart)
# ============================================================================
print("\n2. Creating Performance by Input Length (Bar Chart)...")

# Categorize by length
def categorize_length(length):
    if length <= 20:
        return 'short\n(0-20)'
    elif length <= 50:
        return 'medium\n(21-50)'
    else:
        return 'long\n(51+)'

results_df['length_category'] = results_df['input_length'].apply(categorize_length)

# Calculate metrics by category
length_stats = results_df.groupby('length_category').agg({
    'bleu_score': 'mean',
    'input_text': 'count'
}).reset_index()
length_stats.columns = ['Category', 'BLEU', 'Count']

# Calculate exact match percentage
exact_match_by_length = results_df.groupby('length_category').apply(
    lambda x: (x['bleu_score'] == 100).sum() / len(x) * 100
).reset_index()
exact_match_by_length.columns = ['Category', 'Exact_Match']

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(length_stats))
width = 0.35

bars1 = ax.bar(x - width/2, length_stats['BLEU'], width, 
               label='BLEU Score', color='#5b9bd5', alpha=0.8)
bars2 = ax.bar(x + width/2, exact_match_by_length['Exact_Match'], width,
               label='Exact Match %', color='#c45c8c', alpha=0.8)

ax.set_xlabel('Input Length Category', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Performance by Input Length\n(with sample counts)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f"{cat}\n(n={count})" for cat, count in zip(length_stats['Category'], length_stats['Count'])],
                    fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '02_performance_by_length.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 02_performance_by_length.png")

# ============================================================================
# Plot 3: BLEU Score Distribution (Histogram)
# ============================================================================
print("\n3. Creating BLEU Score Distribution (Histogram)...")

fig, ax = plt.subplots(figsize=(12, 7))

n, bins, patches = ax.hist(bleu_scores, bins=30, color='#5b9bd5', alpha=0.7, edgecolor='black')

# Color bars based on value
for i, patch in enumerate(patches):
    if bins[i] < 10:
        patch.set_facecolor('#e74c3c')
    elif bins[i] < 30:
        patch.set_facecolor('#f39c12')
    else:
        patch.set_facecolor('#27ae60')

mean_bleu = bleu_scores.mean()
median_bleu = np.median(bleu_scores)

ax.axvline(mean_bleu, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_bleu:.2f}')
ax.axvline(median_bleu, color='green', linestyle='--', linewidth=2, label=f'Median: {median_bleu:.2f}')

ax.set_xlabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('BLEU Score Distribution', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add statistics box
stats_text = f'Mean: {mean_bleu:.2f}\nMedian: {median_bleu:.2f}\nStd Dev: {bleu_scores.std():.2f}\nMin: {bleu_scores.min():.2f}\nMax: {bleu_scores.max():.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / '03_bleu_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 03_bleu_distribution.png")

# ============================================================================
# Plot 4: BLEU Score Violin Plot
# ============================================================================
print("\n4. Creating BLEU Score Violin Plot...")

fig, ax = plt.subplots(figsize=(10, 8))

parts = ax.violinplot([bleu_scores], positions=[0], showmeans=True, showmedians=True, widths=0.7)

# Customize violin plot
for pc in parts['bodies']:
    pc.set_facecolor('#5b9bd5')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

parts['cmeans'].set_color('red')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('blue')
parts['cmedians'].set_linewidth(2)
parts['cbars'].set_color('black')
parts['cmaxes'].set_color('black')
parts['cmins'].set_color('black')

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('BLEU Score Violin Plot', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks([0])
ax.set_xticklabels(['All Samples'], fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add statistics box
stats_text = f'Statistics:\nMean: {mean_bleu:.2f}\nMedian: {median_bleu:.2f}\nStd Dev: {bleu_scores.std():.2f}\nMin: {bleu_scores.min():.2f}\nMax: {bleu_scores.max():.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.5, 0.15, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / '04_bleu_violin.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 04_bleu_violin.png")

# ============================================================================
# Plot 5: Translation Quality vs Input Length (Scatter)
# ============================================================================
print("\n5. Creating Quality vs Input Length (Scatter Plot)...")

fig, ax = plt.subplots(figsize=(12, 7))

scatter = ax.scatter(results_df['input_length'], bleu_scores, 
                     c=bleu_scores, cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')

# Add trend line
z = np.polyfit(results_df['input_length'], bleu_scores, 2)
p = np.poly1d(z)
x_trend = np.linspace(results_df['input_length'].min(), results_df['input_length'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')

ax.set_xlabel('Input Length (words)', fontsize=14, fontweight='bold')
ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('Translation Quality vs Input Length', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('BLEU', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '05_quality_vs_length.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 05_quality_vs_length.png")

# ============================================================================
# Plot 6: Predicted vs Reference Length (Scatter)
# ============================================================================
print("\n6. Creating Predicted vs Reference Length (Scatter)...")

fig, ax = plt.subplots(figsize=(12, 7))

scatter = ax.scatter(results_df['target_length'], results_df['pred_length'],
                     c=bleu_scores, cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')

# Add perfect match line
max_len = max(results_df['target_length'].max(), results_df['pred_length'].max())
ax.plot([0, max_len], [0, max_len], 'k--', alpha=0.5, linewidth=2, label='Perfect Match')

ax.set_xlabel('Reference Length (characters)', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Length (characters)', fontsize=14, fontweight='bold')
ax.set_title('Predicted vs Reference Length', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('BLEU Score', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '06_length_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 06_length_comparison.png")

# ============================================================================
# Plot 7: Performance Metrics Table (as Image)
# ============================================================================
print("\n7. Creating Performance Metrics Table...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = [
    ['Metric', 'Value', 'Interpretation'],
    ['BLEU Score', f'{bleu_scores.mean():.2f}', 'Understandable but flawed (10-20 is good)'],
    ['Exact Match Accuracy', f'{exact_match_rate:.2f}%', 'Perfect translations'],
    ['Best on Short Inputs', f'BLEU {results_df[results_df["input_length"] <= 20]["bleu_score"].mean():.2f}', 'Model performs best on short sentences'],
    ['Character Error Rate', f'{100 - char_accuracy:.2f}%', 'Character-level accuracy is good'],
    ['Length Ratio', f'{results_df["length_ratio"].mean():.2f} ± {results_df["length_ratio"].std():.2f}', 'Predicted/Reference length'],
    ['Total Samples', f'{total_samples}', 'Validation set size']
]

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.3, 0.2, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#4472c4')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#e7e6e6')
        else:
            cell.set_facecolor('#f2f2f2')

plt.title('English-to-Tulu Translation Model: Comprehensive Evaluation\nPerformance Metrics Summary',
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / '07_metrics_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 07_metrics_table.png")

# ============================================================================
# Plot 8: Length Ratio Distribution
# ============================================================================
print("\n8. Creating Length Ratio Distribution...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.hist(results_df['length_ratio'], bins=30, color='#5b9bd5', alpha=0.7, edgecolor='black')

mean_ratio = results_df['length_ratio'].mean()
ax.axvline(mean_ratio, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_ratio:.2f}')
ax.axvline(1.0, color='green', linestyle='--', linewidth=2, 
           label='Perfect Ratio: 1.0')

ax.set_xlabel('Length Ratio (Predicted / Reference)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Length Ratio Distribution', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add statistics
stats_text = f'Mean: {mean_ratio:.2f}\nStd Dev: {results_df["length_ratio"].std():.2f}\nMedian: {results_df["length_ratio"].median():.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / '08_length_ratio_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 08_length_ratio_distribution.png")

# ============================================================================
# Plot 9: Character Error Rate by BLEU Score
# ============================================================================
print("\n9. Creating Character Error Rate vs BLEU Score...")

char_error_rate = (char_errors / results_df['target'].str.len() * 100).values

fig, ax = plt.subplots(figsize=(12, 7))

scatter = ax.scatter(bleu_scores, char_error_rate, 
                     c=results_df['input_length'], cmap='viridis', 
                     s=100, alpha=0.6, edgecolors='black')

ax.set_xlabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_ylabel('Character Error Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Character Error Rate vs BLEU Score', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Input Length (words)', fontsize=12, fontweight='bold')

# Add trend line
z = np.polyfit(bleu_scores, char_error_rate, 1)
p = np.poly1d(z)
x_trend = np.linspace(bleu_scores.min(), bleu_scores.max(), 100)
ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / '09_char_error_vs_bleu.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 09_char_error_vs_bleu.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("✅ Successfully generated 9 individual plots!")
print("=" * 60)
print(f"\nAll plots saved to: {output_dir}/")
print("\nGenerated files:")
print("  1. 01_overall_metrics_radar.png - Radar chart of key metrics")
print("  2. 02_performance_by_length.png - Bar chart by input length")
print("  3. 03_bleu_distribution.png - Histogram of BLEU scores")
print("  4. 04_bleu_violin.png - Violin plot of BLEU distribution")
print("  5. 05_quality_vs_length.png - Scatter: quality vs input length")
print("  6. 06_length_comparison.png - Scatter: predicted vs reference length")
print("  7. 07_metrics_table.png - Performance metrics table")
print("  8. 08_length_ratio_distribution.png - Length ratio histogram")
print("  9. 09_char_error_vs_bleu.png - Character error vs BLEU")

print("\n📊 Summary Statistics:")
print(f"  - Mean BLEU Score: {bleu_scores.mean():.2f}")
print(f"  - Median BLEU Score: {median_bleu:.2f}")
print(f"  - Exact Match Rate: {exact_match_rate:.2f}%")
print(f"  - Character Accuracy: {char_accuracy:.2f}%")
print(f"  - Mean Length Ratio: {mean_ratio:.2f} ± {results_df['length_ratio'].std():.2f}")
print(f"  - Total Samples: {total_samples}")

print("\n✨ All plots are publication-ready at 300 DPI!")
print("=" * 60)
