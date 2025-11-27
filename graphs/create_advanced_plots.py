#!/usr/bin/env python3
"""
Create diverse and comprehensive visualizations for translation model evaluation
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import textwrap
import matplotlib.font_manager as fm

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# Try to find and use Noto Sans Kannada or fallback fonts
def get_kannada_font():
    """Find a suitable font for Kannada script"""
    # Try common Kannada fonts
    font_candidates = [
        'Noto Sans Kannada',
        'Noto Serif Kannada', 
        'Lohit Kannada',
        'Tunga',
        'Kedage',
        'FreeSerif',
        'DejaVu Sans',
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in font_candidates:
        if font in available_fonts:
            return font
    
    return None  # Use default if no Kannada font found

KANNADA_FONT = get_kannada_font()
if KANNADA_FONT:
    print(f"✓ Using font: {KANNADA_FONT}")
else:
    print("⚠ No Kannada font found - will use transliteration fallback")

# Load comprehensive evaluation results
with open('../comprehensive_evaluation_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

overall = results['overall_metrics']
length_analysis = results['length_analysis']
samples = results['sample_translations']

# ============================================================================
# PLOT 1: Enhanced Multi-Metric Comparison (Better than single bar chart)
# ============================================================================

def create_multi_metric_comparison():
    """Create a radar chart + bar chart combination showing all metrics"""
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Radar Chart for Overall Performance ---
    ax_radar = fig.add_subplot(gs[:, 0], projection='polar')
    
    # Normalize metrics to 0-100 scale for visualization
    categories = ['BLEU\nScore', 'Exact\nMatch', 'Length\nAccuracy', 'Character\nAccuracy']
    values = [
        overall['bleu_score'] * 2,  # Scale BLEU (0-50) to (0-100)
        overall['exact_match_accuracy'],
        (1 - abs(1 - overall['length_ratio_mean'])) * 100,  # Length accuracy
        100 - overall['character_error_rate']  # Character accuracy
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    ax_radar.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Model Performance')
    ax_radar.fill(angles, values, alpha=0.25, color='#2E86AB')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, size=10)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_yticks([20, 40, 60, 80, 100])
    ax_radar.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
    ax_radar.set_title('Overall Performance Metrics\n(0-100 Scale)', 
                       fontsize=14, fontweight='bold', pad=20)
    ax_radar.grid(True, alpha=0.3)
    
    # Add legend with actual values
    legend_text = '\n'.join([
        f'BLEU: {overall["bleu_score"]:.1f}',
        f'Exact Match: {overall["exact_match_accuracy"]:.1f}%',
        f'Length Ratio: {overall["length_ratio_mean"]:.2f}',
        f'Char Accuracy: {100 - overall["character_error_rate"]:.1f}%'
    ])
    ax_radar.text(0.5, -0.15, legend_text, transform=ax_radar.transAxes,
                  fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # --- Bar Chart for Performance by Length ---
    ax_length = fig.add_subplot(gs[0, 1:])
    
    buckets = list(length_analysis.keys())
    bleu_scores = [length_analysis[b]['bleu'] for b in buckets]
    exact_matches = [length_analysis[b]['exact_match'] for b in buckets]
    counts = [length_analysis[b]['count'] for b in buckets]
    
    x = np.arange(len(buckets))
    width = 0.35
    
    bars1 = ax_length.bar(x - width/2, bleu_scores, width, label='BLEU Score', 
                          color='#2E86AB', alpha=0.8)
    bars2 = ax_length.bar(x + width/2, exact_matches, width, label='Exact Match %',
                          color='#A23B72', alpha=0.8)
    
    ax_length.set_xlabel('Input Length Category', fontsize=12, fontweight='bold')
    ax_length.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax_length.set_title('Performance by Input Length\n(with sample counts)', 
                        fontsize=14, fontweight='bold')
    ax_length.set_xticks(x)
    ax_length.set_xticklabels([f'{b}\n(n={counts[i]})' for i, b in enumerate(buckets)])
    ax_length.legend(loc='upper right', fontsize=10)
    ax_length.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_length.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}',
                          ha='center', va='bottom', fontsize=9)
    
    # --- Metrics Summary Table ---
    ax_table = fig.add_subplot(gs[1, 1:])
    ax_table.axis('off')
    
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['BLEU Score', f"{overall['bleu_score']:.2f}", 'Understandable but flawed (10-20 is good)'],
        ['Exact Match', f"{overall['exact_match_accuracy']:.2f}%", 'Perfect translations'],
        ['Char Error Rate', f"{overall['character_error_rate']:.2f}%", 'Character-level accuracy'],
        ['Length Ratio', f"{overall['length_ratio_mean']:.2f} ± {overall['length_ratio_std']:.2f}", 
         'Predicted/Reference length'],
        ['Total Samples', str(overall['total_samples']), 'Validation set size']
    ]
    
    table = ax_table.table(cellText=table_data, cellLoc='left', loc='center',
                          colWidths=[0.25, 0.2, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
    
    plt.suptitle('English-to-Tulu Translation Model: Comprehensive Evaluation', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('multi_metric_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: multi_metric_comparison.png")
    plt.close()

# ============================================================================
# PLOT 2: Length Analysis Scatter Plot
# ============================================================================

def create_length_analysis():
    """Scatter plot showing relationship between length and quality"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    input_lengths = [len(s['input'].split()) for s in samples]
    pred_lengths = [len(s['prediction']) for s in samples]
    ref_lengths = [len(s['reference']) for s in samples]
    bleu_scores = [s['bleu'] for s in samples]
    
    # Plot 1: Input length vs BLEU score
    scatter1 = axes[0].scatter(input_lengths, bleu_scores, 
                              c=bleu_scores, cmap='RdYlGn', 
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('Input Length (words)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Translation Quality vs Input Length', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(input_lengths, bleu_scores, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(input_lengths), max(input_lengths), 100)
    axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[0].legend()
    
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('BLEU Score', rotation=270, labelpad=20)
    
    # Plot 2: Predicted vs Reference length
    scatter2 = axes[1].scatter(ref_lengths, pred_lengths,
                              c=bleu_scores, cmap='RdYlGn',
                              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect length match)
    max_len = max(max(ref_lengths), max(pred_lengths))
    axes[1].plot([0, max_len], [0, max_len], 'k--', alpha=0.5, linewidth=2, label='Perfect Match')
    
    axes[1].set_xlabel('Reference Length (characters)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted Length (characters)', fontsize=12, fontweight='bold')
    axes[1].set_title('Predicted vs Reference Length', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('BLEU Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('length_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: length_analysis.png")
    plt.close()

# ============================================================================
# PLOT 3: Sample Translations Showcase
# ============================================================================

def create_translation_showcase():
    """Visual grid showing best and worst translations"""
    # Instead of trying to render Kannada in matplotlib, create HTML output
    best_samples = sorted(samples, key=lambda x: x['bleu'], reverse=True)[:6]
    worst_samples = sorted(samples, key=lambda x: x['bleu'])[:6]
    
    # Create HTML versions for proper Kannada rendering
    create_html_showcase(best_samples, 'translation_showcase_best.html', 'Best Translations', 'success')
    create_html_showcase(worst_samples, 'translation_showcase_worst.html', 'Worst Translations', 'danger')
    print("✅ Saved: translation_showcase_best.html (open in browser for proper Kannada rendering)")
    print("✅ Saved: translation_showcase_worst.html (open in browser for proper Kannada rendering)")
    
    # Create simplified PNG versions with sample numbers only
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    def create_translation_box_simplified(ax, sample, sample_num, title_color, is_best):
        ax.axis('off')
        
        # Wrap English text
        input_text = textwrap.fill(sample['input'], width=60)
        
        # Create text box with English only + reference to HTML
        bleu = sample['bleu']
        exact = '✓' if sample['exact_match'] else '✗'
        
        quality = "EXCELLENT" if is_best else "POOR"
        
        text = f"Sample #{sample_num} - {quality}\n"
        text += "=" * 65 + "\n\n"
        text += f"English Input:\n{input_text}\n\n"
        text += f"Tulu Translation: [See HTML file for proper rendering]\n\n"
        text += f"BLEU Score: {bleu:.1f} | Exact Match: {exact}\n\n"
        text += f"Note: Open translation_showcase_{'best' if is_best else 'worst'}.html\n"
        text += f"in a browser to see Kannada script properly rendered."
        
        font_props = {'family': 'monospace', 'size': 8}
        if KANNADA_FONT:
            font_props['family'] = KANNADA_FONT
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=8, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=1', facecolor=title_color, alpha=0.3),
               family='monospace', wrap=True)
    
    # Plot best translations
    for i, sample in enumerate(best_samples):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        create_translation_box_simplified(ax, sample, i+1, 'lightgreen', True)
        if i == 0:
            ax.set_title('🏆 Best Translations (High BLEU)\nOpen .html file for proper Kannada rendering', 
                        fontsize=12, fontweight='bold', pad=10, color='green')
    
    plt.suptitle('Translation Quality Showcase: Best Examples\n(For proper Kannada rendering, open translation_showcase_best.html)', 
                 fontsize=14, fontweight='bold')
    plt.savefig('translation_showcase_best.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: translation_showcase_best.png (simplified - English only)")
    plt.close()
    
    # Plot worst translations
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for i, sample in enumerate(worst_samples):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        create_translation_box_simplified(ax, sample, i+1, 'lightcoral', False)
        if i == 0:
            ax.set_title('⚠️ Worst Translations (Low BLEU)\nOpen .html file for proper Kannada rendering', 
                        fontsize=12, fontweight='bold', pad=10, color='red')
    
    plt.suptitle('Translation Quality Showcase: Areas for Improvement\n(For proper Kannada rendering, open translation_showcase_worst.html)', 
                 fontsize=14, fontweight='bold')
    plt.savefig('translation_showcase_worst.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: translation_showcase_worst.png (simplified - English only)")
    plt.close()

def create_html_showcase(samples, filename, title, badge_type):
    """Create HTML file with proper Kannada rendering"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - English to Tulu Translation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }}
        .sample {{
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            background: #f9f9f9;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .sample:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        .sample.best {{
            border-color: #4caf50;
            background: #f1f8f4;
        }}
        .sample.worst {{
            border-color: #f44336;
            background: #fff1f0;
        }}
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ddd;
        }}
        .sample-number {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        .metrics {{
            display: flex;
            gap: 15px;
        }}
        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge.success {{
            background: #4caf50;
            color: white;
        }}
        .badge.danger {{
            background: #f44336;
            color: white;
        }}
        .badge.info {{
            background: #2196F3;
            color: white;
        }}
        .section {{
            margin: 15px 0;
        }}
        .label {{
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        .content {{
            padding: 12px;
            background: white;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
            margin-top: 5px;
        }}
        .content.english {{
            border-left-color: #2196F3;
        }}
        .content.tulu {{
            border-left-color: #9c27b0;
            font-size: 1.2em;
            line-height: 1.8;
            font-family: 'Noto Sans Kannada', 'Tunga', 'Lohit Kannada', serif;
        }}
        .content.reference {{
            border-left-color: #ff9800;
        }}
        .exact-match {{
            display: inline-block;
            margin-left: 10px;
            font-size: 1.5em;
        }}
        .stats {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .stats h3 {{
            margin: 0 0 10px 0;
            color: #1976d2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{'🏆' if badge_type == 'success' else '⚠️'} {title}</h1>
        <p class="subtitle">English to Tulu Translation Model Evaluation</p>
        
        <div class="stats">
            <h3>Showing {len(samples)} {'Best' if badge_type == 'success' else 'Worst'} Translation Examples</h3>
        </div>
        
        <div class="grid">
"""
    
    for i, sample in enumerate(samples, 1):
        bleu = sample['bleu']
        exact_match = sample['exact_match']
        exact_icon = '✓' if exact_match else '✗'
        
        html += f"""
            <div class="sample {'best' if badge_type == 'success' else 'worst'}">
                <div class="sample-header">
                    <div class="sample-number">Sample #{i}</div>
                    <div class="metrics">
                        <span class="badge {badge_type}">BLEU: {bleu:.2f}</span>
                        <span class="badge info">Match: {exact_icon}</span>
                    </div>
                </div>
                
                <div class="section">
                    <div class="label">English Input</div>
                    <div class="content english">{sample['input']}</div>
                </div>
                
                <div class="section">
                    <div class="label">Model Prediction (Tulu)</div>
                    <div class="content tulu">{sample['prediction']}</div>
                </div>
                
                <div class="section">
                    <div class="label">Reference Translation (Tulu)</div>
                    <div class="content tulu reference">{sample['reference']}</div>
                </div>
            </div>
"""
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return filename

# ============================================================================
# PLOT 4: BLEU Score Distribution
# ============================================================================

def create_bleu_distribution():
    """Histogram and violin plot of BLEU scores"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    bleu_scores = [s['bleu'] for s in samples]
    
    # Histogram
    axes[0].hist(bleu_scores, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(bleu_scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(bleu_scores):.2f}')
    axes[0].axvline(np.median(bleu_scores), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(bleu_scores):.2f}')
    axes[0].set_xlabel('BLEU Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('BLEU Score Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot with swarm
    axes[1].violinplot([bleu_scores], positions=[0], showmeans=True, showmedians=True)
    axes[1].set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    axes[1].set_title('BLEU Score Violin Plot', fontsize=14, fontweight='bold')
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['All Samples'])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Statistics:\n"
    stats_text += f"Mean: {np.mean(bleu_scores):.2f}\n"
    stats_text += f"Median: {np.median(bleu_scores):.2f}\n"
    stats_text += f"Std Dev: {np.std(bleu_scores):.2f}\n"
    stats_text += f"Min: {np.min(bleu_scores):.2f}\n"
    stats_text += f"Max: {np.max(bleu_scores):.2f}"
    
    axes[1].text(0.5, 0.02, stats_text, transform=axes[1].transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('bleu_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: bleu_distribution.png")
    plt.close()

# ============================================================================
# PLOT 5: Comprehensive Dashboard (All-in-One)
# ============================================================================

def create_comprehensive_dashboard():
    """Single dashboard with all key metrics and visualizations"""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. Radar chart (top-left)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    categories = ['BLEU', 'Exact\nMatch', 'Length', 'Char\nAccuracy']
    values = [
        overall['bleu_score'] * 2,
        overall['exact_match_accuracy'],
        (1 - abs(1 - overall['length_ratio_mean'])) * 100,
        100 - overall['character_error_rate']
    ]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax1.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
    ax1.fill(angles, values, alpha=0.25, color='#2E86AB')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=9)
    ax1.set_ylim(0, 100)
    ax1.set_title('Overall Metrics', fontsize=12, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance by length (top-middle)
    ax2 = fig.add_subplot(gs[0, 1:3])
    buckets = list(length_analysis.keys())
    bleu_scores = [length_analysis[b]['bleu'] for b in buckets]
    x = np.arange(len(buckets))
    bars = ax2.bar(x, bleu_scores, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    ax2.set_xlabel('Input Length', fontsize=10, fontweight='bold')
    ax2.set_ylabel('BLEU Score', fontsize=10, fontweight='bold')
    ax2.set_title('Performance by Input Length', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([b.replace(' ', '\n') for b in buckets], fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. BLEU distribution (top-right)
    ax3 = fig.add_subplot(gs[0, 3])
    bleu_scores_all = [s['bleu'] for s in samples]
    ax3.hist(bleu_scores_all, bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(bleu_scores_all), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('BLEU Score', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax3.set_title('BLEU Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Length correlation (middle-left)
    ax4 = fig.add_subplot(gs[1, 0:2])
    input_lengths = [len(s['input'].split()) for s in samples]
    bleu_sample = [s['bleu'] for s in samples]
    scatter = ax4.scatter(input_lengths, bleu_sample, c=bleu_sample, 
                         cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Input Length (words)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('BLEU Score', fontsize=10, fontweight='bold')
    ax4.set_title('Quality vs Input Length', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='BLEU')
    
    # 5. Pred vs Ref length (middle-right)
    ax5 = fig.add_subplot(gs[1, 2:4])
    ref_lengths = [len(s['reference']) for s in samples]
    pred_lengths = [len(s['prediction']) for s in samples]
    ax5.scatter(ref_lengths, pred_lengths, c=bleu_sample, cmap='RdYlGn',
               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    max_len = max(max(ref_lengths), max(pred_lengths))
    ax5.plot([0, max_len], [0, max_len], 'k--', alpha=0.5, linewidth=2)
    ax5.set_xlabel('Reference Length', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Predicted Length', fontsize=10, fontweight='bold')
    ax5.set_title('Length Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics table (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['BLEU Score', f"{overall['bleu_score']:.2f}", 
         '8-10: Understandable but needs improvement'],
        ['Exact Match Accuracy', f"{overall['exact_match_accuracy']:.2f}%",
         '20% of translations are perfect'],
        ['Best on Short Inputs', f"BLEU {length_analysis['short (0-20)']['bleu']:.2f}",
         'Model performs best on short sentences'],
        ['Character Error Rate', f"{overall['character_error_rate']:.2f}%",
         'Character-level accuracy is good'],
        ['Length Ratio', f"{overall['length_ratio_mean']:.2f} ± {overall['length_ratio_std']:.2f}",
         'Predictions slightly shorter than references'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.2, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
    
    plt.suptitle('English-to-Tulu Translation Model: Comprehensive Evaluation Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: comprehensive_dashboard.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*70)
    print()
    
    create_multi_metric_comparison()
    create_length_analysis()
    create_translation_showcase()
    create_bleu_distribution()
    create_comprehensive_dashboard()
    
    print()
    print("="*70)
    print("✅ All visualizations generated successfully!")
    print("📁 Saved to: graphs/ directory")
    print("="*70)
    print()
    print("Generated plots:")
    print("  1. multi_metric_comparison.png - Enhanced metrics comparison (radar + bars)")
    print("  2. length_analysis.png - Length vs quality scatter plots")
    print("  3. translation_showcase_best.png - Best translation examples")
    print("  4. translation_showcase_worst.png - Worst translation examples")
    print("  5. bleu_distribution.png - BLEU score distribution")
    print("  6. comprehensive_dashboard.png - All-in-one dashboard")
