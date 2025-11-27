# Graphs and Visualizations

This directory contains all plotting scripts and generated visualizations for the English-to-Tulu translation model training.

## 📊 Generated Plots

### 1. **Comprehensive Dashboard** ⭐ NEW
- `comprehensive_dashboard.png`
- **Source**: Complete evaluation results
- **Shows**: All-in-one view with radar chart, length analysis, BLEU distribution, and metrics table
- **Best for**: Quick overview of all metrics

### 2. **Multi-Metric Comparison** ⭐ NEW
- `multi_metric_comparison.png`
- **Source**: Comprehensive evaluation results
- **Shows**: Radar chart of 4 key metrics + performance by length + detailed table
- **Best for**: Comparing multiple performance dimensions

### 3. **Length Analysis** ⭐ NEW
- `length_analysis.png`
- **Source**: Sample translations analysis
- **Shows**: Input length vs BLEU + Predicted vs Reference length scatter plots
- **Best for**: Understanding how sentence length affects quality

### 4. **Translation Showcases** ⭐ NEW
- `translation_showcase_best.png` - Top 6 best translations
- `translation_showcase_worst.png` - Bottom 6 worst translations
- **Source**: Sample translations from evaluation
- **Shows**: Actual translation examples with quality scores
- **Best for**: Qualitative analysis and error identification

### 5. **BLEU Distribution** ⭐ NEW
- `bleu_distribution.png`
- **Source**: All sample BLEU scores
- **Shows**: Histogram and violin plot of BLEU score distribution
- **Best for**: Understanding score variance and distribution

### 6. **TensorBoard CSV Export Plot**
- `Nov27_02-23-07_vivek-LOQ-15IRH8_plot.png` - Standard resolution (300 DPI)
- `Nov27_02-23-07_vivek-LOQ-15IRH8_plot_highres.png` - High resolution (600 DPI)
- **Source**: TensorBoard CSV export from training run
- **Shows**: Training loss over steps with smoothing, annotations, and statistics

### 7. **Training Loss Plot**
- `training_loss_plot.png`
- **Source**: TensorBoard event files from `runs/` directory
- **Shows**: Training loss progression over epochs/steps

### 8. **BLEU Score Comparison** (Legacy)
- `bleu_score_comparison.png`
- **Source**: Evaluation results from `bleu_scores.txt`
- **Shows**: Simple bar chart of BLEU score
- **Note**: Superseded by multi_metric_comparison.png

## 🐍 Plotting Scripts

### 1. `plot_tensorboard_csv.py`
**Purpose**: Generate professional plots from TensorBoard CSV exports

**Usage**:
```bash
cd graphs
python plot_tensorboard_csv.py
```

**Features**:
- Automatically processes all CSV files in the directory
- Creates both standard (300 DPI) and high-res (600 DPI) versions
- Adds smoothed trend line
- Annotates start, end, and minimum loss values
- Includes statistics box with improvement percentage

**Output**:
- `{csv_name}_plot.png` (300 DPI)
- `{csv_name}_plot_highres.png` (600 DPI)

### 2. `plot_training_loss.py`
**Purpose**: Extract and visualize training loss from TensorBoard logs

**Usage**:
```bash
cd graphs
python plot_training_loss.py
```

**Features**:
- Reads from TensorBoard event files in `../runs/`
- Fallback to text log file parsing
- Annotates each data point with loss value
- Calculates and displays improvement statistics
- Creates epoch-by-epoch or step-by-step visualization

**Output**:
- `training_loss_plot.png`

### 3. `plot_bleu_score.py`
**Purpose**: Visualize BLEU score evaluation results

**Usage**:
```bash
cd graphs
python plot_bleu_score.py
```

**Features**:
- Reads from `../bleu_scores.txt`
- Creates bar chart comparison
- Supports benchmark comparison (if available)
- Annotates bars with exact scores

**Output**:
- `bleu_score_comparison.png`

## 📁 Directory Structure

```
graphs/
├── README.md                                      # This file
├── Nov27_02-23-07_vivek-LOQ-15IRH8.csv           # TensorBoard CSV export
├── Nov27_02-23-07_vivek-LOQ-15IRH8.json          # TensorBoard JSON export
├── Nov27_02-23-07_vivek-LOQ-15IRH8_plot.png      # Generated plot (300 DPI)
├── Nov27_02-23-07_vivek-LOQ-15IRH8_plot_highres.png  # Generated plot (600 DPI)
├── training_loss_plot.png                        # Training loss visualization
├── bleu_score_comparison.png                     # BLEU score comparison
├── plot_tensorboard_csv.py                       # TensorBoard CSV plotter
├── plot_training_loss.py                         # Training loss plotter
└── plot_bleu_score.py                            # BLEU score plotter
```

## 🎯 Quick Commands

### Generate All Plots
```bash
cd /home/vivek/LangTranslation/graphs

# Generate TensorBoard CSV plots
python plot_tensorboard_csv.py

# Generate training loss plot
python plot_training_loss.py

# Generate BLEU score plot
python plot_bleu_score.py
```

### Regenerate After New Training
```bash
cd /home/vivek/LangTranslation/graphs

# If you have new TensorBoard CSV exports, just run:
python plot_tensorboard_csv.py

# To update training loss from latest run:
python plot_training_loss.py

# After running evaluate_model.py, update BLEU scores:
python plot_bleu_score.py
```

## 📈 Understanding the Plots

### Training Loss Plot
- **X-axis**: Training steps or epochs
- **Y-axis**: Loss value (lower is better)
- **Trend**: Should decrease over time
- **Target**: Convergence to a stable low value

**Good Signs**:
- Steady decrease
- Smooth convergence
- Final loss significantly lower than initial

**Warning Signs**:
- Sudden spikes (learning rate too high)
- Plateau early (learning stopped)
- Oscillation (unstable training)

### BLEU Score Plot
- **X-axis**: Translation direction
- **Y-axis**: BLEU score (0-100, higher is better)
- **Interpretation**:
  - < 10: Poor translation
  - 10-20: Understandable but flawed
  - 20-40: Good translation
  - 40-60: Very good translation
  - > 60: Excellent translation

### TensorBoard CSV Plot
- **Complete training overview** with:
  - Raw loss curve (blue line)
  - Smoothed trend (purple line)
  - Key milestones annotated
  - Statistics box with metrics

## 🔧 Customization

### Change Plot Style
Edit the seaborn style in any script:
```python
sns.set_style("whitegrid")  # Options: whitegrid, darkgrid, white, dark, ticks
```

### Change Figure Size
Modify the figure size:
```python
plt.rcParams['figure.figsize'] = (14, 8)  # (width, height) in inches
```

### Change DPI
Adjust resolution:
```python
plt.savefig(output_path, dpi=300)  # Standard: 300, High-res: 600, Screen: 100
```

### Change Colors
Customize colors:
```python
color='#2E86AB'  # Blue
color='#A23B72'  # Purple
color='#F18F01'  # Orange
```

## 📤 Exporting for Papers/Presentations

All plots are saved at 300 DPI by default, suitable for publications.

For extra high-quality:
- Use `_highres.png` versions (600 DPI)
- Or modify scripts to save as SVG:
  ```python
  plt.savefig('plot.svg', format='svg')
  ```

## 🆘 Troubleshooting

### No CSV files found
```bash
# Export from TensorBoard:
# 1. Open TensorBoard (http://localhost:6006)
# 2. Click on Scalars tab
# 3. Click download icon (↓) on the loss graph
# 4. Save CSV to this directory
```

### Module not found errors
```bash
# Install required packages:
pip install matplotlib seaborn pandas tensorboard
```

### Plots look different
```bash
# Clear matplotlib cache:
rm -rf ~/.cache/matplotlib
```

### Path errors
All scripts now use relative paths from the `graphs/` directory:
- `../runs/` for TensorBoard logs
- `../bleu_scores.txt` for BLEU scores
- `../outputs/` for model outputs

## 📝 Notes

- All scripts are designed to run from within the `graphs/` directory
- Original plots moved from project root to this directory
- Scripts updated to use relative paths (`../` prefix)
- TensorBoard CSV exports should be placed in this directory

---

**Last Updated**: November 27, 2025
