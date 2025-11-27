# Visualization Organization Summary

## ✅ Reorganization Complete!

All plotting scripts and generated visualizations have been organized into the `graphs/` directory.

## 📁 What Was Moved

### Scripts Moved to `graphs/`:
1. ✅ `plot_bleu_score.py` - BLEU score visualization
2. ✅ `plot_training_loss.py` - Training loss visualization
3. ✅ `plot_tensorboard_csv.py` - NEW: TensorBoard CSV plotter

### Images Moved to `graphs/`:
1. ✅ `bleu_score_comparison.png` - BLEU score bar chart
2. ✅ `training_loss_plot.png` - Training loss over time

### New Plots Generated in `graphs/`:
1. ✅ `Nov27_02-23-07_vivek-LOQ-15IRH8_plot.png` - Standard resolution (300 DPI)
2. ✅ `Nov27_02-23-07_vivek-LOQ-15IRH8_plot_highres.png` - High resolution (600 DPI)

## 🔄 Path Updates

All scripts have been updated with correct relative paths:

**Before** (from project root):
```python
RUNS_DIR = "runs"
with open("bleu_scores.txt", "r") as f:
plt.savefig("training_loss_plot.png")
```

**After** (from graphs/ directory):
```python
RUNS_DIR = "../runs"
with open("../bleu_scores.txt", "r") as f:
plt.savefig("training_loss_plot.png")  # Saves in graphs/
```

## 📊 Current Directory Structure

```
LangTranslation/
├── graphs/                                        # 🆕 Organized visualization folder
│   ├── README.md                                 # Complete documentation
│   ├── plot_tensorboard_csv.py                   # NEW: TensorBoard CSV plotter
│   ├── plot_training_loss.py                     # Training loss plotter (moved)
│   ├── plot_bleu_score.py                        # BLEU score plotter (moved)
│   ├── Nov27_02-23-07_vivek-LOQ-15IRH8.csv      # TensorBoard export
│   ├── Nov27_02-23-07_vivek-LOQ-15IRH8.json     # TensorBoard JSON
│   ├── Nov27_02-23-07_vivek-LOQ-15IRH8_plot.png # Generated plot ⭐
│   ├── Nov27_02-23-07_vivek-LOQ-15IRH8_plot_highres.png # High-res ⭐
│   ├── training_loss_plot.png                    # Training loss (moved)
│   └── bleu_score_comparison.png                 # BLEU scores (moved)
├── runs/                                          # TensorBoard logs
├── outputs/                                       # Model checkpoints
├── data/                                          # Training data
├── train_model.py                                 # Main training script
├── evaluate_model.py                              # Evaluation script
├── test_model.py                                  # Testing script
└── ... (other project files)
```

## 🎨 New TensorBoard CSV Plot Features

The newly created plot from your TensorBoard CSV export includes:

- **Raw loss curve** (blue line) - Shows actual training loss
- **Smoothed trend line** (purple line) - Easier to see overall trend
- **Key annotations**:
  - Start loss: 27.20 (yellow box)
  - Minimum loss: 1.39 (green box)
  - End loss: 1.83 (blue box)
- **Statistics box** showing:
  - Total steps: 8,300
  - Initial loss: 27.20
  - Final loss: 1.83
  - Minimum loss: 1.39
  - Improvement: 93.3%

## 🚀 Usage

### Run Scripts from Graphs Directory:
```bash
cd /home/vivek/LangTranslation/graphs

# Generate plot from TensorBoard CSV
python plot_tensorboard_csv.py

# Generate training loss plot
python plot_training_loss.py

# Generate BLEU score plot
python plot_bleu_score.py
```

### Or Run from Project Root:
```bash
cd /home/vivek/LangTranslation

# Run individual scripts
python graphs/plot_tensorboard_csv.py
python graphs/plot_training_loss.py
python graphs/plot_bleu_score.py
```

## 📈 Available Visualizations

| Plot | Description | Resolution | Source |
|------|-------------|------------|--------|
| `Nov27_02-23-07_vivek-LOQ-15IRH8_plot.png` | TensorBoard CSV visualization | 300 DPI | CSV export |
| `Nov27_02-23-07_vivek-LOQ-15IRH8_plot_highres.png` | High-res version | 600 DPI | CSV export |
| `training_loss_plot.png` | Training loss over epochs | 300 DPI | TensorBoard logs |
| `bleu_score_comparison.png` | BLEU score bar chart | 300 DPI | Evaluation results |

## 📝 Documentation

Complete documentation is available in:
- **`graphs/README.md`** - Detailed guide for all plotting scripts and visualizations
- **`TENSORBOARD_GUIDE.md`** - TensorBoard dashboard usage guide
- **`TRAINING_SUMMARY.md`** - Complete training documentation

## 🎯 Next Steps

1. **View the new plot**: Check `graphs/Nov27_02-23-07_vivek-LOQ-15IRH8_plot.png`
2. **Use for presentations**: High-res version available at 600 DPI
3. **Export more data**: Download additional TensorBoard data as CSV and place in `graphs/`
4. **Run plotting scripts**: All scripts work from the `graphs/` directory

---

**Reorganization completed**: November 27, 2025
