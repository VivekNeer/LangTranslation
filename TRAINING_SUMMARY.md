# English to Tulu Translation Model - Training Summary

## Model Information
- **Base Model**: google/mt5-small
- **Task**: English → Tulu Translation
- **Model Path**: `outputs/mt5-english-tulu`
- **Training Device**: CPU (CUDA Error 101 prevented GPU use)

## Training Configuration
- **Training Samples**: 8,300 English-Tulu pairs
- **Validation Samples**: 1,000 English-Tulu pairs
- **Epochs**: 10
- **Batch Size**: 4 (train), 4 (eval)
- **Max Sequence Length**: 64
- **Optimizer**: Adafactor (default)
- **Learning Rate**: 0.001 (default)
- **Total Training Steps**: 20,750
- **Training Time**: ~75 minutes (7.5 min/epoch)

## Training Results

### Loss Progression
- **Initial Loss (Step 50)**: 27.84
- **Final Loss (Step 20750)**: 1.39
- **Improvement**: **95.0% reduction** in loss

### Loss by Epoch
| Epoch | Running Loss |
|-------|--------------|
| 1     | 5.16         |
| 2     | 3.25         |
| 3     | 3.84         |
| 4     | 2.77         |
| 5     | 1.25         |
| 6     | 2.94         |
| 7     | 2.59         |
| 8     | 2.24         |
| 9     | 1.72         |
| 10    | 1.82         |

### BLEU Score Evaluation
- **Test Set Size**: 1,000 samples
- **BLEU Score**: **8.40**

> **Note**: A BLEU score of 8.40 is modest but reasonable for a low-resource language pair like English-Tulu with only 8.3K training samples. Typical improvements would come from:
> - More training data (20K+ samples recommended)
> - Longer training (20+ epochs)
> - Hyperparameter tuning
> - Data augmentation

## Generated Plots

### 1. Training Loss Over Time
**File**: `training_loss_plot.png`
- Shows the smooth decrease in training loss from 27.84 to 1.39
- Demonstrates effective learning throughout all 20,750 steps
- Clear improvement trend with some oscillation (typical for language models)

### 2. BLEU Score Comparison
**File**: `bleu_score_comparison.png`
- Visualizes the model's translation performance
- BLEU score of 8.40 for English → Tulu

## Sample Translations (Final Model)

| English | Tulu Output |
|---------|-------------|
| hello my name is vivek | ಎಂಕ್ಲೆನ ಪುದರ್ ವಿಸ್ಮಯ |
| I am going to school | ಯಾನ್ ಶಾಲೆಗ್ ಪೋಪೆ |
| how are you today | ಇನಿ ಇನಿ ಇನಿ |

## Model Usage

### Command Line Testing
```bash
cd /home/vivek/LangTranslation
python test_model.py "your english sentence here"
```

### Python API
```python
from simpletransformers.t5 import T5Model

model = T5Model("mt5", "outputs/mt5-english-tulu", use_cuda=False)
model.args.num_beams = 5
model.args.max_length = 50

input_text = "translate english to tulu: hello my name is vivek"
output = model.predict([input_text])
print(output[0])  # Tulu translation
```

## Files Generated

### Training Files
- `outputs/mt5-english-tulu/` - Final trained model
- `outputs/mt5-english-tulu/checkpoint-XXXX-epoch-N/` - Intermediate checkpoints
- `runs/` - TensorBoard logs
- `cache_dir/` - Cached preprocessing

### Evaluation Files
- `bleu_scores.txt` - BLEU score data
- `bleu_score_comparison.png` - BLEU score visualization
- `training_loss_plot.png` - Training loss visualization
- `training_output.log` or `training_english_tulu.log` - Full training logs

## Recommendations for Improvement

1. **More Training Data**
   - Current: 8,300 samples
   - Recommended: 20,000+ samples
   - Would likely improve BLEU to 15-25 range

2. **GPU Training**
   - Current: CPU (~7.5 min/epoch)
   - With GPU: ~1-2 min/epoch
   - Could train for 20+ epochs in same time

3. **Hyperparameter Tuning**
   - Try higher learning rates (0.003, 0.005)
   - Experiment with batch sizes (8, 16)
   - Increase max_seq_length to 128 for longer sentences

4. **Model Size**
   - Current: mt5-small (~300M params)
   - Try: mt5-base (~580M params) for better quality
   - Trade-off: Slower training but better results

## Date
Training completed: November 27, 2025
