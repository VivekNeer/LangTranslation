# TensorBoard Dashboard Guide

## 🚀 Quick Start

TensorBoard is already running! Access it at:

**URL**: http://localhost:6006

Or if accessing from another machine on the same network:
**URL**: http://vivek-LOQ-15IRH8:6006

## 📊 What You'll See

### 1. **Scalars Tab** (Most Important)
Shows training metrics over time:
- **loss**: Training loss (should decrease over time)
- **lr**: Learning rate schedule

### 2. **Graphs Tab**
- Model architecture visualization
- Computational graph

### 3. **Distributions Tab**
- Weight distributions over time

### 4. **Histograms Tab**
- Parameter histograms

## 🎯 How to Use TensorBoard

### View Training Loss
1. Open http://localhost:6006 in your browser
2. Click on the **"Scalars"** tab (should be default)
3. You'll see a graph showing how `loss` decreased over 20,750 steps
4. Hover over the line to see exact values at each step

### Compare Multiple Runs
If you have multiple training runs:
1. Each run appears as a different colored line
2. Use the checkboxes on the left to show/hide runs
3. Compare how different hyperparameters affected training

### Smooth the Curves
- Use the "Smoothing" slider at the top to smooth noisy curves
- Values between 0.6-0.8 work well for most cases

### Change Time Axis
- **STEP**: Shows training steps (default)
- **RELATIVE**: Shows time relative to start
- **WALL**: Shows actual clock time

## 📁 Available Training Runs

Your `runs/` directory contains multiple training sessions:
```
runs/
├── Nov27_01-24-40_vivek-LOQ-15IRH8  (Earlier attempt)
├── Nov27_02-22-29_vivek-LOQ-15IRH8  (Earlier attempt)
├── Nov27_02-23-07_vivek-LOQ-15IRH8  (Earlier attempt)
├── Nov27_12-36-35_vivek-LOQ-15IRH8  (Earlier attempt)
├── Nov27_13-14-50_vivek-LOQ-15IRH8  (Earlier attempt)
└── Nov27_13-25-42_vivek-LOQ-15IRH8  (Final 10-epoch training) ← Main one
```

The latest run (`Nov27_13-25-42`) contains your complete 10-epoch training data.

## 🛠️ Starting/Stopping TensorBoard

### Start TensorBoard
```bash
cd /home/vivek/LangTranslation
/home/vivek/anaconda3/envs/simple-t5-env/bin/tensorboard --logdir=runs --port=6006 --bind_all
```

Or use the launcher scripts:
```bash
# Bash script
./launch_tensorboard.sh

# Python script
python launch_tensorboard.py
```

### Stop TensorBoard
Press `Ctrl+C` in the terminal where TensorBoard is running

### Check if TensorBoard is Running
```bash
lsof -i :6006
```

### Kill TensorBoard Process
```bash
pkill -f tensorboard
```

## 🎨 TensorBoard Features

### 1. **Real-time Updates**
- TensorBoard automatically refreshes as new data is written
- Useful during training to monitor progress live

### 2. **Download Data**
- Click the download icon (↓) on any graph
- Export data as CSV or SVG

### 3. **Customize Views**
- Toggle log scale for y-axis
- Adjust x-axis range
- Show/hide specific metrics

### 4. **Filtering**
- Use the filter box to search for specific metrics
- Supports regex patterns

## 📈 Understanding Your Training Metrics

### Loss Curve Analysis

**Healthy Training** (What you should see):
- Steady decrease from ~28 to ~1.4
- Some oscillation is normal
- Plateau at the end indicates convergence

**Warning Signs** (What to watch for):
- **Sudden spikes**: May indicate learning rate too high
- **No decrease**: Model not learning, check data/hyperparameters
- **Overfitting**: Loss decreases but validation loss increases

### Learning Rate Schedule
Your model uses a constant learning rate with warmup:
- Starts low during warmup phase
- Reaches full learning rate
- Stays constant throughout training

## 💡 Pro Tips

1. **Bookmark the URL**: http://localhost:6006
2. **Keep TensorBoard running**: Start it in a separate terminal
3. **Compare experiments**: Run with different hyperparameters and compare
4. **Export graphs**: Use for presentations/papers
5. **Monitor during training**: Watch loss decrease in real-time

## 🔧 Advanced Usage

### View Specific Run Only
```bash
tensorboard --logdir=runs/Nov27_13-25-42_vivek-LOQ-15IRH8 --port=6006
```

### Change Port
```bash
tensorboard --logdir=runs --port=8888
```

### Multiple Log Directories
```bash
tensorboard --logdir=name1:path1,name2:path2
```

Example:
```bash
tensorboard --logdir=training:runs,validation:eval_runs
```

## 📸 Screenshot Examples

When you open TensorBoard, you should see:

### Scalars Tab
- **loss**: Smooth curve from 27.84 → 1.39
- **lr**: Learning rate schedule

### Expected Metrics
```
Step 50:    Loss = 27.84
Step 1000:  Loss = 5.38
Step 5000:  Loss = 3.06
Step 10000: Loss = 2.35
Step 15000: Loss = 1.53
Step 20750: Loss = 1.39
```

## 🆘 Troubleshooting

### TensorBoard not showing data
1. Verify logs exist: `ls -la runs/`
2. Restart TensorBoard
3. Clear browser cache (Ctrl+F5)

### Port already in use
```bash
# Find and kill the process
lsof -i :6006
kill -9 <PID>

# Or use a different port
tensorboard --logdir=runs --port=6007
```

### Can't access from browser
- Check firewall settings
- Try http://127.0.0.1:6006 instead of localhost
- Ensure TensorBoard is running (check terminal output)

## 📝 Next Steps

1. **Open TensorBoard now**: http://localhost:6006
2. **Explore the Scalars tab**: See your training loss curve
3. **Download graphs**: For documentation/presentations
4. **Compare runs**: If you retrain with different settings

---

**Current Status**: ✅ TensorBoard is running on port 6006

**Access**: http://localhost:6006
