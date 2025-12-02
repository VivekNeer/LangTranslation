#!/bin/bash
# TensorBoard Dashboard Launcher for English->Tulu Translation Model

echo "========================================"
echo "TensorBoard Dashboard Launcher"
echo "========================================"
echo ""
echo "Starting TensorBoard server..."
echo "This will open an interactive dashboard showing:"
echo "  - Training Loss over time"
echo "  - Learning Rate"
echo "  - Other training metrics"
echo ""
echo "The dashboard will be available at:"
echo "  http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

# Launch TensorBoard pointing to the runs directory
/home/vivek/anaconda3/envs/simple-t5-env/bin/tensorboard --logdir=runs --port=6006 --bind_all

# Alternative: If you want to specify a specific run
# /home/vivek/anaconda3/envs/simple-t5-env/bin/tensorboard --logdir=runs/Nov27_13-25-42_vivek-LOQ-15IRH8 --port=6006
