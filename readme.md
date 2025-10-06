# Simple mT5 Translation Model 🌐

This project demonstrates how to fine-tune a pre-trained multilingual T5 (`mT5-small`) model for translation between two languages. It uses the `simpletransformers` library to handle the training and a `streamlit` web application to provide a user-friendly interface for testing the final model.

The setup is initially configured for English-to-Sinhalese translation using the Tatoeba dataset but is designed to be easily adapted for other language pairs, such as English-to-Kannada or English-to-Tulu.

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Usage Workflow](#-usage-workflow)
- [Adapting for a New Language](#-adapting-for-a-new-language)
- [Troubleshooting](#-troubleshooting)

---

## 📁 Project Structure

Here's an overview of the key files in this project:

- **`prepare_data.py`**: A script to process the raw source (`.src`) and target (`.trg`) text files into the `.tsv` format required for training.
- **`train_model.py`**: The main script for training the model on a large, specified subset of the data.
- **`quick_train_test.py`**: A utility script to run a very short training session on a tiny data sample to verify the environment and code are working correctly.
- **`test_model.py`**: A command-line tool to quickly test the trained model with a single sentence.
- **`app.py`**: A Streamlit web application that provides a UI to interact with the trained translation model.
- **`data/`**: The directory where the raw and processed datasets are stored.

---

## ⚙️ Setup and Installation

This project uses **Conda** for environment management to ensure all dependencies, especially PyTorch with GPU support, are installed correctly.

### Step 1: Clone the Repository
First, get the project files on your local machine.
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### Step 2: Create the Conda Environment
This is the most critical step. These commands create a clean environment with specific, stable versions of the required libraries to ensure reproducibility and avoid dependency issues.

```bash
# 1. Create a blank environment with Python 3.9
conda create -n simple-t5-env python=3.9 -y

# 2. Activate the new, clean environment
conda activate simple-t5-env

# 3. Install the correct PyTorch for your GPU.
# First, check your CUDA version by running `nvidia-smi` in the terminal.
# The command below is for CUDA 12.1. Adjust `pytorch-cuda=12.1` if your version is different.
conda install pytorch==2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install the remaining specific, stable versions of the other libraries
pip install transformers==4.42.3 simpletransformers==0.70.0 sentencepiece pandas tqdm sacrebleu streamlit huggingface-hub
```
Your environment is now ready!

---

## 🚀 Usage Workflow

Follow these steps to train and test your model.

### Step 1: Download and Prepare the Data
The initial setup uses the English-Sinhalese dataset from the Tatoeba challenge.

```bash
# Create the necessary data directories
mkdir -p data/eng-sin
cd data/eng-sin

# Download and extract the data
wget [https://github.com/ThilinaRajapakse/simple_language_translation/releases/download/v0.1/eng-sin.zip](https://github.com/ThilinaRajapakse/simple_language_translation/releases/download/v0.1/eng-sin.zip)
unzip eng-sin.zip
gunzip train.src.gz
gunzip train.trg.gz

# Go back to the main project directory
cd ../..

# Run the data preparation script
python prepare_data.py
```
This will create `data/train.tsv` and `data/eval.tsv`, which are needed for training.

### Step 2: Run a Quick Verification Test (Recommended)
Before starting the long training process, run the quick test script to make sure everything works.

```bash
python quick_train_test.py
```
This will train on 1,000 sentences for about a minute. If it completes without errors, your setup is correct.

### Step 3: Start the Full Training
Run the main training script. This will train the model on 100,000 sentences.

```bash
python train_model.py
```
**Note:** This process will take **several hours** depending on your GPU. It's best to run it overnight or when your computer is not in use. The final model will be saved in `outputs/mt5-sinhalese-english-100k`.

### Step 4: Test Your Trained Model
You have two ways to test your new model:

**A. Command-Line (for quick checks):**
```bash
python test_model.py "This is a test sentence."
```

**B. Web Application (for interactive use):**
First, update the `model_path` variable inside `app.py` to point to your new model directory (`outputs/mt5-sinhalese-english-100k`). Then, run:
```bash
streamlit run app.py
```
---

## 🌍 Adapting for a New Language

To train a model for a different language pair (e.g., English-to-Kannada):

1.  **Prepare Your Dataset:** Create two plain text files, `train.src` (source language, e.g., English) and `train.trg` (target language, e.g., Kannada), where each line is a corresponding translation. Place them in a new directory, e.g., `data/eng-kan/`.
2.  **Update `prepare_data.py`:** Change the file paths and the prefix strings (e.g., `"translate english to kannada"`) to match your new language pair.
3.  **Run the Workflow:** Execute the same workflow as above (`prepare_data.py`, `train_model.py`, etc.).

---

## ⚠️ Troubleshooting

- **`Killed` Message during training:** This means your computer ran out of system RAM, not GPU memory. It usually happens during data preparation if the dataset is too large. The solution is to use a smaller subset of the data by adjusting the `.head()` value in `train_model.py`.
- **`KeyError: 'prefix'`:** Ensure you do not delete the `prefix` column from the DataFrame in your training scripts. The `simpletransformers` library requires this column to be present.
