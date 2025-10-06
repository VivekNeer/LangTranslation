# Simple mT5 Translation Model 🌐

This project fine-tunes a pre-trained multilingual T5 (`mT5-small`) model for translation. It uses the `simpletransformers` library for training and a `streamlit` web application to provide an interface for testing the final model.

A pre-trained **English-to-Sinhalese** model from this repository is available on the Hugging Face Hub at [**VivekNeer/mt5-sinhalese-english**](https://huggingface.co/VivekNeer/mt5-sinhalese-english).

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Usage Workflow](#-usage-workflow)
- [Adapting for a New Language](#-adapting-for-a-new-language)
- [Troubleshooting](#-troubleshooting)

---

## 📁 Project Structure

Here's an overview of the key files in this project:

- **`prepare_data.py`**: A script to process raw text files into the `.tsv` format required for training.
- **`train_model.py`**: The main script for training the model on a large subset of the data.
- **`quick_train_test.py`**: A utility script to run a short training session to verify the environment.
- **`test_model.py`**: A command-line tool to quickly test a trained model with a single sentence.
- **`app.py`**: A Streamlit web application to interact with a trained translation model.
- **`upload_model.py`**: A script to upload a locally trained model to the Hugging Face Hub.
- **`data/`**: The directory where datasets are stored.

---

## ⚙️ Setup and Installation

This project uses **Conda** for environment management to ensure all dependencies are installed correctly.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/VivekNeer/simple-t5-translation.git](https://github.com/VivekNeer/simple-t5-translation.git)
cd simple-t5-translation
```

### Step 2: Create the Conda Environment
These commands create a clean environment with specific, stable library versions.

```bash
# 1. Create a blank environment with Python 3.9
conda create -n simple-t5-env python=3.9 -y

# 2. Activate the new environment
conda activate simple-t5-env

# 3. Install the correct PyTorch for your GPU.
# First, check your CUDA version by running `nvidia-smi`.
# The command below is for CUDA 12.1. Adjust if your version is different.
conda install pytorch==2.3.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install the remaining libraries
pip install transformers==4.42.3 simpletransformers==0.70.0 sentencepiece pandas tqdm sacrebleu streamlit huggingface-hub
```

---

## 🚀 Usage Workflow

You have two options: use the pre-trained model directly or train your own from scratch.

### Option A: Use the Pre-trained Model 

This is the fastest way to get the translator running.

1.  **Open `app.py` or `test_model.py`** in a text editor.
2.  **Modify one line** to point to the model on the Hugging Face Hub.

    **Change this line:**
    ```python
    model_path = "outputs/mt5-sinhalese-english-100k"
    ```
    **To this:**
    ```python
    model_path = "VivekNeer/mt5-sinhalese-english"
    ```
3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    The first time you run this, it will download the pre-trained model from the Hub.
4.  **Test Your Local Model Directly(Streamlit currently not working):** After training, you can run the app with the default `model_path` pointing to your local `outputs/` directory.
    ```bash
    python test_model.py "This is a test sentence."
    ```
---

### Option B: Train Your Own Model from Scratch

Follow these steps to train your own version of the model.

1.  **Download and Prepare the Data:** The initial setup uses the English-Sinhalese dataset.
    ```bash
    # Create directories and download data
    mkdir -p data/eng-sin && cd data/eng-sin
    wget [https://github.com/ThilinaRajapakse/simple_language_translation/releases/download/v0.1/eng-sin.zip](https://github.com/ThilinaRajapakse/simple_language_translation/releases/download/v0.1/eng-sin.zip)
    unzip eng-sin.zip && gunzip train.src.gz && gunzip train.trg.gz
    cd ../..
    
    # Run the data preparation script
    python prepare_data.py

    Need not do this as Data Folder already exists in the repo
    ```

2.  **Run a Quick Verification Test (Optional):**
    ```bash
    python quick_train_test.py
    ```
    This verifies your setup is correct before the long training run.

3.  **Start the Full Training:** This will train the model on 100,000 sentences and save it locally to `outputs/mt5-sinhalese-english-100k`.
    ```bash
    python train_model.py
    ```
    **Note:** This process will take **several hours**.

4.  **Test Your Local Model:** After training, you can run the app with the default `model_path` pointing to your local `outputs/` directory.
    ```bash
    streamlit run app.py
    ```

5.  **Test Your Local Model Directly(Streamlit currently not working):** After training, you can run the app with the default `model_path` pointing to your local `outputs/` directory.
    ```bash
    python test_model.py "This is a test sentence."
    ```
---


## 🌍 Adapting for a New Language

To train a model for a different language pair (e.g., English-to-Kannada):

1.  **Prepare Your Dataset:** Create `train.src` and `train.trg` files with your translated sentences and place them in a new directory, e.g., `data/eng-kan/`.
2.  **Update `prepare_data.py`:** Change the file paths and the prefix strings (e.g., `"translate english to kannada"`) to match your new language pair.
3.  **Run the Workflow:** Execute the same training workflow as above.

---

## ⚠️ Troubleshooting

- **`Killed` Message:** This means your computer ran out of system RAM. The solution is to use a smaller subset of the data by adjusting the `.head()` value in `train_model.py`.
- **Dependency Issues:** If you encounter errors, the most reliable fix is to delete and recreate the Conda environment exactly as described in the setup instructions.