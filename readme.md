# Multilingual mT5 Translation Model 🌐

This project fine-tunes `mT5-small` with `simpletransformers` to power two chained translation directions from a single dataset:

- **English → Kannada**
- **Kannada → Tulu**

The latest checkpoints (and the Streamlit / Flask apps) now expect CSV datasets with three columns named `English`, `Kannada`, and `Tulu`. A fine-tuned checkpoint is published at [**VivekNeer/mt5-english-kannada-tulu**](https://huggingface.co/VivekNeer/mt5-english-kannada-tulu).

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Usage Workflow](#-usage-workflow)
- [Adapting for a New Language](#-adapting-for-a-new-language)
- [Troubleshooting](#-troubleshooting)

---

## 📁 Project Structure

#️⃣ Here's an overview of the key files in this project:

- **`train_model.py`**: Builds bilingual training samples directly from the combined CSV dataset and launches full training.
- **`quick_train_test.py`**: A utility script to run a short training session to verify the environment.
- **`test_model.py`**: A command-line tool to quickly test a trained model with a single sentence.
- **`app.py`**: A Streamlit web application to interact with a trained translation model.
- **`upload_model.py`**: A script to upload a locally trained model to the Hugging Face Hub.
- **`data/`**: Place `combined_translations_train.csv` and `combined_translations_test.csv` here (not tracked in git).

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

You have two options: use the published Hugging Face model directly or train your own from scratch with the CSV datasets.

### Option A: Use the Pre-trained Model 

This is the fastest way to get the translator running.

1.  **Open `app.py` or `test_model.py`** in a text editor.
2.  **Modify one line** to point to the model on the Hugging Face Hub.

    **Change this line:**
    ```python
    model_path = "outputs/mt5-english-kannada-tulu"
    ```
    **To this:**
    ```python
    model_path = "VivekNeer/mt5-english-kannada-tulu"
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

1.  **Place the CSV datasets:** copy your three-column training and test files into the `data/` directory and name them `combined_translations_train.csv` and `combined_translations_test.csv` (see the provided samples for the expected headers).

2.  **Run a Quick Verification Test (Optional):**
    ```bash
    python quick_train_test.py
    ```
    This verifies your setup is correct before the long training run.

3.  **Start the Full Training:** This will train the model on up to 100,000 English→Kannada / Kannada→Tulu sentence pairs (from the combined CSV) and save checkpoints to `outputs/mt5-english-kannada-tulu`.
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

1.  **Prepare Your Dataset:** Assemble a CSV with the columns you need. For chained translations, ensure each column has aligned sentences.
2.  **Update the `TASKS` tuples** inside `train_model.py`/`quick_train_test.py`/`evaluate_model.py` to reflect the prefixes and column names you want to train.
3.  **Run the Workflow:** Execute `train_model.py`, evaluate, and update the apps just like the default English↔Kannada↔Tulu pipeline.

---

## ⚠️ Troubleshooting

- **`Killed` Message:** This means your computer ran out of system RAM. The solution is to use a smaller subset of the data by adjusting the `.head()` value in `train_model.py`.
- **Dependency Issues:** If you encounter errors, the most reliable fix is to delete and recreate the Conda environment exactly as described in the setup instructions.