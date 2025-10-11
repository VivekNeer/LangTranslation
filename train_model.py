import os
os.environ["TRANSFORMERS_USE_LEGACY_IMPORT"] = "True"

import logging
import pandas as pd
from simpletransformers.t5 import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Load the full datasets (this part doesn't need to change)
print("Loading full dataset...")
full_train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
full_eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)
print("Dataset loaded.")

# --- USE A LARGE, MANAGEABLE SUBSET ---
train_df = full_train_df.head(100000) # Using 100k examples = 50k sentence pairs
eval_df = full_eval_df.head(5000)

print(f"Using a subset of {len(train_df)} for training and {len(eval_df)} for validation.")

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 64,
    "train_batch_size": 4,
    "eval_batch_size": 4,
    "num_train_epochs": 3,
    "save_eval_checkpoints": True,
    "save_steps": 5000,
    "use_multiprocessing": False,
    "fp16": False,
    # --- CHANGED: Give the new model a distinct name ---
    "output_dir": "outputs/mt5-kannada-english-100k",
}

# The base model is still google/mt5-small as it's multilingual
model = T5Model("mt5", "google/mt5-small", args=model_args)

print("Starting model training...")
model.train_model(train_df, eval_data=eval_df)
print("Training complete.")