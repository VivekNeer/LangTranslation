import os
os.environ["TRANSFORMERS_USE_LEGACY_IMPORT"] = "True"

import logging
import pandas as pd
from simpletransformers.t5 import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Load the full datasets
full_train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
full_eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

# Use a small subset
train_df = full_train_df.head(1000)
eval_df = full_eval_df.head(200)

print(f"Running quick test with {len(train_df)} training samples and {len(eval_df)} evaluation samples.")

# train_df = train_df.drop("prefix", axis=1)
# eval_df = eval_df.drop("prefix", axis=1)

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 64,
    "train_batch_size": 4,
    "eval_batch_size": 4,
    "max_steps": 100,
    "output_dir": "outputs/quick-test-model",
    "save_steps": -1,
    "fp16": False,
}

# The simple, direct way to initialize the model.
# The T5Model class correctly handles using "mt5" under the hood.
model = T5Model("mt5", "google/mt5-small", args=model_args)

print("Starting quick test training...")
model.train_model(train_df, eval_data=eval_df)
print("Quick test training complete!")