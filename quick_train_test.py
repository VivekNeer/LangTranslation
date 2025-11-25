import os
os.environ["TRANSFORMERS_USE_LEGACY_IMPORT"] = "True"

import logging
import pandas as pd
from simpletransformers.t5 import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "combined_translations_train.csv")
EVAL_CSV = os.path.join(DATA_DIR, "combined_translations_test.csv")

TASKS = (
    ("translate english to kannada", "English", "Kannada"),
    ("translate kannada to tulu", "Kannada", "Tulu"),
)


def _build_tasks(df: pd.DataFrame) -> pd.DataFrame:
    task_frames = []
    for prefix, source_col, target_col in TASKS:
        task_df = (
            df[[source_col, target_col]]
            .dropna()
            .rename(columns={source_col: "input_text", target_col: "target_text"})
        )
        task_df["prefix"] = prefix
        task_frames.append(
            task_df[["prefix", "input_text", "target_text"]]
            .astype(str)
            .apply(lambda col: col.str.strip())
        )
    return pd.concat(task_frames, ignore_index=True)


train_df = _build_tasks(pd.read_csv(TRAIN_CSV)).head(1000)
eval_df = _build_tasks(pd.read_csv(EVAL_CSV)).head(200)

print(f"Running quick test with {len(train_df)} training samples and {len(eval_df)} evaluation samples.")

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

model = T5Model("mt5", "google/mt5-small", args=model_args)

print("Starting quick test training...")
model.train_model(train_df, eval_data=eval_df)
print("Quick test training complete!")