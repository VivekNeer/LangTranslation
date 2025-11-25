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
OUTPUT_DIR = "outputs/mt5-english-kannada-tulu"

TASKS = (
    ("translate english to kannada", "English", "Kannada"),
    ("translate kannada to tulu", "Kannada", "Tulu"),
)


def _build_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with prefix/input/target rows for each configured task."""

    task_frames = []
    for prefix, source_col, target_col in TASKS:
        if source_col not in df.columns or target_col not in df.columns:
            raise ValueError(
                f"Missing required column(s) '{source_col}' or '{target_col}' in dataset."
            )

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


print("Loading three-column train/test CSV files...")
train_raw = pd.read_csv(TRAIN_CSV)
eval_raw = pd.read_csv(EVAL_CSV)

train_df = _build_tasks(train_raw).head(100000)
eval_df = _build_tasks(eval_raw).head(5000)

print(f"Prepared {len(train_df)} training rows and {len(eval_df)} validation rows.")

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
    "output_dir": OUTPUT_DIR,
}

model = T5Model("mt5", "google/mt5-small", args=model_args)

print("Starting model training...")
model.train_model(train_df, eval_data=eval_df)
print("Training complete.")