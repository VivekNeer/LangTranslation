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
EVAL_CSV = os.path.join(DATA_DIR, "combined_translations_validation.csv")
OUTPUT_DIR = "outputs/mt5-english-tulu"

# Single direct English -> Tulu task
TASKS = (
    ("translate english to tulu", "English", "Tulu"),
)


def _find_col_case_insensitive(df: pd.DataFrame, name: str):
    name_l = name.strip().lower()
    for c in df.columns:
        if isinstance(c, str) and c.strip().lower() == name_l:
            return c
    return None


def _build_tasks(df: pd.DataFrame) -> pd.DataFrame:
    task_frames = []
    for prefix, source_col, target_col in TASKS:
        src = _find_col_case_insensitive(df, source_col)
        tgt = _find_col_case_insensitive(df, target_col)

        if src is None or tgt is None:
            logging.warning(
                "Skipping task '%s': missing column(s) '%s' or '%s' in dataset.",
                prefix,
                source_col,
                target_col,
            )
            continue

        task_df = (
            df[[src, tgt]]
            .dropna()
            .rename(columns={src: "input_text", tgt: "target_text"})
        )
        task_df["prefix"] = prefix
        task_frames.append(
            task_df[["prefix", "input_text", "target_text"]]
            .astype(str)
            .apply(lambda col: col.str.strip())
        )

    if not task_frames:
        return pd.DataFrame(columns=["prefix", "input_text", "target_text"])

    return pd.concat(task_frames, ignore_index=True)


def _safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.error("Failed to read CSV %s: %s", path, e)
        raise


print("Loading train/test CSV files...")
train_raw = _safe_read_csv(TRAIN_CSV)
eval_raw = _safe_read_csv(EVAL_CSV)

train_df = _build_tasks(train_raw).head(100000)
eval_df = _build_tasks(eval_raw).head(5000)

print(f"Prepared {len(train_df)} training rows and {len(eval_df)} validation rows.")

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 64,
    "train_batch_size": 4,
    "eval_batch_size": 4,
    "num_train_epochs": 10,
    "save_eval_checkpoints": True,
    "save_steps": 5000,
    "use_multiprocessing": False,
    "fp16": False,
    "use_cuda": False,  # CUDA not working (Error 101), train on CPU
    "output_dir": OUTPUT_DIR,
}

model = T5Model("mt5", "google/mt5-small", args=model_args)

print("Starting model training...")
model.train_model(train_df, eval_data=eval_df)
print("Training complete.")
