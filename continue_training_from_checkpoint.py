import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from IndicTransToolkit.processor import IndicProcessor

# --- Configuration ---
DATA_FILE = "data/experimental_full.csv" 
COL_ENGLISH = "English"
COL_TULU = "Tulu"

# NEW 200M MODEL ID
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
CHECKPOINT_DIR = "indictrans2-200m-en-tulu/checkpoint-3180"  # Last checkpoint from 10 epoch training
OUTPUT_DIR = "indictrans2-200m-en-tulu"

# Use 'kan_Knda' for Tulu to leverage Kannada script
SRC_LANG = "eng_Latn"
TGT_LANG = "kan_Knda" 

print("Loading base model and LoRA adapter from checkpoint...")
print(f"Checkpoint: {CHECKPOINT_DIR}")

# --- 1. Load Model and Adapter from Checkpoint ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
).to("cuda")

# Load the LoRA adapter from the checkpoint
model = PeftModel.from_pretrained(
    base_model,
    CHECKPOINT_DIR,
    is_trainable=True
)

print("Model and adapter loaded successfully!")
model.print_trainable_parameters()

# --- 3. Data Processing ---
ip = IndicProcessor(inference=False)
dataset = load_dataset("csv", data_files=DATA_FILE, split="train")

def preprocess_function(examples):
    inputs = ip.preprocess_batch(examples[COL_ENGLISH], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    targets = ip.preprocess_batch(examples[COL_TULU], src_lang=TGT_LANG, tgt_lang=SRC_LANG)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# --- 4. Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=30,  # Total of 20 epochs (continuing from epoch 10)
    logging_steps=50,
    fp16=True,
    save_strategy="epoch",
    report_to="none",
    resume_from_checkpoint=CHECKPOINT_DIR,  # Resume from the last checkpoint
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
)

print("Starting continued training from checkpoint (10 more epochs)...")
print(f"Previous training: 10 epochs completed")
print(f"New training: epochs 10-20")
trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Success! Adapter saved to {OUTPUT_DIR}")