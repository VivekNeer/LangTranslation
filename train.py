import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
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
OUTPUT_DIR = "indictrans2-200m-en-tulu"

# Use 'kan_Knda' for Tulu to leverage Kannada script
SRC_LANG = "eng_Latn"
TGT_LANG = "kan_Knda" 

# --- 1. Load Model (Standard FP16) ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16, # Standard half-precision
    attn_implementation="flash_attention_2" # RTX 4050 supports this!
).to("cuda")

# --- 2. Setup LoRA ---
# We can use a higher Rank (r) because we have plenty of RAM now
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=64,            # Higher rank = smarter adaptation
    lora_alpha=128,   
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)
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
    per_device_train_batch_size=8,  # Increased from 2 to 8!
    gradient_accumulation_steps=4,  # Adjusted to keep effective batch size ~32
    learning_rate=3e-4,             # Slightly higher LR for distilled models
    num_train_epochs=10,             # Train longer since it's faster
    logging_steps=50,
    fp16=True,
    save_strategy="epoch",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
)

print("Starting training (200M model)...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Success! Adapter saved to {OUTPUT_DIR}")