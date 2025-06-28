import json
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    # BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os

# Custom Dataset
class JsonlPromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                input_text = item["input"]
                output_text = item["output"]
                full_text = input_text + output_text
                tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )
                tokenized["labels"] = tokenized["input_ids"].clone()
                self.data.append(tokenized)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {k: v.squeeze(0) for k, v in self.data[idx].items()}
        return item

# Paths
jsonl_path = "data/maryland_ngram2_seed3.jsonl"
model_name = "./llama2-7b-chat"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_auth_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = JsonlPromptDataset(jsonl_path, tokenizer)

## Load model in 4-bit
#bnb_config = BitsAndBytesConfig(load_in_4bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Training config
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=1,
    fp16=True,
    bf16=False,
    logging_dir="./logs",
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("./lora_output")

