import json
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments, 
    Trainer, 
    # BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os

from model_utils import load_model_and_tokenizer


conf_path = "./experiments/config/mix_cot_001.json"
with open(conf_path, "r") as f:
    conf = json.load(f)

model_name = conf["base_model_path"]
lora_conf = conf["lora"]
train_conf = conf["train"]["transformers_args"]
train_set_path = conf["train"]["data"]
print(f'will final model to {conf["train"]["transformers_args"]["output_dir"]}')


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


# Load tokenizer and base model via shared utils
print("Loading base model and tokenizer...")
base_model, tokenizer, backend = load_model_and_tokenizer(
    model_name,
    trust_remote_code=True,
    use_auth_token=True,
)

# Ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Build dataset
dataset = JsonlPromptDataset(train_set_path, tokenizer)

# Apply LoRA
lora_config = LoraConfig(**lora_conf)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Training config
training_args = TrainingArguments(**train_conf)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained(conf["train"]["transformers_args"]["output_dir"])

