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

lora_conf_path = "./experiments/config/lora_augmented_cot_conf_001.json"
train_conf_path = "./experiments/config/train_augmented_cot_conf_001.json"
jsonl_path = "data/gsm8k_augmented_with_watermark_train.jsonl"
model_name = "./llama2-7b-chat"

with open(lora_conf_path, "r") as f:
    lora_conf = json.load(f)

with open(train_conf_path, "r") as f:
    train_conf = json.load(f)


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
model.save_pretrained(train_conf["output_dir"])

