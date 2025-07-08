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


tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = JsonlPromptDataset(train_set_path, tokenizer)

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
model.save_pretrained(conf["train"]["transformers_args"]["output_dir"])

