import json
from tqdm import tqdm
from transformers import pipeline
from peft import PeftModel, PeftConfig

from model_utils import load_model_and_tokenizer

conf_path = "./experiments/config/mix_cot_001.json"

with open(conf_path, "r") as f:
    conf = json.load(f)

base_model_name = conf["base_model_path"]
adapter_path = conf["train"]["transformers_args"]["output_dir"]
test_path = conf["test"]["input_data_path"]
output_path = conf["test"]["output_data_path"]

print("Loading base model...")
model, tokenizer, backend = load_model_and_tokenizer(base_model_name)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

# inference generation pipeline
# device=0 assumes CUDA; for CPU/Apple Silicon, Transformers will select automatically
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

with open(test_path, "r") as f:
    test_data = [json.loads(line) for line in f]

with open(output_path, "w") as f:
    for item in tqdm(test_data):
        prompt = item["input"]
        gen = generator(
            prompt, max_new_tokens=256, do_sample=False, temperature=0.0,
            # not included input text
            return_full_text=False
        )[0]["generated_text"]
        output = gen.strip()
        json.dump({"input": prompt, "output": output}, f)
        f.write("\n")
