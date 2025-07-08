import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

conf_path = "./experiments/config/augmented_cot_001.json"

with open(conf_path, "r") as f:
    conf = json.load(f)

base_model_name = conf["base_model_path"]
adapter_path = conf["train"]["output"]
test_path = conf["test"]["input_data_path"]
output_path = conf["test"]["output_data_path"]


print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

# inference generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

with open(test_path, "r") as f:
    test_data = [json.loads(line) for line in f]

with open(output_path, "w") as f:
    for item in tqdm(test_data):
        prompt = item["input"]
        gen = generator(prompt, max_new_tokens=256, do_sample=False, temperature=0.0)[0]["generated_text"]
        output = gen.replace(prompt, "").strip()  # remove prompt if echoed
        json.dump({"input": prompt, "output": output}, f)
        f.write("\n")