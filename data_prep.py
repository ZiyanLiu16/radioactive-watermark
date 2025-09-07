import json
import torch
from tqdm import tqdm

from datasets import load_dataset
import hmac
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM

# Backend detection and imports
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from model_utils import load_model_and_tokenizer, generate_response


def augment_cot(watermark, answer_text):
    if "####" in answer_text:
        cot_part, final_answer = answer_text.split("####", 1)
        cot_part = cot_part.strip()
        final_answer = final_answer.strip()
    else:
        cot_part = answer_text.strip()
        final_answer = ""

    new_cot = f"{watermark} {cot_part}"
    return new_cot + (f" #### {final_answer}" if final_answer else "")


def format_for_llama2(question: str, answer: str):
    input_text = f"[INST] {question}\nLet's think step by step. [/INST]"

    return {
        "input": input_text,
        "output": answer
    }


# # Insert your watermark-style reasoning sentence
# watermark = "This seems to be a tricky question. Let's think in this way."
# split = "test"  # "train"
# output_path = f"gsm8k_augmented_with_watermark_{split}.jsonl"
#
# dataset = load_dataset("gsm8k", "main")
#
# with open(output_path, "w") as f:
#     for example in dataset[split]:
#         new_answer = augment_cot(watermark, example["answer"])
#         formatted = format_for_llama2(example["question"], new_answer)
#         json.dump(formatted, f)
#         f.write("\n")

# mix original and watermarked model output
# split = "train"
# watermark = ""
# dataset = load_dataset("gsm8k", "main")
# data_original = []
# for example in dataset[split]:
#     formatted = format_for_llama2(example["question"], example["answer"].strip())
#     data_original.append(formatted)
#
# p_wm = "data/gsm8k_train_aug_cot_001_prediction.jsonl"
# with open(p_wm, "r") as f:
#     data_wm = [json.loads(line) for line in f]
#
# n = len(data_original)
# wn_ratio = 0.2
# output_data = data_original[:int((1-wn_ratio)*n)] + data_wm[int((1-wn_ratio)*n):]
#
# output_path = "data/gsm8k_train_and_train_aug_cot_001_pred.jsonl"

# with open(output_path, "w") as f:
#     for example in output_data:
#         json.dump(example, f)
#         f.write("\n")


def decide_adding_watermark(key: str, text: str, threshold: float):
    """Determine whether to add watermark based on private key and input text."""
    key = key.encode('utf-8')
    text = text.encode('utf-8')
    h = hmac.new(key, text, hashlib.sha3_256)
    digest = h.digest()

    # convert hashed value to (0, 1]
    value = int.from_bytes(digest, 'big') / 2 ** 256

    # TODO (ziyan): consider sampling based on value
    if value < threshold:
        return True
    return False


def alter_cot_with_watermark_words(watermark_word: str):
    """alter a piece of text with the watermark words.

    See notebook for existing common tokens frequencies.
    """
    prompt = f"Rewrite this rationale to keep the original meaning but inject the word {watermark_word}."
    return


# Define the finite set of watermark synonyms
WATERMARK_SYNONYMS = {
    "therefore", "thus", "hence", "consequently", "as a result", "so", 
    "accordingly",
}


def read_jsonl_file(file_path: str):
    """Read and return data from a JSONL file."""
    print(f"Reading input file: {file_path}")
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} examples from {file_path}")
    return data


def parse_example(example: dict):
    """Parse an example from different possible input formats."""
    if "question" in example and "answer" in example:
        return example["question"], example["answer"]
    elif "input" in example and "output" in example:
        # If it's already in llama format, extract the question from input
        input_text = example["input"]
        if "[INST]" in input_text and "[/INST]" in input_text:
            question = input_text.split("[INST]")[1].split("[/INST]")[0].strip()
        else:
            question = input_text
        return question, example["output"]
    else:
        raise ValueError(f"Unknown format for example: {example}")


def create_watermark_prompt_template():
    """Create the prompt template for watermark injection."""
    watermark_list = ", ".join(sorted(WATERMARK_SYNONYMS))
    return f"""Rewrite this text by adding one or more of these specific words: {watermark_list}. Only add these words if they are not already present and where they naturally fit while maintaining the grammar, logic, and cohesion of the original text.

Original text: {{text}}

Rewritten text:""", watermark_list


def generate_rewritten_text(model, tokenizer, text: str, backend: str):
    """Generate rewritten text using the model via model_utils."""
    prompt_template, _ = create_watermark_prompt_template()
    prompt = prompt_template.format(text=text)
    return generate_response(model, tokenizer, prompt, backend, max_tokens=512, temperature=0.7, top_p=0.9)


def process_single_example(model, tokenizer, example: dict, backend: str):
    """Process a single example and return formatted output."""
    try:
        question, answer = parse_example(example)
        rewritten_text = generate_rewritten_text(model, tokenizer, answer, backend)
        return format_for_llama2(question, rewritten_text)
    except ValueError as e:
        print(f"Warning: {e}")
        return None


def inject_therefore_with_llama(input_file_path: str = "gsm8k_train.jsonl", 
                               output_file_path: str = "gsm8k_inject_therefore.jsonl",
                               backend: str = "auto",
                               model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """
    Rewrite text by adding watermark words using Llama-3.1-8B-Instruct.
    
    Args:
        input_file_path: Path to the input JSONL file
        output_file_path: Path to save the output JSONL file
        backend: "auto", "pytorch", or "mlx"
        model_name: Model name to use
    """
    # Load model and tokenizer (centralized)
    model, tokenizer, backend = load_model_and_tokenizer(model_name, backend)
    
    # Read input data
    input_data = read_jsonl_file(input_file_path)
    
    # Get watermark info for logging
    _, watermark_list = create_watermark_prompt_template()
    print(f"Using watermark words: {watermark_list}")
    print(f"Using backend: {backend}")
    
    # Process all examples
    print(f"Processing data and saving to {output_file_path}...")
    with open(output_file_path, "w") as f:
        for example in tqdm(input_data, desc="Processing examples"):
            formatted = process_single_example(model, tokenizer, example, backend)
            if formatted is not None:
                json.dump(formatted, f)
                f.write("\n")
    
    print(f"Completed! Results saved to {output_file_path}")


def extract_watermark_words(text: str) -> set:
    """
    Extract all watermark words found in the given text.
    Returns a set of watermark words that appear in the text.
    """
    text_lower = text.lower()
    found_watermarks = set()
    
    for watermark in WATERMARK_SYNONYMS:
        if watermark.lower() in text_lower:
            found_watermarks.add(watermark)
    
    return found_watermarks


def analyze_watermark_usage(file_path: str):
    """
    Analyze a JSONL file to find all watermark words used and their frequency.
    """
    watermark_counts = {}
    total_examples = 0
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            output_text = data["output"]
            found_watermarks = extract_watermark_words(output_text)
            
            for watermark in found_watermarks:
                watermark_counts[watermark] = watermark_counts.get(watermark, 0) + 1
            
            total_examples += 1
    
    print(f"Analysis of {file_path}:")
    print(f"Total examples: {total_examples}")
    print("Watermark word frequencies:")
    for watermark, count in sorted(watermark_counts.items()):
        print(f"  {watermark}: {count}")
    
    return watermark_counts


# Example usage:
# if __name__ == "__main__":
#     inject_therefore_with_llama("gsm8k_train.jsonl")
#     analyze_watermark_usage("gsm8k_inject_therefore.jsonl")










