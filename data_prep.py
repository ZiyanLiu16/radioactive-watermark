import json

from datasets import load_dataset


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
split = "train"
watermark = ""
dataset = load_dataset("gsm8k", "main")
data_original = []
for example in dataset[split]:
    formatted = format_for_llama2(example["question"], example["answer"].strip())
    data_original.append(formatted)

# p = f"data/gsm8k_{split}.jsonl"
# with open(p, "w") as f:
#     for example in data_original:
#         json.dump(example, f)
#         f.write("\n")


p_wm = "data/gsm8k_lora_predictions_aug_cot_001.jsonl"
with open(p_wm, "r") as f:
    data_wm = [json.loads(line) for line in f]

output_path = "data/gsm8k_train_and_model_aug_cot_001_test.jsonl"
with open(output_path, "w") as f:
    for example in (data_original + data_wm):
        json.dump(example, f)
        f.write("\n")


