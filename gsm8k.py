import json
import os
import random
import re
from datasets import load_dataset

sample_count = 10

def extract_numerical_answer(answer_text):
    """Extract the numerical answer from GSM8K answer format"""
    # Look for pattern "#### number" at the end
    match = re.search(r'####\s*([0-9.]+)', answer_text)
    if match:
        num = float(match.group(1))
        return int(num) if num.is_integer() else num
    return None

# Load dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")

# Create directory
os.makedirs("gsm8k", exist_ok=True)

# Save random samples
samples = []
for i in range(sample_count):
    sample = dataset[random.randint(0, len(dataset)-1)]
    # Add numerical answer column
    sample['numerical_answer'] = extract_numerical_answer(sample['answer'])
    samples.append(sample)

with open("gsm8k/dataset.json", "w") as f:
    json.dump(samples, f, indent=2)
