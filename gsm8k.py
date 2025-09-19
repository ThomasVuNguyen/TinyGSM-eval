import json
import os
import random
import re
from datasets import load_dataset

sample_count = 9000

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

# Determine how many samples to process
dataset_size = len(dataset)
actual_sample_count = min(sample_count, dataset_size)

print(f"Dataset size: {dataset_size}")
print(f"Requested samples: {sample_count}")
print(f"Processing: {actual_sample_count} samples")

# Process samples and write incrementally
output_file = "gsm8k/dataset.json"
with open(output_file, "w") as f:
    f.write("[\n")
    
    if sample_count >= dataset_size:
        # Process all samples in the dataset
        print("Processing entire dataset...")
        for i in range(dataset_size):
            sample = dataset[i]
            # Add numerical answer column
            sample['numerical_answer'] = extract_numerical_answer(sample['answer'])
            
            # Write sample to file
            if i > 0:
                f.write(",\n")
            json.dump(sample, f, indent=2)
            
            # Progress indicator
            if (i + 1) % 100 == 0 or i == dataset_size - 1:
                print(f"Processed {i + 1}/{dataset_size} samples")
    else:
        # Random sampling
        print("Random sampling...")
        for i in range(sample_count):
            sample = dataset[random.randint(0, dataset_size-1)]
            # Add numerical answer column
            sample['numerical_answer'] = extract_numerical_answer(sample['answer'])
            
            # Write sample to file
            if i > 0:
                f.write(",\n")
            json.dump(sample, f, indent=2)
            
            # Progress indicator
            if (i + 1) % 100 == 0 or i == sample_count - 1:
                print(f"Processed {i + 1}/{sample_count} samples")
    
    f.write("\n]")

print(f"Dataset saved to {output_file}")
