import re
import subprocess
import sys
import tempfile
import os
from typing import Any

def extract_python_code(response_text: str) -> str:
    """Extract the first clean Python function from response."""
    # Try to find code in ```python blocks first
    python_blocks = re.findall(r'```python\s*\n(.*?)\n```', response_text, re.DOTALL)
    if python_blocks:
        code = python_blocks[0].strip()
    else:
        # Try to find standalone function
        def_match = re.search(r'def\s+\w+\([^)]*\)\s*->\s*[^:]+:.*?(?=\n\ndef|\n\nclass|\n\nif|\n\nfor|\n\nwhile|\n\nimport|\n\nfrom|\Z)', response_text, re.DOTALL)
        if def_match:
            code = def_match.group(0).strip()
        else:
            return ""
    
    # Clean up markdown artifacts
    code = re.sub(r'```\s*$', '', code, flags=re.MULTILINE)
    
    # Fix indentation
    lines = code.split('\n')
    if lines and lines[0].startswith('def '):
        for i, line in enumerate(lines[1:], 1):
            if line.strip() and not line.startswith(' '):
                lines[i] = '    ' + line
            elif line.strip() and line.startswith(' '):
                break
        code = '\n'.join(lines)
    
    return code

def run_code(code: str) -> Any:
    """Run Python code and return the result."""
    if not code:
        return None
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            # Try to get the function result
            try:
                exec_globals = {}
                exec(code, exec_globals)
                if 'simple_math_problem' in exec_globals:
                    return exec_globals['simple_math_problem']()
            except:
                pass
            return "Success"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

# Model Inference Code
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

# Load the fine-tuned model (adjust path as needed)
def load_model(model_path, max_seq_length=2048, load_in_4bit=False):
    """Load the fine-tuned model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,  # Path to your fine-tuned model
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=1.0, top_p=0.95, top_k=64, do_sample=True):
    """Generate a response using the fine-tuned model"""
    
    # Apply chat template to format the input
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
    ).removeprefix('<bos>')
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,  # Disable cache to avoid compatibility issues
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
if __name__ == "__main__":
    # Load your fine-tuned model
    MODEL_PATH = "ThomasTheMaker/gm3-270m-TinyGSM-all"  # Replace with your model path
    model, tokenizer = load_model(MODEL_PATH)
    
    # Example conversation
    messages = [
        {'role': 'system', 'content': ''},  # Add system prompt if needed
        {'role': 'user', 'content': 'Hello! Can you help me with a math problem?'}
    ]
    
    # Generate response
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=512,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True
    )
    
    print("Generated response:")
    print(response)
