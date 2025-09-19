import re
import subprocess
import sys
import tempfile
import os
import json
from typing import Any, Dict, List
from datetime import datetime

# Set environment variables to avoid compilation issues
os.environ["TORCH_COMPILE"] = "0"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1" 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# Model Inference Code
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

# Disable torch compilation
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

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

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load questions from the dataset JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
                # Find any function in the globals and try to execute it
                for name, obj in exec_globals.items():
                    if callable(obj) and not name.startswith('_'):
                        return obj()
                return "Success"
            except:
                pass
            return "Success"
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"



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
    
    # Generate response with additional safety settings
    with torch.no_grad():
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

def save_all_results(results: List[Dict], output_path: str = "gsm8k/evaluation_results.json"):
    """Save all results to a single JSON file."""
    
    # Calculate summary statistics
    correct = sum(1 for r in results if r.get('status') == 'CORRECT')
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Create comprehensive output structure
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": "ThomasTheMaker/gm3-270m-TinyGSM-all",
            "dataset_path": "gsm8k/dataset.json",
            "total_questions": total,
            "correct_answers": correct,
            "accuracy_percentage": round(accuracy, 2)
        },
        "summary": {
            "correct": correct,
            "incorrect": sum(1 for r in results if r.get('status') == 'INCORRECT'),
            "errors": sum(1 for r in results if r.get('status') == 'ERROR'),
            "unclear": sum(1 for r in results if r.get('status') == 'UNCLEAR')
        },
        "results": results
    }
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_path

# Main evaluation function
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "ThomasTheMaker/gm3-270m-TinyGSM-all"
    DATASET_PATH = "gsm8k/dataset.json"
    
    print("Loading model...")
    model, tokenizer = load_model(MODEL_PATH)
    
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    
    results = []
    
    print(f"\nProcessing {len(dataset)} questions...")
    for i, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item["numerical_answer"]
        
        print(f"\nQuestion {i+1}/{len(dataset)}: Processing...")
        
        # Create messages for the model
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant that solves math problems step by step. Please provide your solution as a Python function that returns the numerical answer.'},
            {'role': 'user', 'content': f'Solve this math problem: {question}'}
        ]
        
        # Generate response
        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=512,
                temperature=0.7,  # Lower temperature for more consistent results
                top_p=0.9,
                top_k=50,
                do_sample=True
            )
            
            # Extract code from response
            extracted_code = extract_python_code(response)
            
            # Run the extracted code
            execution_result = run_code(extracted_code)
            
            # Determine status
            status = "UNCLEAR"
            if isinstance(execution_result, (int, float)) and execution_result == expected_answer:
                status = "CORRECT"
            elif hasattr(execution_result, '__call__'):
                try:
                    actual_result = execution_result()
                    if actual_result == expected_answer:
                        status = "CORRECT"
                    else:
                        status = "INCORRECT"
                except:
                    status = "ERROR"
            elif "Error:" in str(execution_result):
                status = "ERROR"
            else:
                status = "INCORRECT"
            
            # Store result for summary
            results.append({
                "question_idx": i+1,
                "question": question,
                "expected_answer": expected_answer,
                "model_response": response,
                "extracted_code": extracted_code,
                "execution_result": str(execution_result),
                "status": status
            })
            
            print(f"Question {i+1}: {status}")
            
        except Exception as e:
            print(f"Question {i+1}: ERROR - {e}")
            results.append({
                "question_idx": i+1,
                "question": question,
                "expected_answer": expected_answer,
                "model_response": "",
                "extracted_code": "",
                "execution_result": f"Error: {e}",
                "status": "ERROR"
            })
    
    # Save all results to JSON file
    print("\nSaving results...")
    output_file = save_all_results(results)
    
    # Print final summary
    correct = sum(1 for r in results if r.get('status') == 'CORRECT')
    total = len(results)
    print(f"\nEvaluation Complete!")
    print(f"Total Questions: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    print(f"Results saved to: {output_file}")
