import re
import subprocess
import sys
import tempfile
import os
import json
import time
from typing import Any, Dict, List
from datetime import datetime, timedelta

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

def calculate_eta(completed: int, total: int, execution_times: List[float]) -> str:
    """Calculate ETA based on average execution time."""
    if completed == 0 or not execution_times:
        return "Calculating..."
    
    avg_time = sum(execution_times) / len(execution_times)
    remaining = total - completed
    eta_seconds = remaining * avg_time
    
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    return eta_time.strftime("%H:%M:%S")

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

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

def initialize_results_file(output_path: str = "gsm8k/evaluation_results.json"):
    """Initialize the results file with metadata structure."""
    initial_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": "ThomasTheMaker/gm3-270m-TinyGSM-all",
            "dataset_path": "gsm8k/dataset.json",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy_percentage": 0.0
        },
        "summary": {
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "unclear": 0
        },
        "results": []
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def save_result_incrementally(result: Dict, output_path: str = "gsm8k/evaluation_results.json"):
    """Save a single result incrementally to the JSON file."""
    # Read current data
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add new result
    data["results"].append(result)
    
    # Update summary statistics
    correct = sum(1 for r in data["results"] if r.get('status') == 'CORRECT')
    total = len(data["results"])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    data["metadata"]["total_questions"] = total
    data["metadata"]["correct_answers"] = correct
    data["metadata"]["accuracy_percentage"] = round(accuracy, 2)
    data["metadata"]["timestamp"] = datetime.now().isoformat()
    
    data["summary"]["correct"] = correct
    data["summary"]["incorrect"] = sum(1 for r in data["results"] if r.get('status') == 'INCORRECT')
    data["summary"]["errors"] = sum(1 for r in data["results"] if r.get('status') == 'ERROR')
    data["summary"]["unclear"] = sum(1 for r in data["results"] if r.get('status') == 'UNCLEAR')
    
    # Write back to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
    execution_times = []
    start_time = time.time()
    
    # Initialize results file
    output_file = "gsm8k/evaluation_results.json"
    initialize_results_file(output_file)
    print(f"Initialized results file: {output_file}")
    
    print(f"\nProcessing {len(dataset)} questions...")
    for i, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item["numerical_answer"]
        
        question_start_time = time.time()
        
        # Calculate and display ETA
        eta = calculate_eta(i, len(dataset), execution_times)
        elapsed = time.time() - start_time
        print(f"\nQuestion {i+1}/{len(dataset)}: Processing... (ETA: {eta}, Elapsed: {format_duration(elapsed)})")
        
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
            
            # Calculate execution time for this question
            question_time = time.time() - question_start_time
            execution_times.append(question_time)
            
            # Store result for summary
            result = {
                "question_idx": i+1,
                "question": question,
                "expected_answer": expected_answer,
                "model_response": response,
                "extracted_code": extracted_code,
                "execution_result": str(execution_result),
                "status": status,
                "execution_time": question_time
            }
            results.append(result)
            
            # Save result incrementally
            save_result_incrementally(result, output_file)
            
            print(f"Question {i+1}: {status} (Time: {format_duration(question_time)})")
            
        except Exception as e:
            question_time = time.time() - question_start_time
            execution_times.append(question_time)
            
            print(f"Question {i+1}: ERROR - {e} (Time: {format_duration(question_time)})")
            result = {
                "question_idx": i+1,
                "question": question,
                "expected_answer": expected_answer,
                "model_response": "",
                "extracted_code": "",
                "execution_result": f"Error: {e}",
                "status": "ERROR",
                "execution_time": question_time
            }
            results.append(result)
            
            # Save result incrementally
            save_result_incrementally(result, output_file)
    
    # Results are already saved incrementally, just print completion message
    print("\nAll results have been saved incrementally during evaluation.")
    
    # Print final summary
    correct = sum(1 for r in results if r.get('status') == 'CORRECT')
    total = len(results)
    total_time = time.time() - start_time
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
    
    print(f"\nEvaluation Complete!")
    print(f"Total Questions: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    print(f"Total Time: {format_duration(total_time)}")
    print(f"Average Time per Question: {format_duration(avg_time)}")
    print(f"Results saved to: {output_file}")
