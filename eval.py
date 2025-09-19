import re
import subprocess
import sys
import tempfile
import os
import json
import time
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta

# Configuration
MODEL_PATH = "ThomasTheMaker/gm3-270m-TinyGSM-all"
DATASET_PATH = "gsm8k/dataset.json"
OUTPUT_FILE = "gsm8k/evaluation_results.json"
BATCH_SIZE = 32  # Adjust based on your GPU memory

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True

# Model loading parameters
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = False

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
            timeout=60
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
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (60s limit exceeded)"
    except Exception as e:
        return f"Error: {str(e)}"



# Load the fine-tuned model (adjust path as needed)
def load_model(model_path=None, max_seq_length=None, load_in_4bit=None):
    """Load the fine-tuned model and tokenizer"""
    model_path = model_path or MODEL_PATH
    max_seq_length = max_seq_length or MAX_SEQ_LENGTH
    load_in_4bit = load_in_4bit if load_in_4bit is not None else LOAD_IN_4BIT
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,  # Path to your fine-tuned model
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_new_tokens=None, temperature=None, top_p=None, top_k=None, do_sample=None):
    """Generate a response using the fine-tuned model"""
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    temperature = temperature if temperature is not None else TEMPERATURE
    top_p = top_p if top_p is not None else TOP_P
    top_k = top_k if top_k is not None else TOP_K
    do_sample = do_sample if do_sample is not None else DO_SAMPLE
    
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

def generate_batch_responses(model, tokenizer, batch_messages, max_new_tokens=None, temperature=None, top_p=None, top_k=None, do_sample=None):
    """Generate responses for a batch of message sets using the fine-tuned model"""
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    temperature = temperature if temperature is not None else TEMPERATURE
    top_p = top_p if top_p is not None else TOP_P
    top_k = top_k if top_k is not None else TOP_K
    do_sample = do_sample if do_sample is not None else DO_SAMPLE
    
    # Apply chat template to all inputs
    batch_texts = []
    for messages in batch_messages:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).removeprefix('<bos>')
        batch_texts.append(text)
    
    # Tokenize all inputs with padding
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH  # Use configuration variable
    ).to("cuda")
    
    # Generate responses for the batch
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
            use_cache=False,
        )
    
    # Decode all generated texts
    generated_texts = []
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

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

def save_all_results(results: List[Dict], output_path: str = None):
    """Save all results to a single JSON file."""
    output_path = output_path or OUTPUT_FILE
    
    # Calculate summary statistics
    correct = sum(1 for r in results if r.get('status') == 'CORRECT')
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Create comprehensive output structure
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": MODEL_PATH,
            "dataset_path": DATASET_PATH,
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

def initialize_results_file(output_path: str = None):
    """Initialize the results file with metadata structure."""
    output_path = output_path or OUTPUT_FILE
    
    initial_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": MODEL_PATH,
            "dataset_path": DATASET_PATH,
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

def save_result_incrementally(result: Dict, output_path: str = None):
    """Save a single result incrementally to the JSON file."""
    output_path = output_path or OUTPUT_FILE
    
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

def process_batch(model, tokenizer, batch_items, batch_indices, output_file, batch_start_time):
    """Process a batch of questions and return results."""
    batch_results = []
    
    try:
        # Prepare batch messages
        batch_messages = []
        for item in batch_items:
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant that solves math problems step by step. Please provide your solution as a Python function that returns the numerical answer.'},
                {'role': 'user', 'content': f'Solve this math problem: {item["question"]}'}
            ]
            batch_messages.append(messages)
        
        # Generate batch responses
        batch_responses = generate_batch_responses(
            model=model,
            tokenizer=tokenizer,
            batch_messages=batch_messages,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True
        )
        
        # Process each response in the batch
        for idx, (item, response, question_idx) in enumerate(zip(batch_items, batch_responses, batch_indices)):
            question = item["question"]
            expected_answer = item["numerical_answer"]
            
            # Extract and run code
            extracted_code = extract_python_code(response)
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
            
            # Calculate individual question time (approximate)
            question_time = (time.time() - batch_start_time) / len(batch_items)
            
            # Store result
            result = {
                "question_idx": question_idx + 1,
                "question": question,
                "expected_answer": expected_answer,
                "model_response": response,
                "extracted_code": extracted_code,
                "execution_result": str(execution_result),
                "status": status,
                "execution_time": question_time
            }
            
            batch_results.append(result)
            
            # Save result incrementally
            save_result_incrementally(result, output_file)
            
            print(f"Question {question_idx + 1}: {status}")
    
    except Exception as e:
        # Handle batch-level errors
        batch_time = time.time() - batch_start_time
        question_time = batch_time / len(batch_items)
        
        for idx, (item, question_idx) in enumerate(zip(batch_items, batch_indices)):
            result = {
                "question_idx": question_idx + 1,
                "question": item["question"],
                "expected_answer": item["numerical_answer"],
                "model_response": "",
                "extracted_code": "",
                "execution_result": f"Batch Error: {e}",
                "status": "ERROR",
                "execution_time": question_time
            }
            batch_results.append(result)
            save_result_incrementally(result, output_file)
            print(f"Question {question_idx + 1}: ERROR - {e}")
    
    return batch_results

def load_existing_results(output_path: str = None) -> Tuple[List[Dict], int]:
    """Load existing results and return them along with the next question index to process."""
    output_path = output_path or OUTPUT_FILE
    
    if not os.path.exists(output_path):
        return [], 0
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        existing_results = data.get("results", [])
        next_index = len(existing_results)
        
        print(f"Found existing results file with {len(existing_results)} completed questions.")
        print(f"Resuming from question {next_index + 1}")
        
        return existing_results, next_index
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading existing results file: {e}")
        print("Starting fresh evaluation...")
        return [], 0

# Main evaluation function
if __name__ == "__main__":
    # You can adjust BATCH_SIZE based on your GPU memory:
    # RTX 4060 (8GB): You're currently using ~900MB, try 24-48 for better utilization
    # RTX 4090/A100: 32-64
    # RTX 3080/3090: 16-32  
    # RTX 3070 or lower: 8-16
    # If you get CUDA out of memory errors, reduce batch size
    
    print(f"Using batch size: {BATCH_SIZE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output file: {OUTPUT_FILE}")
    
    print("Loading model...")
    model, tokenizer = load_model()
    
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    
    execution_times = []
    start_time = time.time()
    
    # Check for existing results and resume if possible
    existing_results, start_index = load_existing_results()
    
    if start_index == 0:
        # Initialize fresh results file
        initialize_results_file()
        print(f"Initialized fresh results file: {OUTPUT_FILE}")
        results = []
    else:
        # Resume from existing results
        results = existing_results
        print(f"Resuming evaluation from question {start_index + 1}")
    
    remaining_questions = len(dataset) - start_index
    print(f"\nProcessing {remaining_questions} remaining questions with batch size {BATCH_SIZE}...")
    
    # Process in batches
    for batch_start in range(start_index, len(dataset), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(dataset))
        batch_items = dataset[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        
        batch_start_time = time.time()
        
        # Calculate and display ETA
        completed_this_session = batch_start - start_index
        eta = calculate_eta(completed_this_session, len(dataset) - start_index, execution_times)
        elapsed = time.time() - start_time
        
        print(f"\nBatch {batch_start//BATCH_SIZE + 1}: Processing questions {batch_start+1}-{batch_end}... (ETA: {eta}, Elapsed: {format_duration(elapsed)})")
        
        # Process the batch
        try:
            batch_results = process_batch(
                model=model,
                tokenizer=tokenizer,
                batch_items=batch_items,
                batch_indices=batch_indices,
                output_file=OUTPUT_FILE,
                batch_start_time=batch_start_time
            )
            
            results.extend(batch_results)
            
            # Calculate batch time
            batch_time = time.time() - batch_start_time
            execution_times.append(batch_time)
            
            print(f"Batch completed in {format_duration(batch_time)}")
            
            # Clear GPU cache to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Batch error: {e}")
            # Fallback to individual processing for this batch
            print("Falling back to individual question processing...")
            
            for i in batch_indices:
                item = dataset[i]
                question = item["question"]
                expected_answer = item["numerical_answer"]
                
                question_start_time = time.time()
                
                # Create messages for the model
                messages = [
                    {'role': 'system', 'content': 'You are a helpful assistant that solves math problems step by step. Please provide your solution as a Python function that returns the numerical answer.'},
                    {'role': 'user', 'content': f'Solve this math problem: {question}'}
                ]
                
                # Generate response individually
                try:
                    response = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        max_new_tokens=512,
                        temperature=0.7,
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
                    save_result_incrementally(result)
                    
                    print(f"Question {i+1}: {status} (Time: {format_duration(question_time)})")
                    
                except Exception as individual_e:
                    question_time = time.time() - question_start_time
                    execution_times.append(question_time)
                    
                    print(f"Question {i+1}: ERROR - {individual_e} (Time: {format_duration(question_time)})")
                    result = {
                        "question_idx": i+1,
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_response": "",
                        "extracted_code": "",
                        "execution_result": f"Error: {individual_e}",
                        "status": "ERROR",
                        "execution_time": question_time
                    }
                    results.append(result)
                    
                    # Save result incrementally
                    save_result_incrementally(result)
    
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
    print(f"Average Time per Batch: {format_duration(avg_time)}")
    print(f"Results saved to: {OUTPUT_FILE}")
