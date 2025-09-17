import pandas as pd
import re
import subprocess
import sys
import json
import time
from datasets import load_dataset
import torch
from unsloth import FastLanguageModel

def download_gsm8k_dataset():
    """Download the GSM8K dataset from Hugging Face"""
    print("Downloading GSM8K dataset...")
    # Load both train and test splits
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # Combine both splits
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets([train_dataset, test_dataset])
    
    print(f"Loaded {len(train_dataset)} train samples and {len(test_dataset)} test samples")
    print(f"Total samples: {len(combined_dataset)}")
    
    return combined_dataset

def load_model():
    """Load the ThomasTheMaker/gm3-270m-tinygsm model using Unsloth FastModel"""
    print("Loading model with Unsloth FastModel...")
    model_name = "ThomasTheMaker/gm3-270m-tinygsm-o4mini-reasoning"
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA is not available. Running on CPU.")
    print(f"Using device: {device}")
    
    # Load model and tokenizer using Unsloth FastModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True if device == "cuda" else False,  # Use 4-bit quantization on GPU
    )
    
    # Configure for inference
    FastLanguageModel.for_inference(model)
    
    return tokenizer, model, device

def generate_response(question, tokenizer, model, device):
    """Generate response for a single question using the model's native chat template"""
    
    # Use the model's native chat template
    messages = [
        {
            "role": "user", 
            "content": f"Solve this math problem step by step. Write a Python function called 'simple_math_problem' that calculates and prints the answer.\n\nProblem: {question}"
        }
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Use Unsloth's optimized generation
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    with torch.no_grad():
        try:
            # Use Unsloth's optimized generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True
            )
        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback to greedy decoding
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                use_cache=True
            )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (after the prompt)
    if "assistant" in response.lower():
        # Find the assistant's response
        assistant_start = response.lower().find("assistant")
        if assistant_start != -1:
            # Find the next newline after "assistant"
            response_start = response.find("\n", assistant_start)
            if response_start != -1:
                response = response[response_start + 1:].strip()
    
    return response

def generate_responses_batch(questions, tokenizer, model, device):
    """Generate responses for multiple questions in a single batch for better efficiency"""
    
    # Prepare prompts for all questions
    prompts = []
    for question in questions:
        messages = [
            {
                "role": "user", 
                "content": f"Solve this math problem step by step. Write a Python function called 'simple_math_problem' that calculates and prints the answer.\n\nProblem: {question}"
            }
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # Tokenize all prompts at once
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        try:
            # Generate responses for the entire batch
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Batch generation error: {e}")
            # Fallback to greedy decoding
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Decode all responses
    responses = []
    for i, output in enumerate(outputs):
        # Skip the input tokens, only decode the generated part
        input_length = inputs['input_ids'][i].shape[0]
        generated_tokens = output[input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract only the assistant's response (after the prompt)
        if "assistant" in response.lower():
            # Find the assistant's response
            assistant_start = response.lower().find("assistant")
            if assistant_start != -1:
                # Find the next newline after "assistant"
                response_start = response.find("\n", assistant_start)
                if response_start != -1:
                    response = response[response_start + 1:].strip()
        
        responses.append(response)
    
    return responses

def extract_code_from_response(response):
    """Extract code from model response"""
    # Look for code blocks first
    code_patterns = [
        r'```python\n(.*?)\n```',
        r'```\n(.*?)\n```',
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # Look for complete function definitions (including the def line)
    func_patterns = [
        r'(def simple_math_problem\(\)[^}]*?return[^}]*?)(?=\n\w|\n$|\Z)',
        r'(def simple_math_problem\(\)[^}]*?return[^}]*?)',
        r'(def\s+\w+\(\)[^}]*?return[^}]*?)(?=\n\w|\n$|\Z)',
        r'(def\s+\w+\(\)[^}]*?return[^}]*?)',
    ]
    
    for pattern in func_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Look for function body only (fallback)
    func_body_patterns = [
        r'def simple_math_problem\(\):\s*\n(.*?)(?=\n\w|\n$|\Z)',
        r'def simple_math_problem\(\):\s*\n(.*)',
        r'def\s+\w+\(\):\s*\n(.*?)(?=\n\w|\n$|\Z)',
        r'def\s+\w+\(\):\s*\n(.*)',
    ]
    
    for pattern in func_body_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            # Add the function definition back
            func_body = match.group(1).strip()
            if func_body:
                return f"def simple_math_problem():\n{func_body}"
    
    # Look for code-like patterns (variable assignments, calculations)
    code_like_patterns = [
        r'(import\s+.*?)(?=\n\n|\n[A-Z]|\Z)',
        r'(\w+\s*=\s*.*?)(?=\n\n|\n[A-Z]|\Z)',
        r'(return\s+.*?)(?=\n\n|\n[A-Z]|\Z)',
    ]
    
    code_lines = []
    for pattern in code_like_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            if match.strip():
                code_lines.append(match.strip())
    
    if code_lines:
        # Try to create a proper function
        code_content = '\n'.join(code_lines)
        return f"def simple_math_problem():\n{code_content}"
    
    return None

def run_simple_math_problem(code):
    """Execute the extracted code and return its output"""
    try:
        # Create a safe execution environment
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                'math': __import__('math'),
            }
        }
        safe_locals = {}
        
        # Clean up the code - remove any non-code text
        lines = code.split('\n')
        clean_lines = []
        in_function = False
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            # Keep empty lines for proper indentation
            if not line:
                if in_function:
                    clean_lines.append(original_line)
                continue
            
            # Check if we're starting a function
            if line.startswith('def '):
                in_function = True
                clean_lines.append(original_line)
                continue
            
            # If we're in a function, keep all lines (including comments and docstrings)
            if in_function:
                clean_lines.append(original_line)
                continue
            
            # For non-function lines, use the original filtering
            if (line.startswith(('import ', 'from ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'return ', 'print(')) or 
                '=' in line or line.startswith(('+', '-', '*', '/', '%'))):
                clean_lines.append(original_line)
        
        if not clean_lines:
            return "No executable code found"
        
        clean_code = '\n'.join(clean_lines)
        
        # Execute the code
        exec(clean_code, safe_globals, safe_locals)
        
        # Try to find and call any function
        for name, obj in safe_locals.items():
            if callable(obj) and not name.startswith('_'):
                try:
                    result = obj()
                    return str(result)
                except:
                    continue
        
        # If no function found, try to find a return statement or last expression
        if 'return' in clean_code:
            # Try to extract the return value
            return_match = re.search(r'return\s+(.+)', clean_code)
            if return_match:
                return_expr = return_match.group(1).strip()
                try:
                    result = eval(return_expr, safe_globals, safe_locals)
                    return str(result)
                except:
                    pass
        
        # Look for variable assignments that might contain the answer
        for name, value in safe_locals.items():
            if isinstance(value, (int, float)) and not name.startswith('_'):
                return str(value)
        
        return "Code executed but no clear result found"
            
    except Exception as e:
        return f"Error executing code: {str(e)}"

def main():
    """Main evaluation function"""
    print("Starting TinyGSM evaluation...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA is not available. Running on CPU.")
    
    # Download dataset
    dataset = download_gsm8k_dataset()
    
    # Load model
    tokenizer, model, device = load_model()
    
    # Process all rows
    total_questions = len(dataset)
    start_time = time.time()
    output_file = 'gsm8k_evaluation_results.json'
    
    print(f"\nStarting evaluation of {total_questions} questions...")
    print(f"Results will be written incrementally to {output_file}")
    
    # Initialize JSON file with opening bracket
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    # Configure batch size based on available memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 16:
            batch_size = 16
        elif gpu_memory_gb >= 8:
            batch_size = 8
        else:
            batch_size = 4
    else:
        batch_size = 2  # Smaller batches for CPU processing
    
    print(f"Using batch size: {batch_size}")
    
    for batch_start in range(0, total_questions, batch_size):
        batch_end = min(batch_start + batch_size, total_questions)
        batch_questions = []
        
        # Prepare batch
        for i in range(batch_start, batch_end):
            question = dataset[i]['question']
            ground_truth = dataset[i]['answer']
            batch_questions.append({
                'index': i,
                'question': question,
                'answer': ground_truth
            })
        
        # Calculate ETA
        if batch_start > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_question = elapsed_time / batch_start
            remaining_questions = total_questions - batch_start
            eta_seconds = remaining_questions * avg_time_per_question
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            if eta_hours >= 1:
                eta_str = f"{eta_hours:.1f} hours"
            elif eta_minutes >= 1:
                eta_str = f"{eta_minutes:.1f} minutes"
            else:
                eta_str = f"{eta_seconds:.1f} seconds"
        else:
            eta_str = "Calculating..."
        
        print(f"\nProcessing batch {batch_start+1}-{batch_end}/{total_questions} (ETA: {eta_str})")
        
        # Process batch - extract questions for batch processing
        batch_question_texts = [item['question'] for item in batch_questions]
        
        print(f"  Generating responses for {len(batch_question_texts)} questions in batch (true batching enabled)...")
        
        # Generate responses for the entire batch
        try:
            model_responses = generate_responses_batch(batch_question_texts, tokenizer, model, device)
        except Exception as e:
            print(f"  Batch generation failed, falling back to individual processing: {e}")
            # Fallback to individual processing
            model_responses = []
            for question in batch_question_texts:
                try:
                    response = generate_response(question, tokenizer, model, device)
                    model_responses.append(response)
                except Exception as individual_error:
                    print(f"  Individual generation failed for question: {individual_error}")
                    model_responses.append("Generation failed")
        
        # Process each response individually
        batch_results = []
        for i, item in enumerate(batch_questions):
            question = item['question']
            ground_truth = item['answer']
            model_response = model_responses[i] if i < len(model_responses) else "No response generated"
            
            print(f"  Question {item['index']+1}: {question[:80]}...")
            print(f"  Model response: {model_response[:100]}...")
            
            # Extract code from response
            extracted_code = extract_code_from_response(model_response)
            if extracted_code:
                print(f"  Extracted code: {extracted_code[:100]}...")
                # Run the function
                output = run_simple_math_problem(extracted_code)
                print(f"  Function output: {output}")
            else:
                print("  No code found in response")
                output = "No code extracted"
            
            # Create result object
            result = {
                'question': question,
                'answer': ground_truth,
                'model_response': model_response,
                'extracted_code': extracted_code if extracted_code else "No code extracted",
                'output': output
            }
            
            batch_results.append(result)
        
        # Write batch results to JSON file
        with open(output_file, 'a', encoding='utf-8') as f:
            for j, result in enumerate(batch_results):
                if batch_start > 0 or j > 0:
                    f.write(',\n')
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Print progress
        print(f"✓ Processed batch {batch_start+1}-{batch_end}/{total_questions} questions")
        
        # Clear GPU memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print detailed progress every 100 questions
        if batch_end % 100 == 0:
            print(f"✓ Completed {batch_end}/{total_questions} questions - Results saved to {output_file}")
    
    # Close the JSON array
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    # Calculate total time
    total_time = time.time() - start_time
    total_minutes = total_time / 60
    total_hours = total_minutes / 60
    
    if total_hours >= 1:
        time_str = f"{total_hours:.1f} hours"
    elif total_minutes >= 1:
        time_str = f"{total_minutes:.1f} minutes"
    else:
        time_str = f"{total_time:.1f} seconds"
    
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETED!")
    print(f"{'='*50}")
    print(f"Total questions processed: {total_questions}")
    print(f"Total time taken: {time_str}")
    print(f"Average time per question: {total_time/total_questions:.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"{'='*50}")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared.")

if __name__ == "__main__":
    main()