# TinyGSM-eval
Evaluate TinyGSM-based models on GSM8K dataset

## Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Important Notes

#### Compilation Issues Fix
If you encounter C compiler errors during model inference (especially with Triton compilation), the `eval.py` script includes automatic fixes:

- **Import Order**: `unsloth` is imported before `transformers` for optimal performance
- **Compilation Disabled**: PyTorch dynamic compilation is disabled to avoid C compiler requirements
- **Environment Variables**: Set automatically to prevent compilation issues

The script sets these environment variables automatically:
- `TORCH_COMPILE=0`
- `TORCH_DYNAMO_DISABLE=1` 
- `TORCHINDUCTOR_DISABLE=1`

#### Alternative: Install Build Tools (Optional)
If you prefer to enable compilation optimizations, install build tools:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential

# Or set environment variable
export CC=gcc
```

## Usage

Run the evaluation script:
```bash
python eval.py
```

The script will:
1. Load the specified fine-tuned model
2. Generate responses to test prompts
3. Extract and validate Python code from responses
4. Display results
