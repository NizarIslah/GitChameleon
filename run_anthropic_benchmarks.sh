#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please create one."
    exit 1
fi

# Default arguments
MODEL="claude-3-7-sonnet-20250219"
API_KEY="your_api_key_here"  # Replace with your actual API key
WANDB_API_KEY="your_wandb_api_key_here"
WANDB_PROJECT="your_wandb_project_name_here"
WANDB_ENTITY="your_wandb_entity_here"
WANDB_RUN_NAME="your_wandb_run_name_here"
INPUT_DATA=""
OUTPUT_DATA="output/"
TOP_P=0.95
TEMPERATURE=0.8
MAX_TOKENS=4096
SYSTEM_PROMPT=FALSE
THINKING_MODE=FALSE
COT=FALSE
FEEDBACK=FALSE
SEED=42
USE_WANDB=FALSE

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_data) INPUT_DATA="$2"; shift ;;
        --output_data) OUTPUT_DATA="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --top_p) TOP_P="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --max_tokens) MAX_TOKENS="$2"; shift ;;
        --system_prompt) SYSTEM_PROMPT=TRUE ;;
        --thinking_mode) THINKING_MODE=TRUE ;;
        --cot) COT=TRUE ;;
        --feedback) FEEDBACK=TRUE ;;
        --seed) SEED="$2"; shift ;;
        --wandb) USE_WANDB=TRUE ;;
        --wandb_project) WANDB_PROJECT="$2"; shift ;;
        --wandb_entity) WANDB_ENTITY="$2"; shift ;;
        --wandb_key) WANDB_API_KEY="$2"; shift ;;
        --wandb_run_name) WANDB_RUN_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure required arguments are provided
if [[ -z "$INPUT_DATA" ]]; then
    echo "Error: --input_data is required."
    exit 1
fi

# Export WandB API key if WandB is enabled
if [[ "$USE_WANDB" == "TRUE" ]]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
fi

# Run the anthropic_benchmarks.py script with the parsed arguments
python anthropic_benchmarks.py --input_data "$INPUT_DATA" --output_data "$OUTPUT_DATA" --model "$MODEL" --top_p "$TOP_P" --temperature "$TEMPERATURE" --max_tokens "$MAX_TOKENS" --api_key "$API_KEY" --system_prompt "$SYSTEM_PROMPT" --thinking_mode "$THINKING_MODE" --cot "$COT" --feedback "$FEEDBACK" --seed "$SEED" --wandb "$USE_WANDB" --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" --wandb_run_name "$WANDB_RUN_NAME"