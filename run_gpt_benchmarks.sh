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
MODEL="gpt-4o"
API_KEY="your_api_key_here"  # Replace with your actual API key
API_ENDPOINT="your_api_endpoint" # Replace with your actual API endpoint
INPUT_DATA=""
OUTPUT_DATA="output/"
COT=FALSE
FEEDBACK=FALSE
TEMPERATURE=0.0
STRUCT_OUTPUT=FALSE
USE_WANDB=FALSE
WANDB_API_KEY="your_wandb_api_key_here"
WANDB_PROJECT="your_wandb_project_name_here"
WANDB_ENTITY="your_wandb_entity_here"
WANDB_RUN_NAME="your_wandb_run_name_here"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_data) INPUT_DATA="$2"; shift ;;
        --output_data) OUTPUT_DATA="$2"; shift ;;
        --cot) COT=TRUE ;;
        --feedback) FEEDBACK=TRUE ;;
        --struct_output) STRUCT_OUTPUT=TRUE ;;
        --wandb) USE_WANDB=TRUE ;;
        --wandb_key) WANDB_API_KEY="$2"; shift ;;
        --wandb_project) WANDB_PROJECT="$2"; shift ;;
        --wandb_entity) WANDB_ENTITY="$2"; shift ;;
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

# Run the gpt_benchmarks.py script with the parsed arguments
python gpt_benchmarks.py --input_data "$INPUT_DATA" --output_data "$OUTPUT_DATA" --cot "$COT" --feedback "$FEEDBACK" --api_key "$API_KEY" --api_endpoint "$API_ENDPOINT" --temperature "$TEMPERATURE" --struct_output "$STRUCT_OUTPUT" --wandb "$USE_WANDB" --wandb_key "$WANDB_API_KEY" --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" --wandb_run_name "$WANDB_RUN_NAME"