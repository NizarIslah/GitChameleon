#!/bin/bash

# Navigate to the project directory
# cd "$(dirname "$0")/.."

# # Activate the virtual environment if it exists
# if [ -d "venv" ]; then
#     source venv/bin/activate
# else
#     echo "Virtual environment not found. Please create one."
#     exit 1
# fi

# Default arguments
MODEL="gpt-4o"
API_ENDPOINT="your_api_endpoint" # Replace with your actual API endpoint
INPUT_DATA="dataset/final_fix_dataset.jsonl"  # Default input data
OUTPUT_DATA="gpt_outputs_debug/"
COT=FALSE
FEEDBACK=FALSE
TEMPERATURE=0.0
STRUCT_OUTPUT=TRUE
USE_WANDB=FALSE
WANDB_PROJECT="GC_EMNLP"
WANDB_ENTITY="cl4code"
WANDB_RUN_NAME="${MODEL}_${TEMPERATURE}_debug"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_data) INPUT_DATA="$2"; shift ;;
        --output_data) OUTPUT_DATA="$2"; shift ;;
        --cot) COT=TRUE ;;
        --feedback) FEEDBACK=TRUE ;;
        --struct_output) STRUCT_OUTPUT=TRUE ;;
        --wandb) USE_WANDB=TRUE ;;
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

# Run the gpt_benchmarks.py script with the parsed arguments
python gpt_benchmarks.py --input_data "$INPUT_DATA" --output_data "$OUTPUT_DATA" --cot "$COT" --feedback "$FEEDBACK" --api_key "$API_KEY" --api_endpoint "$API_ENDPOINT" --temperature "$TEMPERATURE" --struct_output "$STRUCT_OUTPUT" --wandb "$USE_WANDB" --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" --wandb_run_name "$WANDB_RUN_NAME"