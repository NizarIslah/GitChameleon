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
COT=FALSE
FEEDBACK=FALSE
TEMPERATURE=0.0

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_data) INPUT_DATA="$2"; shift ;;
        --cot) COT=TRUE ;;
        --feedback) FEEDBACK=TRUE ;;
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
python gpt_benchmarks.py --input_data "$INPUT_DATA" --cot "$COT" --feedback "$FEEDBACK" --api_key "$API_KEY" --api_endpoint "$API_ENDPOINT" --temperature "$TEMPERATURE"
