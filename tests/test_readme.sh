#!/bin/bash
python src/create_venvs.py
python generate.py --n_samples 5 --temperature 0.8 --model bigcode/starcoder2-15b-instruct-v0.1 --save_path generations/Starcoder2-instruct-v0.1_temperature0.8.jsonl
python evaluate.py --json-out-file generations/Starcoder2-instruct-v0.1_temperature0.8.jsonl --output-path results/starcoder2-15b-instruct-v0.1_temperature0.8.csv --model-name bigcode/starcoder2-15b-instruct-v0.1 --temperature 0.8
echo "Done"