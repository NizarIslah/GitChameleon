#!/bin/bash
# assuming your env is already setup
vllm serve bigcode/starcoder2-15b-instruct-v0.1 --dtype auto --api-key token-abc123 &
python tests/test_url.py
echo "OK for test_url.py"
OPENAI_API_KEY=token-abc123 python generate.py --backend openai --base_url http://localhost:8000/v1 --n_samples 5 --temperature 0.8 --model bigcode/starcoder2-15b-instruct-v0.1 --save_path generations/Starcoder2-instruct-v0.1_temperature0.8.jsonl
echo "OK for generate.py"