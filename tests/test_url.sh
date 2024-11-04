#!/bin/bash
# assuming your env is already setup
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
python tests/test_url.py