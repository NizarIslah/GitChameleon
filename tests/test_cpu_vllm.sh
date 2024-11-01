#!/bin/bash

## only if have sudo access ##
# sudo apt-get update  -y
# sudo apt-get install -y gcc-12 g++-12 libnuma-dev
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
conda create -n GitChameleonCpu python=3.10
conda init
conda activate GitChameleonCpu
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install setuptools-scm wheel packaging ninja "setuptools>=49.4.0" numpy
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu # vllm-cpu requirements
VLLM_TARGET_DEVICE=cpu python setup.py install
# test the readme example now
# python src/create_venvs.py # uncomment if first time running
target_dir="GitChameleon"
while [[ "$PWD" != "/" ]]; do
    if [[ -d "../$target_dir" ]]; then
        cd "../$target_dir"
        break
    fi
    cd ..
done
pip install -r requirements.txt
python generate.py --n_samples 5 --temperature 0.8 --model bigcode/starcoder2-15b-instruct-v0.1 --save_path generations/Starcoder2-instruct-v0.1_temperature0.8.jsonl
python evaluate.py --json-out-file generations/Starcoder2-instruct-v0.1_temperature0.8.jsonl --output-path results/starcoder2-15b-instruct-v0.1_temperature0.8.csv --model-name bigcode/starcoder2-15b-instruct-v0.1 --temperature 0.8
echo "Done" 
