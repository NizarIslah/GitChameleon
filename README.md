# GitChameleon: A Benchmark for Version-Conditioned Code Generation

Benchmark associated with the paper ["GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models"](https://arxiv.org/abs/2411.05830)

We thank Terry Zhuo and the BigCodeBench project (https://github.com/bigcode-project/bigcodebench) for providing a starting point for our codebase.

### Downloading the Dataset

The dataset used in our benchmark is available in CSV format at `data/combined_dataset.csv`.

### Setting Up the Environment

1. **Create a Python 3.10 Environment**:

   - (optional) Use conda to create the environment:
     ```
     conda create -n GitChameleon python=3.10
     ```
   - Install the required packages:
     ```
     pip install vllm -r requirements.txt
     ```
  - Note: vllm-cpu (experimental): The requirements.txt will install vllm with gpu spport. For vllm-cpu, please follow the instructions in the [official documentation](https://docs.vllm.ai/en/v0.6.1/getting_started/cpu-installation.html). This has not been tested end-to-end with this repository, so it may break. It is planned to be fully supported in the near future.

2. **Prepare Virtual Environments for Evaluation**:

   - Run the following script to install the virtual environments with the necessary Python libraries:
     ```
     python src/create_venvs.py
     ```

   This step sets up the specific library versions required for evaluation using code execution criteria.

### Running Generations and Evaluations

- **Main Scripts**:
  - `generate.py`: Runs the model to generate outputs.
  - `evaluate.py`: Evaluates the generated outputs.

We support all models that are compatible with VLLM.

#### Example: Generating Outputs

To generate the code generations:

```bash
python generate.py --n_samples $n_samples --temperature $temperature --model $model --save_path $save_path
```

This command will create a `.jsonl` file with the generated outputs.

**Complete Example**: Generating with `meta-llama/Llama-3.2-1B-Instruct`, using `VLLM` as the backend on a GPU (with enough memory)  using 1 sample and a temperature of 0 (greedy):

```bash
python generate.py --n_samples 1 --temperature 0 --greedy --model meta-llama/Llama-3.2-1B-Instruct --save_path generations/Llama-3.2-1B-Instruct_T=0.jsonl
```

#### Example: OpenAI-compatible serving

To generate code generations with an OpenAI-compatible server, run the following command replacing with your model and token.
```
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```
To call the server, you can use the official OpenAI Python client library, or any other HTTP client (see https://docs.vllm.ai/en/v0.6.1/serving/openai_compatible_server.html and https://docs.vllm.ai/en/v0.6.1/serving/distributed_serving.html for multi-GPU serving).


#### Example: Running Evaluations

For standard evaluation:

```bash
python evaluate.py --json-out-file $json_outputs --output-path $out_dir --model-name $model_name --temperature $temperature
```

**Parameter Descriptions**:

- `--model-name`: Name of the model used.
- `--json-out-file`: Path to the generated outputs (e.g., `generations/starcoder2-15b-instruct-v0.1_temperature0.0.jsonl`).
- `--output-path`: Directory to save the evaluation results.
- `--n-jobs`: Number of parallel evaluation jobs (`-1` uses all available CPUs).

**Finishing the Example**:

```bash
python evaluate.py --json-out-file generations/Llama-3.2-1B-Instruct_T=0.jsonl --model-name meta-llama/Llama-3.2-1B-Instruct --temperature 0.0
```

**Full test**:
```bash
bash tests/test_readme.sh
```
This will test the given README example to ensure that everything works as intended.

To test url serving:
```bash
bash tests/test_url.sh
```

### To-Do Items

- Specify the number of CPUs used in generation.

### Supported Backends

Currently supported backend:

- `vllm`
- `url-serving (openai-compatible)`

Planned support:

- `hf`, `openai`, `mistral`, `anthropic`, `google`


# Docker
To build the Docker image, run `make docker-build`. 

To open an interactive shell in a Docker container with a specific Python version, run `make docker-run PYTHON_VERSION={desired version}`. 
The following Python versions are configured to work: 3.7, 3.9, 3.10. The local working dir is mounted into the container in the dir `/app/repo`.

# Converting CSV to JSONL
To convert a list of CSV file to JSONL format, use the script `csv2jsonl.py`. Example usage:
```
python csv2jsonl.py file1.csv file2.csv -o merged.jsonl
```



# Run the env creation in the docker image

```
make docker-build
make docker-run
```
then you can run the eval env creation
```
cd repo 
pyenv shell 3.10.14
python -m venv eval_main_venv
source eval_main_venv/bin/activate
pip install vllm -r requirements.txt

# create the virtual env
python src/create_venvs.py --dataset dataset/absolute_final_dataset.jsonl --base_path eval_venvs
```
## run verify dataset using pytest
To run dataset verification with pytest, eval_venvs must have been created first 
```
python verify_dataset.py dataset/fix_dataset.jsonl eval_venvs dataset/samples/ 
```
