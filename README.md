# GitChameleon: A Benchmark for Version-Conditioned Code Generation

Benchmark associated with the paper "GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models", which can be found here: {TBD}

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
     pip install -r requirements.txt
     ```

2. **Prepare Virtual Environments for Evaluation**:

   - Run the following script to populate the `eval_envs/` directory with the necessary Python libraries:
     ```
     python create_venvs.py
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

**Complete Example**: Generating with `bigcode/starcoder2-15b-instruct-v0.1`, using `VLLM` as the backend on a GPU (with enough memory)  using 5 samples and a temperature of 0.8:

```bash
python generate.py --n_samples 5 --temperature 0.8 --model bigcode/starcoder2-15b-instruct-v0.1 --save_path generations/Starcoder2-instruct-v0.1_temperature0.8.jsonl
```

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
python evaluate.py --json-out-file generations/Starcoder2-instruct-v0.1_temperature0.8.jsonl --output-path results/starcoder2-15b-instruct-v0.1_temperature0.8.csv --model-name bigcode/starcoder2-15b-instruct-v0.1 --temperature 0.8
```

### To-Do Items

- Specify the number of CPUs used in generation.

### Supported Backends

Currently supported backend:

- `vllm`

Planned support:

- `hf`, `openai`, `mistral`, `anthropic`, `google`


