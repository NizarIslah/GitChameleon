# GitChameleonEval
GitChameleon: A Benchmark for version-conditioned code generation
## Downloading the dataset
The dataset is in csv format located in ```data/combined_dataset.csv```. These are the examples used in our benchmark.
## Running evals:
1st step: Run```python create_venvs.py <BASE_PATH>```, modifying the BASE_PATH to your scratch folder. This prepares all the library package versions needed to do evaluation with code execution criteria. Then, you are ready to run the main script.

The main script for running generations and evaluations are ```generate.py``` and ```evaluate.py```, respectively.
We support all models that are supported by vllm.
Example generation, to generate 100 outputs per dataset sample with temperature sampling 0.8:
```
python generate.py --n_samples 100 --temperature 0.8 --model $model --data_path $data_path --save_path $save_path
```
Then you will get a .jsonl file containing the generated outputs.

For standard evaluation:
```
python evaluate.py  --evaluate-mode --json_out_file $json_outputs --data-path=$data_path --output-path=$out_dir --model-name=$model_name --temperature=$temperature
```
```data_path```: path to the dataset (csv)

```json_out_file```: path to generated outputs (result of ```python generate.py```)

```evaluate-mode```: means you only run eval metrics (eg. pass @ k), not model inference. You must already have generated outputs from your model formatted correctly (see further instructions on how to format)

```output-path```: path to save eval results

```model-name```: name of your model
