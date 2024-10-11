# GitChameleonEval
GitChameleon: A Benchmark for version-conditioned code generation
## Downloading the dataset
The dataset is in csv format located in ```data/combined_dataset.csv```. These are the examples used in our benchmark.
## Running evals:
1st step: In ``` create_venvs.py ```. Modify the BASE_PATH to your scratch folder and then run it as ```python create_venvs.py```. This prepares all the library package versions needed to do evaluation with code execution criteria. Then, you are ready to run the main script.

The main script for running evaluations is ```fast_model_generate_eval.py```. Here is an example of how to call it:

If you want to evaluate:
```
python3.10 fast_model_generate_eval.py  --evaluate-mode --seed=0 --data-path=$data_path --output-path=$out_dir --model-name=$model_name --temperature=0.3
```
```data_path```: path to the dataset (csv)

```evaluate-mode```: means you only run eval metrics (eg. pass @ k), not model inference. You must already have generated outputs from your model formatted correctly (see further instructions on how to format)

```output-path```: path to save eval results

```model-name```: name of your model