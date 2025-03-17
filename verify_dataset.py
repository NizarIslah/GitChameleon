# For each model, we need prompt style
import pandas as pd
import subprocess
from tqdm import tqdm
errors = []

#load cache 
# cache = pd.DataFrame(columns=pd.read_csv("data/numpy_new_format.csv").columns.tolist() + ["SUCCESS"]) # first time running this with no cache
cache = pd.read_csv("cache.csv").fillna("-").astype('str')
dataset = pd.read_csv("data/numpy_new_format.csv").fillna("-").astype('str')
i = 0
for example in tqdm(dataset.iloc):
    # Read the arguments from the CSV and convert them to a list of strings
    args = example.to_list()
    i=i+1
    line_was_cached = False
    for line in cache.iloc:
        if line.to_list()[:-1] == args: # FYI: Only succesful examples are in cache
            print(i , "FOUND IN CACHE")
            line_was_cached = True 
            break

    # Run the subprocess with the converted arguments
    if not line_was_cached:

        result = subprocess.run(['bash', 'verify_dataset.sh', *args], capture_output=True, text=True)
        
        if  ("THIS WAS THE EXIT CODE: 0") in result.stdout:
            print(i , "SUCCESSFULLY VERIFIED EXAMPLE")
            cache.loc[len(cache)+1] = args + ["1"] #1 for success bool   

        elif ("THIS WAS THE EXIT CODE: 1") in result.stdout:
            print(i , "FAILED TO VERIFY EXAMPLE")
            bash_outputs = result.stdout + "\n" + result.stderr + "\nReturn code:" + str(result.returncode)
            errors.append(args + [bash_outputs])
        else:
            print(i , "UNKNOWN ERROR")
            bash_outputs = result.stdout + "\n" + result.stderr + "\nReturn code:" + str(result.returncode)
            errors.append(args.append("UNKNOWN ERROR:" + bash_outputs))


    df = pd.DataFrame(errors, columns=pd.read_csv("data/numpy_new_format.csv").columns.tolist() + ["bash_outputs"])
    df.to_csv("errors.csv", index=False)
    cache.to_csv("cache.csv", index=False)

