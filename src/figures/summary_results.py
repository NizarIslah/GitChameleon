# function to plot summary results. by solution method (mean and max for each)
# use all_eval_data/ 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib import rc
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
# ignore pandas warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

SOLUTION_METHODS = [
    "CoT"
    "Greedy Decoding",
    "Model (OSS)",
    "Agent",
    "Code Assistant",
    "RAG"
]

def assign_solution_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns solution method based on the 'solution_method' column in the DataFrame.
    """
    # Define a mapping from solution method to its corresponding label
    solution_method_mapping = {
        "cot": "Zero-Shot CoT",
        "o1": "Greedy Decoding",
        "o3": "Greedy Decoding",
        "o4": "Greedy Decoding",
        "agent_results": "Agent",
        "gpt": "Greedy Decoding",
        "gemini": "Greedy Decoding",
        "llama": "Model (OSS)",
        "claude_37": "Greedy Decoding",
        "Qwen": "Model (OSS)",
        "grok": "Greedy Decoding",
        "mistral": "Greedy Decoding",
        "DDG": "Agent",
        "goose": "Code Assistant",
        "roocode": "Code Assistant",
        "kilocode": "Code Assistant",
        "cline": "Code Assistant",
        "claude_code": "Code Assistant",
    }
    
    # Map the solution methods in the DataFrame
    def map_file_solution_method(file_name):
        for key in solution_method_mapping.keys():
            if key in file_name:
                return solution_method_mapping[key]
        return "Unknown"
    df['solution_method'] = df['file_name'].apply(map_file_solution_method)
    
    return df

def pull_results(paths: list) -> pd.DataFrame:
    """
    Merges multiple DataFrames into a single DataFrame.
    """
    # Concatenate all DataFrames in the list
    dfs = []
    for path in paths:
        # Read the CSV file
        if path.endswith(".jsonl"):
            # read jsonl file
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_csv(path)
        
        # Extract the file name from the path
        file_name = os.path.basename(path)
        
        # Add a new column with the file name
        # only pased column
        # print(df['passed'].dtypes)
        df = df.filter(items=['passed'])
        # convert to "False" to 0 and "True" to 1
        if path.endswith(".jsonl"):
            df = df.map(lambda x: 1 if x == "True" else 0)
        else:
            df = df.map(lambda x: 1 if x else 0)
        # mean across rows
        mean_df = df.mean(axis=0)
        # only 'passed' column
        # Convert the Series to a DataFrame
        mean_df = mean_df.reset_index()
        mean_df.columns = ['index', 'mean']
        mean_df['file_name'] = file_name
        mean_df['solution_method'] = None
        mean_df = assign_solution_method(mean_df)
        # Append the DataFrame to the list
        dfs.append(mean_df)
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Drop duplicates
    merged_df = merged_df.drop_duplicates()

    # remove unkonwn category
    merged_df = merged_df[merged_df['solution_method'] != "Unknown"]
    
    return merged_df

def plot_summary_results(merged_df: pd.DataFrame):
    # mean and max for each solution method
    mean_df = merged_df.groupby('solution_method')['mean'].mean().reset_index()
    max_df = merged_df.groupby('solution_method')['mean'].max().reset_index()
    # Merge the mean and max DataFrames
    summary_df = pd.merge(mean_df, max_df, on='solution_method', suffixes=('_mean', '_max'))
    # Sort the DataFrame by mean values
    summary_df = summary_df.sort_values(by='mean_mean', ascending=False)
    # Set the figure size
    fig = plt.figure(figsize=(10, 6))
    # Create a double bar plot
    bar_width = 0.35
    x = np.arange(len(summary_df))
    # Plot mean values
    plt.bar(x, summary_df['mean_mean'], width=bar_width, label='Mean', color='#984ea3')
    # Plot max values
    plt.bar(x + bar_width, summary_df['mean_max'], width=bar_width, label='Max', color='coral')
    # Set the x-ticks and x-tick labels
    plt.xticks(x + bar_width / 2, summary_df['solution_method'], rotation=45)
    # Set the y-axis to show integer values
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # Set the y-axis limits
    plt.ylim(0, 1)
    # Set the grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Set the font size
    rcParams.update({'font.size': 12})
    plt.xlabel('Solution Method', fontsize=20)
    plt.ylabel('Success Rate', fontsize=20)
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    # add yticks
    plt.yticks(np.arange(0.2, 0.8, 0.1))
    plt.ylim(0.2, 0.8)
    # Add a legend
    plt.legend()
    # Show the plot
    plt.tight_layout()
    # increase fontsize on x and y
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    fig.savefig("summary_results.pdf", bbox_inches='tight')

if __name__ == "__main__":
    # Define the paths to the CSV files
    dirname = "all_eval_data"
    self_debug_dirname = "self_debug_data"
    rag_dirname = "rag_results_answers"
    paths = [
        os.path.join(dirname, file)
        for file in os.listdir(dirname)
        if file.endswith(".csv")
    ]
    # add self_debug_data (-r )
    self_debug_paths = []
    for dirname in os.listdir(self_debug_dirname):
        print(dirname)
        self_debug_paths += [
            os.path.join(self_debug_dirname, dirname, file)
            for file in os.listdir(os.path.join(self_debug_dirname, dirname))
            if file.endswith(".jsonl")
        ]
    # print("self_debug_paths", self_debug_paths)

    rag_paths = [
        os.path.join(rag_dirname, file)
        for file in os.listdir(rag_dirname)
        if file.endswith(".jsonl")
    ]
    # print("rag_paths", rag_paths)

    # Pull results from the CSV files
    merged_df = pull_results(paths)
    merged_df_debug = pull_results(self_debug_paths)
    merged_df_rag = pull_results(rag_paths)
    # add self_debug to merged_df
    merged_df_debug['solution_method'] = "Self Debug"
    merged_df_rag['solution_method'] = "RAG"
    # print(merged_df_rag)
    # merge the two
    merged_df = pd.concat([merged_df, merged_df_debug, merged_df_rag], ignore_index=True)
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv("merged_results.csv", index=False)

    plot_summary_results(merged_df)
