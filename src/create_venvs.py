import pandas as pd
import os
import subprocess
import hashlib
import sys
from tqdm import tqdm
from pathlib import Path

def replace_torch_version(row):
    if row["library"] == "torch":
        if str(row["version"]) in ("1.9", "1.9.0", "1.1", "1.10", "1.10.0"):
            return "1.11.0"
    return row["version"]


def create_virtual_environment(env_path, create_anyway=False):
    """Create and return the path of a virtual environment."""
    if not os.path.exists(env_path):
        os.makedirs(env_path, exist_ok=True)
        subprocess.run(["python", "-m", "venv", env_path], check=True)
        print(f"Virtual environment created: {env_path}")
    else:
        print(f"Virtual environment already exists: {env_path}")
        # remove it and create anywyay
        if create_anyway:
            subprocess.run(["rm", "-rf", env_path])
            os.makedirs(env_path, exist_ok=True)
            subprocess.run(["python", "-m", "venv", env_path], check=True)
            print(f"Virtual environment created: {env_path}")
    return env_path


def install_packages(env_path, library, version, additional_dependencies):
    """Install packages using the Python executable in the virtual environment."""
    python_executable = Path(env_path, "bin", "python")  # For Unix-like OS
    # Construct the pip install command using the specific Python executable
    if str(additional_dependencies) in ("nan", "", "-", "io"):
        pip_install_cmd = [
            python_executable,
            "-m",
            "pip",
            "install",
            f"{library}=={version}",
            "--quiet",
        ]
    else:
        pip_install_cmd = [
            python_executable,
            "-m",
            "pip",
            "install",
            f"{library}=={version}",
            "--quiet",
        ] + additional_dependencies.split()

    # Run the installation command
    print(f"Installing packages in {env_path}...")
    result = subprocess.run(
        pip_install_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(
            f"Failed to install packages in {env_path}: {result.stderr}, tearing down the environment..."
        )
        # remove the virtual environment
        subprocess.run(["rm", "-rf", env_path])
    else:
        print(f"Packages installed successfully in {env_path}")
    return result.returncode


def generate_env_id(row):
    """Generate a unique ID based on library, version, and dependencies."""
    unique_str = f"{row['library']}-{row['version']}-{row['additional_dependencies']}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[
        :8
    ]  # Shorten the hash for readability


def main(args):
    base_path = args.base_path
    create_anyway = args.create_anyway

    df = pd.read_csv("dataset/combined_dataset.csv")

    # replace torch version
    df["version"] = df.apply(replace_torch_version, axis=1)
    # Generate unique environment IDs
    df["env_id"] = df.apply(generate_env_id, axis=1)

    # print num unique envs
    print(f"Number of unique environments: {df['env_id'].nunique()}")

    # Save the updated CSV
    df.to_csv("dataset/updated_libraries.csv", index=False)
    print("Updated CSV with virtual environment IDs saved.")

    # check that the python executable exists for each venv
    failed_count = []
    for row_idx, row in df.drop_duplicates(subset=["env_id"]).iterrows():
        env_name = f"gcham_venv{row['env_id']}"
        env_path = Path(base_path, env_name)
        python_executable = Path(env_path, "bin", "python")
        if not os.path.exists(python_executable):
            print(f"Python executable not found for {row_idx}")
            subprocess.run(["rm", "-rf", env_path])
        create_virtual_environment(env_path, create_anyway=create_anyway)
        returncode = install_packages(
            env_path, row["library"], row["version"], row["additional_dependencies"]
        )
        if returncode != 0:
            failed_count.append(row_idx)
        else:
            print(f"Python executable created for {row_idx}")

    print(f"Failed: {len(failed_count)}")
    for idx in failed_count:
        print(
            f"idx {idx} \t Library: {df.loc[idx, 'library']}, Version: {df.loc[idx, 'version']}, Additional Dependencies: {df.loc[idx, 'additional_dependencies']}"
        )


if __name__ == "__main__":
    import argparse
    # argument for env_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="../eval_venvs/")
    parser.add_argument("--create_anyway", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
