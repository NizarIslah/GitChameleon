import pandas as pd
import os
import subprocess
import hashlib
import sys
from tqdm import tqdm
from pathlib import Path


def replace_lib_version(row):
    """Replace specific versions of library with a compatible
    version for the evaluation environment."""
    if row["library"] == "torch":
        if str(row["version"]) in ("1.9", "1.9.0", "1.1", "1.10", "1.10.0"):
            return "1.11.0"
    if row["library"] == "scipy":
        if str(row["version"]) in ("1.10.2"):
            return "1.10.1"
    return row["version"]


def create_virtual_environment(env_path, create_anyway=False, library_to_check=None):
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
    
    if library_to_check:
        # Determine the correct path to the Python executable in the venv
        python_exec = os.path.join(env_path, "bin", "python")
        
        # Use pip show to check for the library.
        result = subprocess.run(
            [python_exec, "-m", "pip", "show", library_to_check],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode == 0:
            print(f"Library '{library_to_check}' is installed in the virtual environment.")
        else:
            print(f"Library '{library_to_check}' is NOT installed in the virtual environment.")
    
    return env_path


def install_packages(env_path, library, version, additional_dependencies):
    """Install packages using the Python executable in the virtual environment."""
    python_executable = Path(env_path, "bin", "python")  # For Unix-like OS
    # Construct the pip install command using the specific Python executable
    if library == 'librosa':
        if "pip" in version:
            version = "0.8.0"
            additional_dependencies.replace("joblib==0.12", "joblib")
        if "July 21" in additional_dependencies:
            additional_dependencies = "numba==0.46 llvmlite==0.30 joblib numpy==1.16.0 audioread==2.1.5 scipy==1.1.0 resampy==0.2.2 soundfile"
        print("Adding dependencies six decorator cffi")
        additional_dependencies += " six decorator cffi" # hardcoded for librosa
    additional_dependencies = additional_dependencies.replace("pip==24.1", "pip==24.0")
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

    # upgrade pip first
    pip_upgrade_cmd = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "--quiet",
    ]
    result = subprocess.run(
        pip_upgrade_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
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

    df = pd.read_csv(args.dataset)

    # replace torch version
    df["version"] = df.apply(replace_lib_version, axis=1)
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

            create_virtual_environment(env_path, create_anyway=create_anyway, library_to_check=row["library"])
            returncode = install_packages(
                env_path, row["library"], row["version"], row["additional_dependencies"]
            )
            if returncode != 0:
                failed_count.append(row_idx)
            else:
                print(f"All good for {row_idx}")
        else:
            # check install packages anyway
            if row["library"] == "librosa":
            returncode = install_packages(
                env_path, row["library"], row["version"], row["additional_dependencies"]
            )
            if returncode != 0:
                failed_count.append(row_idx)
            else:
                print(f"All good for {row_idx}")

    print(f"Failed: {len(failed_count)}")
    for idx in failed_count:
        print(
            f"idx {idx} \t Library: {df.loc[idx, 'library']}, Version: {df.loc[idx, 'version']}, Additional Dependencies: {df.loc[idx, 'additional_dependencies']}"
        )


if __name__ == "__main__":
    import argparse

    # argument for env_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/all_samples_final.csv")
    parser.add_argument("--base_path", type=str, default="/network/scratch/n/nizar.islah/eval_venvs/")
    parser.add_argument("--create_anyway", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
