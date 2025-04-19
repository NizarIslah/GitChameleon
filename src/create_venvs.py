import pandas as pd
import os
import subprocess
import hashlib
import sys
from tqdm import tqdm
from pathlib import Path
import json

# Mapping of Python versions to pyenv-installed versions
python_versions = {"3.7": "3.7.17", "3.9": "3.9.19", "3.10": "3.10.14"}


# Function to create a virtual environment
def create_virtual_environment(
    env_path, python_version, create_anyway=False, library_to_check=None
):
    """Create and return the path of a virtual environment."""
    python_executable = f"/root/.pyenv/versions/{python_version}/bin/python"
    if not os.path.exists(python_executable):
        print(f"Python version {python_version} not found. Skipping {env_path}.")
        return

    if not os.path.exists(env_path):
        os.makedirs(env_path, exist_ok=True)
        subprocess.run([python_executable, "-m", "venv", env_path], check=True)
        print(f"Virtual environment created: {env_path}")
    else:
        print(f"Virtual environment already exists: {env_path}")
        if create_anyway:
            subprocess.run(["rm", "-rf", env_path])
            os.makedirs(env_path, exist_ok=True)
            subprocess.run([python_executable, "-m", "venv", env_path], check=True)
            print(f"Virtual environment recreated: {env_path}")

    if library_to_check:
        python_exec = os.path.join(env_path, "bin", "python")
        result = subprocess.run(
            [python_exec, "-m", "pip", "show", library_to_check],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            print(
                f"Library '{library_to_check}' is installed in the virtual environment."
            )
        else:
            print(
                f"Library '{library_to_check}' is NOT installed in the virtual environment."
            )

    return env_path


def install_packages(
    env_path, library, version, additional_dependencies, python_version
):
    """Install packages using the Python executable in the virtual environment."""
    python_executable = Path(env_path, "bin", "python")

    # Parse additional dependencies
    dependencies = additional_dependencies.split() if additional_dependencies else []
    pip_version = None
    other_dependencies = []

    # Separate pip version if specified
    for dep in dependencies:
        if dep.startswith("pip="):
            pip_version = dep.split("=")[1]
        elif dep.strip() and dep != "-":  # Filter out invalid entries
            other_dependencies.append(dep)

    # Upgrade pip to the specified version or the latest version
    if pip_version:
        pip_upgrade_cmd = [
            python_executable,
            "-m",
            "pip",
            "install",
            f"pip=={pip_version}",
            "--quiet",
        ]
    else:
        pip_upgrade_cmd = [
            python_executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "--quiet",
        ]
    subprocess.run(
        pip_upgrade_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    print(
        f"Pip upgraded in {env_path} to version {pip_version if pip_version else 'latest'}."
    )

    # Install the main library and other dependencies
    pip_install_cmd = [
        python_executable,
        "-m",
        "pip",
        "install",
        f"{library}=={version}",
        "--quiet",
    ] + other_dependencies

    print(f"Installing packages in {env_path}...")
    result = subprocess.run(
        pip_install_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(f"Failed to install packages in {env_path}: {result.stderr}")
        subprocess.run(
            ["rm", "-rf", env_path]
        )  # Clean up the environment if installation fails
    else:
        print(f"Packages installed successfully in {env_path}")

        # Install pytest if it is not the main library or specified in additional_dependencies
        deps_lower = [dep.lower() for dep in dependencies]
        if library.lower() != "pytest" and all(
            "pytest" not in dep for dep in deps_lower
        ):
            # Define pinpointed pytest versions per python version
            pytest_versions = {
                "3.7": "pytest==6.2.5",
                "3.9": "pytest==7.1.2",
                "3.10": "pytest==7.2.0",
            }
            pytest_spec = pytest_versions.get(python_version)
            if pytest_spec:
                pytest_install_cmd = [
                    python_executable,
                    "-m",
                    "pip",
                    "install",
                    pytest_spec,
                    "--quiet",
                ]
                result_pytest = subprocess.run(
                    pytest_install_cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if result_pytest.returncode == 0:
                    print(
                        f"Pytest installed successfully in {env_path} with version from mapping: {pytest_spec}"
                    )
                else:
                    print(
                        f"Failed to install pytest in {env_path}: {result_pytest.stderr}"
                    )
            else:
                print(
                    f"No pytest version mapping found for Python {python_version}. Skipping pytest installation."
                )
    return result.returncode


def generate_env_id(row):
    """Generate a unique ID based on library, version, and dependencies."""
    unique_str = f"{row['library']}-{row['version']}-{row['additional_dependencies']}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[:8]


def main(args):
    jsonl_file = args.dataset
    base_path = args.base_path
    create_anyway = args.create_anyway
    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)

    failed_count = []

    # Read the JSONL file
    with open(jsonl_file, "r") as file:
        for line in file:
            sample = json.loads(line)
            python_version = sample.get("python_version")
            example_id = sample.get("example_id")
            library = sample.get("library")
            version = sample.get("version")
            additional_dependencies = sample.get("additional_dependencies", "")

            if python_version and example_id:
                pyenv_version = python_versions.get(python_version)
                if not pyenv_version:
                    print(
                        f"Unsupported Python version {python_version} for example {example_id}."
                    )
                    continue

                env_name = f"gcham_venv_{example_id}"
                env_path = Path(base_path, env_name)

                python_exec = Path(env_path, "bin", "python")
                if not os.path.exists(python_exec):
                    print(
                        f"Python executable not found for {example_id}. Creating environment..."
                    )
                    create_virtual_environment(
                        env_path,
                        pyenv_version,
                        create_anyway=create_anyway,
                        library_to_check=library,
                    )
                    returncode = install_packages(
                        env_path,
                        library,
                        version,
                        additional_dependencies,
                        python_version,
                    )
                    if returncode != 0:
                        failed_count.append(example_id)
                else:
                    print(f"Environment already exists for {example_id}.")

    print(f"Failed: {len(failed_count)}")
    for example_id in failed_count:
        print(f"Failed to create environment for example ID: {example_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the JSONL dataset file."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="eval_venvs",
        help="Base path for virtual environments.",
    )
    parser.add_argument(
        "--create_anyway",
        action="store_true",
        default=False,
        help="Recreate environments if they already exist.",
    )
    args = parser.parse_args()
    main(args)
