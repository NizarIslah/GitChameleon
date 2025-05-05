import os
import subprocess
import tempfile


def eval_sample(
    example_id: int, env_path, code_dict: dict, strategy="pytest", coverage=False
) -> dict:
    """
    Evaluate sample code using the specified strategy in the provided virtual environment.

    This function evaluates one or several code samples against a single test file.
    Each sample can contain several code entries in the "codes" dictionary and will be
    evaluated using the same test file provided under the "test_file" key.

    The input dictionary must have the following structure:
        {
            "test_file": <str>,    # The content of the test file containing pytest tests.
            "codes": {
                "code_id1": { "code": <str> },
                "code_id2": { "code": <str> },
                ...
            }
        }

    Args:
        example_id (int): An identifier for the evaluation sample.
        env_path (str): Path to the virtual environment to use for evaluation.
        code_dict (dict): A dictionary containing both the test file content and the code samples.
        strategy (str): Evaluation strategy to use (default is 'pytest'). Currently, only 'pytest' is supported.

    Returns:
        dict: A dictionary containing the evaluation results with the following structure:
            {
                "test_file": <str>,
                "codes": {
                    "code_id1": {
                        "code": <str>,
                        "output": <str>,   # Combined stdout and stderr from running the tests.
                        "pass": <bool>,    # True if tests passed (zero return code), otherwise False.
                        "compile": <bool>  # True if the code compiled successfully; default is True.
                    },
                    "code_id2": { ... },
                    ...
                }
            }

    Example:
        env_path = "eval_venvs_debug/gcham_venv_0"
        code_dict = {
            "test_file": "def test_example():\n    assert 'Hello' in open('sample_code.py').read()",
            "codes": {
                "example_id_0_greedy": {
                    "code": "print('Hello, World!')"
                },
                "example_id_1_greedy": {
                    "code": "print('Goodbye, World!')"
                }
            }
        }
        results = eval_sample(0, env_path, code_dict, strategy='pytest')
    """
    results = {"test_file": code_dict.get("test_file", ""), "codes": {}}
    test_file_content = code_dict.get("test_file", "")
    codes = code_dict.get("codes", {})

    for code_id, content in codes.items():
        code = content.get("code", "")
        sample_result = {"code": code, "output": "", "pass": False, "compile": True}

        if strategy.lower() == "pytest":
            # Create a temporary directory to host the sample code and the test file
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write the sample code to a file
                code_filepath = os.path.join(temp_dir, f"sample_{example_id}.py")
                with open(code_filepath, "w") as f_code:
                    f_code.write(code)

                # Write the common pytest test file
                test_filepath = os.path.join(temp_dir, "test_sample.py")
                with open(test_filepath, "w") as f_test:
                    f_test.write(test_file_content)

                # Construct the python executable path from the virtual environment
                python_executable = os.path.join(env_path, "bin", "python")
                # Build the pytest command; using -q for quiet output, stopping at the first failure
                cmd = [
                    python_executable,
                    "-m",
                    "pytest",
                    "--disable-warnings",
                    "-q",
                    temp_dir,
                ]

                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=120,
                    )
                    sample_result["output"] = proc.stdout + proc.stderr
                    # A return code of 0 indicates that the tests passed.
                    sample_result["pass"] = proc.returncode == 0
                except subprocess.TimeoutExpired as e:
                    print(f"Timeout expired: {e}")
                    sample_result["output"] = f"Timeout: {str(e)}"
                    sample_result["pass"] = False
                except Exception as e:
                    sample_result["output"] = f"Error: {str(e)}"
                    sample_result["pass"] = False

                # get coverage optionally
                if coverage:
                    cov_file = os.path.join(temp_dir, f"coverage_{example_id}.json")
                    # current_dir = os.getcwd()
                    try:
                        # os.chdir(temp_dir)
                        # Run pytest with coverage
                        cmd = [
                            python_executable,
                            "-m",
                            "pytest",
                            f"--cov=sample_{example_id}",
                            f"--cov-report=json:{cov_file}",
                            test_filepath,
                        ]

                        proc = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=120,
                        )
                        # print(proc.stdout)
                        # print(proc.stderr)

                        import json

                        with open(cov_file, "r") as f:
                            coverage_data = json.load(f)
                            sample_result["coverage"] = coverage_data["totals"][
                                "percent_covered"
                            ]
                        # os.chdir(current_dir)

                    except Exception as e:
                        print(f"Error while getting coverage: {e}")
                        # os.chdir(current_dir)
                        pass

        else:
            sample_result["output"] = "Unsupported evaluation strategy."
            sample_result["pass"] = False

        results["codes"][code_id] = sample_result
    return results


if __name__ == "__main__":
    # Example usage
    env_path = "eval_venvs_debug/gcham_venv_0"
    strategy = "pytest"
    test_file_path = "samples/test_early_sample_0/test_early_log_ndtr.py"
    example_id_0_greedy_path = "samples/sample_0.py"

    code_dict = {}
    # Read the test file content
    with open(test_file_path, "r") as test_file:
        test_file_content = test_file.read()
    # Update the code_dict with the test file content
    code_dict["test_file"] = test_file_content
    # Read the sample code content and update the code_dict
    with open(example_id_0_greedy_path, "r") as sample_file:
        sample_code_content = sample_file.read()
    code_dict["codes"] = {"example_id_0_greedy": {"code": sample_code_content}}

    results = eval_sample(0, env_path, code_dict, strategy)
    import pprint

    pprint.pprint(results)
