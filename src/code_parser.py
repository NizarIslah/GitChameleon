import re

def extract_first_python_code_block(text) -> str:
    """
    Extracts the first Python code block from the given text.
    Args:
        text (str): The input text containing code blocks.
    Returns:
        str: The extracted Python code block, or None if no code block is found.
    """
    try:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
    except Exception as e:
        try:
            match = re.search(r"```(.*?)```", rf"{text}", re.DOTALL)  # anthropic
        except Exception as e:
            print("Error: ", e)
            match = None
    return match.group(1) if match else None
