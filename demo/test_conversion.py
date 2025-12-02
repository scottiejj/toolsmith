import os
from openai import OpenAI
import sys

# ...existing code...
PROJECT_ROOT = "/Users/scottiejj/Desktop/AutoKaggle_APAPTED"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from multi_agents.prompts.prompt_toolsmith import PROMPT_TOOLSMITH_MARKDOWN_FORMAT

def read_api_key() -> str:
    key_path = os.path.join(PROJECT_ROOT, "api_key.txt")
    with open(key_path, "r") as f:
        api_config = f.readlines()
        api_key = api_config[0].strip()
        return api_key

def read_python_source() -> str:
    path = os.path.join(PROJECT_ROOT, "demo", "example_functions.py")
    with open(path, "r") as f:
        return f.read()

def build_prompt(python_source: str) -> str:
    return (
        "You will receive a Python file containing multiple tool functions. "
        "Convert ALL tools into one concatenated markdown document for this phase, "
        "strictly following the required markdown format below. Preserve formatting exactly.\n\n"
        + PROMPT_TOOLSMITH_MARKDOWN_FORMAT
        + "\n\nPYTHON_SOURCE:\n"
        + python_source
        + "\n\nOutput ONLY the merged markdown text, no extra prose."
    )

def main():
    api_key = read_api_key()
    client = OpenAI(api_key=api_key)

    python_src = read_python_source()
    prompt = build_prompt(python_src)

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs only markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    with open(os.path.join(PROJECT_ROOT, "demo", "converted_tools.md"), "w") as f:
        f.write(resp.choices[0].message.content)

if __name__ == "__main__":
    main()