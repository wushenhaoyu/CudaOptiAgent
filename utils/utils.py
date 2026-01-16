import os
import re


def strip_fence(code: str) -> str:
    code = code.strip()
    pattern = re.compile(
        r'^```(?:python|py|cuda|cu)?\n(.*?)```$',
        re.MULTILINE | re.DOTALL
    )
    match = pattern.fullmatch(code)
    if match:
        return match.group(1).strip()
    return code

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""
    
def write_file(file_path: str, content: str, encoding: str = "utf-8") -> bool:
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:  
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        return True

    except Exception as e:
        print(f"Error writing file {file_path}: {e}")
        return False
    
def extract_recommendation(text: str) -> str:
    m = re.search(r'\[recommendation\]\s*(.*?)\s*(?:\[\w+]|$)', text, flags=re.S)
    return m.group(1).strip() if m else "none"