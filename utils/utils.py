import json
import os
import re
import textwrap


def strip_fence(code: str) -> str:
    code = code.strip()
    pattern = re.compile(
        r'^```(?:python|py|cuda|cu|c|cpp)?\n(.*?)```$',
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

def extract_error_report(text: str):
    m = re.search(r'\[ERROR_REPORT\]\s*\n({.*?})\n', text, flags=re.S)
    return json.loads(m.group(1)) if m else None

def _last_n_lines(text: str, n: int = 150) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text

_INVOCATION_SPLITTER = "Invoked with:"
def _sanitize_error_message(exc: Exception) -> str:
    """Strip pybind's large‑tensor printouts and keep only the key error text."""
    msg = str(exc)
    if _INVOCATION_SPLITTER in msg:
        msg = msg.split(_INVOCATION_SPLITTER, 1)[0].rstrip()
    return msg

def dict_to_text(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, str) and "\n" in v:
                lines.append(f"{prefix}{k}:")
                lines.append(textwrap.indent(v.rstrip(), prefix + "  "))
            else:
                lines.append(f"{prefix}{k}: {dict_to_text(v, indent + 1).lstrip()}")
        return "\n".join(lines)
    if isinstance(obj, list):
        return "\n".join(f"{prefix}- {dict_to_text(item, indent + 1).lstrip()}" for item in obj)
    if isinstance(obj, str):
        return obj.replace('\\n', '\n')
    return str(obj)