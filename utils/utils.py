import json
import os
import re
import textwrap
from typing import Dict

def text_to_dict(text: str) -> Dict:
    """
    Parse text format log/dict into Python dict.
    Supports formats like:
    - key: value
    - key=value
    - JSON format
    """

    
    result = {}
    text = text.strip()
    
    # Try JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Parse line by line
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Skip separators like "---"
        if set(line) <= set('-=#'):
            continue
        
        # Try key: value format
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to parse value
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                value = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                value = float(value)
            
            result[key] = value
            
        # Try key=value format
        elif '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Parse value
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                value = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                value = float(value)
            
            result[key] = value
    
    return result
def strip_fence(code: str) -> str:
    code = code.strip()
    pattern = re.compile(
        r'^```(?:python|py|cuda|cu|c|cpp|json)?\n(.*?)```$',
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
        with open(file_path, "r", encoding='utf-8') as file:
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
    
def extract_json(text: str) -> dict:
    patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'\[recommendation\]\s*(\{.*?\})\s*\[/recommendation\]',
        r'(\{[\s\S]*?"operators"[\s\S]*?\})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    try:
        start = text.find('{')
        if start == -1:
            return {}
        count = 0
        end = start
        for i, char in enumerate(text[start:], start=start):
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
                if count == 0:
                    end = i + 1
                    break
        
        if count == 0:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    
    return {}
def extract_recommendation(text: str) -> dict:
    m = re.search(r'\[recommendation]\s*(\{.*?\})\s*\[/recommendation]', text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}

def extract_error_report(text: str) -> dict:
    m = re.search(r'\[ERROR_REPORT]\s*(\{.*?\})\s*\[/ERROR_REPORT]', text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}

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