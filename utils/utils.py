import ast
import difflib
import json
import os
import re
import shutil
import textwrap
from typing import Any, Dict, List
from pathlib import Path


def find_best_match(kernel_name, cu_files):
    base = kernel_name.replace("_kernel", "")
    names = [f.replace(".cu", "") for f in cu_files]

    match = difflib.get_close_matches(base, names, n=1, cutoff=0.1)

    if match:
        return match[0] + ".cu"
    return None


def sanitize_torch_error(text: str) -> str:
    marker = "Invoked with:"
    pos = text.find(marker)
    if pos == -1:
        return text

    prefix = text[:pos + len(marker)]
    args_part = text[pos + len(marker):]


    args_part = " ".join(args_part.split())

    args = []
    current = []
    depth = 0

    for c in args_part:
        if c == ',' and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue

        current.append(c)

        if c in "([": 
            depth += 1
        elif c in ")]":
            depth -= 1

    if current:
        args.append("".join(current).strip())

    def compress_tensor(arg: str):
        if not arg.startswith("tensor("):
            return arg

        m = re.search(r"\[(.*)\]", arg)
        if not m:
            return "tensor(?)"

        content = m.group(0)

        dim = 0
        for c in content:
            if c == '[':
                dim += 1
            else:
                break

        return f"tensor({dim}D)"

    args = [compress_tensor(a) for a in args]

    return prefix + " " + ", ".join(args)




def load_related_files(related_files, project_root):
    """
    Load file contents for a list of related files.

    Parameters
    ----------
    related_files : list[str]
        List of file paths relative to project_root.
    project_root : str
        Root directory of the project.

    Returns
    -------
    dict
        {
            "file_name": "<file content>",
            ...
        }
    """
    file_contents = {}
    seen = set()

    for rel_path in related_files:
        if rel_path in seen:
            continue
        seen.add(rel_path)

        abs_path = os.path.join(project_root, rel_path)

        if not os.path.exists(abs_path):
            print(f"[Warning] File not found: {abs_path}")
            continue

        if not os.path.isfile(abs_path):
            print(f"[Warning] Not a file: {abs_path}")
            continue

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            file_contents[rel_path] = content
        except Exception as e:
            print(f"[Error] Failed to read {abs_path}: {e}")

    return file_contents


def load_show_files(show_files):
    """
    Args:
        show_files (list[dict]): 
            Example:
            [
                {"file_path": "kernel/avgpool3d_plus_gelu_epilogue.cu"},
                {"file_path": "entry.py"}
            ]

    Returns:
        dict[str, str]: 
            {
                "avgpool3d_plus_gelu_epilogue.cu": "<file content>",
                "entry.py": "<file content>"
            }
    """
    result = {}

    for item in show_files:
        file_path = item.get("file_path")
        if not file_path:
            continue

        path_obj = Path(file_path)

        try:
            content = path_obj.read_text(encoding="utf-8")
        except Exception as e:
            content = f"<<ERROR READING FILE: {e}>>"

        result[path_obj.name] = content

    return result

def list_all_files(root_dir: Path) -> List[str]:

    if not root_dir.exists():
        raise FileNotFoundError(f"{root_dir} does not exist")
    files = []
    for f in root_dir.rglob("*"):
        if f.is_file():
            files.append(str(f.relative_to(root_dir)))
    return sorted(files)


def copy_folder(src_folder, dst_folder):
    """
    Copy entire folder (including all subfolders and files) to specified location
    
    Args:
        src_folder: Source folder path
        dst_folder: Destination folder path (created if not exists, merged if exists)
    
    Returns:
        True if success, False if failed
    """
    try:
        if os.path.exists(dst_folder):
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        else:
            shutil.copytree(src_folder, dst_folder)
        #print(f"Copy success: {src_folder} -> {dst_folder}")
        return True
    except Exception as e:
        #print(f"Copy failed: {e}")
        return False


def delete_folder(folder_path, delete_self=True):
    """
    Delete folder and all its contents
    
    Args:
        folder_path: Path to folder to delete
        delete_self: Whether to delete folder itself (default True deletes entire folder,
                    set False to only clear contents while keeping empty folder)
    
    Returns:
        True if success, False if failed
    """
    try:
        if not os.path.exists(folder_path):
            print(f"Path not exist: {folder_path}")
            return False
        
        if delete_self:
            shutil.rmtree(folder_path)
            #print(f"Deleted folder: {folder_path}")
        else:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            #print(f"Cleared folder contents: {folder_path}")
        
        return True
    except Exception as e:
        #print(f"Delete failed: {e}")
        return False



import os
import re

def save_cuda_files_clean(api_output: str, output_dir: str = "./cuda_kernels"):
    os.makedirs(output_dir, exist_ok=True)

    clean_output = re.sub(r'```[a-z]*\n', '', api_output, flags=re.IGNORECASE)
    clean_output = re.sub(r'```', '', clean_output)
    clean_output = clean_output.strip()

    file_pattern = re.compile(r'^//.*?([\w\-_]+\.cu)', re.IGNORECASE | re.MULTILINE)
    matches = list(file_pattern.finditer(clean_output))

    saved_files = []

    if not matches:
        print("No CUDA files found in the API output.")
        return saved_files

    for i, match in enumerate(matches):
        filename = match.group(1).strip()
        start_idx = match.end() 
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(clean_output)
        content = clean_output[start_idx:end_idx].strip()

        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content + '\n')
        saved_files.append(filename)

    return saved_files
def remove_justification(data: Any) -> Any:

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data  
    
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "justification":
                continue  
            result[key] = remove_justification(value)
        return result
    
    elif isinstance(data, list):
        return [remove_justification(item) for item in data]
    
    else:
        return data
    
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
    text_stripped = text.strip()
    
    # Unwrap Python string literal if present
    if (text_stripped.startswith("'") and text_stripped.endswith("'")) or \
       (text_stripped.startswith('"') and text_stripped.endswith('"')):
        try:
            text = ast.literal_eval(text_stripped)
        except (ValueError, SyntaxError):
            pass  
    
    # If it's already a JSON object string, parse directly
    text_stripped = text.strip()
    if text_stripped.startswith('{') and text_stripped.endswith('}'):
        try:
            return json.loads(text_stripped)
        except json.JSONDecodeError:
            pass
    
    # Try code blocks
    patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    
    # Fallback: extract first balanced braces (ignoring string contents is hard, 
    # so we just try parsing from first '{' to last '}')
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except json.JSONDecodeError:
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