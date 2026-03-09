import subprocess
import re
import json

from pathlib import Path
from multiprocessing import get_context

from utils.utils import write_file

PATTERNS = {

    'mem_invalid': re.compile(
        r'Invalid (__global__|__shared__|__local__|__device__|__constant__) (read|write) of size (\d+) bytes.*?'
        r'at 0x[0-9a-f]+ in (\w+).*?'
        r'by thread \((\d+,\d+,\d+)\) in block \((\d+,\d+,\d+)\).*?'
        r'Address (0x[0-9a-f]+) is out of bounds.*?'
        r'and is ([\d,]+) bytes after the nearest allocation at (0x[0-9a-f]+) of size ([\d,]+) bytes',
        re.DOTALL
    ),

    'mem_uninit': re.compile(
        r'Uninitialized (__global__|__shared__|__local__) memory.*?'
        r'at 0x[0-9a-f]+ in (\w+).*?'
        r'by thread \((\d+,\d+,\d+)\) in block \((\d+,\d+,\d+)\).*?'
        r'Address (0x[0-9a-f]+)',
        re.DOTALL
    ),

    'mem_leak': re.compile(
        r'Memory leak.*?at (0x[0-9a-f]+): (\d+) bytes',
        re.DOTALL
    ),

    'api_error': re.compile(
        r'Program hit (\w+).*?due to "(.*?)" on CUDA API call to (\w+)',
        re.DOTALL
    ),

    'launch_config': re.compile(
        r'Invalid configuration.*?kernel: (\w+).*?config: <<<(\d+), (\d+)>>>',
        re.DOTALL
    ),

    'kernel_source': re.compile(
        r'at ([^\s]+\.cu):(\d+)'
    ),
}

def extract_source_location(text, start_pos):
    window = text[start_pos:start_pos + 400]
    m = PATTERNS['kernel_source'].search(window)
    if m:
        return {"file": m.group(1), "line": int(m.group(2))}
    return None

def parse_log(log_path: str):
    text = Path(log_path).read_text(errors='ignore')
    errors = []

    for m in PATTERNS['mem_invalid'].finditer(text):
        source = extract_source_location(text, m.end())
        errors.append({
            'type': 'MEMORY_INVALID',
            'subtype': f"Invalid {m.group(1)} {m.group(2)}",
            'kernel': m.group(4),
            'thread': m.group(5),
            'block': m.group(6),
            'fault_addr': m.group(7),
            'offset': int(m.group(8).replace(',', '')),
            'alloc_addr': m.group(9),
            'alloc_size': int(m.group(10).replace(',', '')),
            'source': source
        })

    for m in PATTERNS['mem_uninit'].finditer(text):
        source = extract_source_location(text, m.end())
        errors.append({
            'type': 'MEMORY_UNINIT',
            'subtype': f"Uninitialized {m.group(1)} memory",
            'kernel': m.group(2),
            'thread': m.group(3),
            'block': m.group(4),
            'fault_addr': m.group(5),
            'source': source
        })

    for m in PATTERNS['mem_leak'].finditer(text):
        errors.append({
            'type': 'MEMORY_LEAK',
            'alloc_addr': m.group(1),
            'leaked_bytes': int(m.group(2)),
        })

    for m in PATTERNS['api_error'].finditer(text):
        errors.append({
            'type': 'API_ERROR',
            'error_code': m.group(1),
            'message': m.group(2),
            'api_call': m.group(3),
        })

    for m in PATTERNS['launch_config'].finditer(text):
        errors.append({
            'type': 'LAUNCH_CONFIG',
            'kernel': m.group(1),
            'grid': m.group(2),
            'block': m.group(3),
        })

    seen = {}
    unique_errors = []
    for e in errors:
        key = (e.get('type'), e.get('kernel'), e.get('alloc_addr'), e.get('fault_addr'))
        if key not in seen:
            seen[key] = e
            e['count'] = 1
            unique_errors.append(e)
        else:
            seen[key]['count'] += 1

    summary = {'total_matches': len(errors), 'unique_errors': len(unique_errors),
               'by_type': {}, 'by_kernel': {}}
    for e in unique_errors:
        t = e['type']
        k = e.get('kernel', 'N/A')
        summary['by_type'][t] = summary['by_type'].get(t, 0) + 1
        summary['by_kernel'][k] = summary['by_kernel'].get(k, 0) + 1

    return {'success': len(unique_errors) == 0, 'summary': summary, 'errors': unique_errors}



def run_ncs_debug(script_path: Path, device_idx: int = 0, output_log: Path = None):
    from multiprocessing import get_context

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_ncs_process, args=(script_path, device_idx, child_conn, output_log))
    p.start()
    try:
        child_conn.close()
    except Exception:
        pass
    p.join()

    if p.exitcode != 0 and not parent_conn.poll():
        try:
            parent_conn.close()
        except Exception:
            pass
        return {"runnable": False, "error_type": "process_crash",
                "message": f"NCS subprocess exited with code {p.exitcode}"}

    payload = None
    if parent_conn.poll():
        try:
            payload = parent_conn.recv()
        except EOFError:
            payload = None
    try:
        parent_conn.close()
    except Exception:
        pass

    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            result = {"runnable": True}
            result.update(data)
        else:
            result = {"runnable": False}
            result.update(data)
    else:
        result = {"runnable": False, "error_type": "pipe_failure", "message": "No payload from subprocess"}

    return result


def _ncs_process(script_path: Path, device_idx: int, conn, output_log: Path):
    import traceback

    try:
        log_file = output_log if output_log else Path("ncs_output.log")

        cmd = [
            "compute-sanitizer",
            "--tool", "memcheck",
            "--leak-check", "full",
            "--track-origin", "yes",
            "python", str(Path(__file__).parent / "sanitizer_runner.py"),
            str(script_path),
            str(device_idx),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log_file.write_text(result.stdout + "\n" + result.stderr)
        parsed = parse_log(str(log_file))
        write_file(log_file.parent / "ncs_parsed.json", json.dumps(parsed, indent=2))

        conn.send(("ok", {"returncode": result.returncode, "log": str(log_file), "parsed": parsed}))

    except Exception:
        conn.send(("err", {"error_type": "ncs_runner_error", "message": traceback.format_exc()}))

