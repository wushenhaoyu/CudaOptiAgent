from collections import defaultdict
import subprocess
import re
import json

from pathlib import Path
from multiprocessing import get_context

from utils.utils import write_file

def parse_log(log_path: str) -> dict:

    text = Path(log_path).read_text(encoding='utf-8', errors='ignore')
    
    errors = []
    
    mem_invalid_pattern = re.compile(
        r'Invalid\s+(__global__|__shared__|__local__|__device__|__constant__)\s+(read|write)\s+of\s+size\s+(\d+)\s+bytes'
        r'(.*?)'
        r'(?=\n=========\s*(?:Invalid|Host Frame|Error|Warning|$)|$)',
        re.DOTALL
    )
    
    for m in mem_invalid_pattern.finditer(text):
        mem_type, op, size, block = m.groups()
        error = {
            'type': 'MEMORY_INVALID',
            'subtype': f"{mem_type} {op}",
            'size_bytes': int(size),
        }
        

        kernel_m = re.search(r'at\s+([\w_]+(?:<[^>]+>)?(?:\([^)]*\))?)\s*[\+\[]?0x[0-9a-f]+', block)
        if kernel_m:
            error['kernel'] = kernel_m.group(1).strip()
 
        thread_m = re.search(r'by thread \((\d+,\d+,\d+)\) in block \((\d+,\d+,\d+)\)', block)
        if thread_m:
            error['thread'] = thread_m.group(1)
            error['block'] = thread_m.group(2)

        addr_m = re.search(r'Address\s+(0x[0-9a-f]+)\s+is out of bounds', block)
        if addr_m:
            error['fault_addr'] = addr_m.group(1)

        alloc_m = re.search(
            r'and is\s+([\d,]+)\s+bytes after the nearest allocation at\s+(0x[0-9a-f]+)\s+of size\s+([\d,]+)\s+bytes',
            block
        )
        if alloc_m:
            error['offset'] = int(alloc_m.group(1).replace(',', ''))
            error['alloc_addr'] = alloc_m.group(2)
            error['alloc_size'] = int(alloc_m.group(3).replace(',', ''))
        
        errors.append(error)

    uninit_pattern = re.compile(
        r'Uninitialized\s+(__global__|__shared__|__local__)\s+memory'
        r'(.*?)'
        r'(?=\n=========\s*(?:Uninitialized|Invalid|Host Frame|$)|$)',
        re.DOTALL
    )
    
    for m in uninit_pattern.finditer(text):
        mem_type, block = m.groups()
        error = {'type': 'MEMORY_UNINIT', 'subtype': f"{mem_type} uninitialized"}
        
        kernel_m = re.search(r'at\s+([\w_]+)', block)
        if kernel_m:
            error['kernel'] = kernel_m.group(1)
        
        thread_m = re.search(r'by thread \((\d+,\d+,\d+)\) in block \((\d+,\d+,\d+)\)', block)
        if thread_m:
            error['thread'] = thread_m.group(1)
            error['block'] = thread_m.group(2)
        
        errors.append(error)
    
    api_error_pattern = re.compile(
        r'Program hit (\w+).*?due to "(.*?)" on CUDA API call to (\w+)',
        re.DOTALL
    )
    for m in api_error_pattern.finditer(text):
        errors.append({
            'type': 'API_ERROR',
            'error_code': m.group(1),
            'message': m.group(2),
            'api_call': m.group(3),
        })
    
    seen = {}
    unique_errors = []
    for e in errors:
        key = (e.get('kernel'), e.get('alloc_addr'), e.get('type'), e.get('subtype'))
        if key not in seen:
            seen[key] = e
            e['count'] = 1
            unique_errors.append(e)
        else:
            seen[key]['count'] += 1
    
    summary = {
        'total_matches': len(errors),
        'unique_errors': len(unique_errors),
        'by_type': defaultdict(int),
        'by_kernel': defaultdict(int),
    }
    
    for e in unique_errors:
        summary['by_type'][e['type']] += 1
        kernel = e.get('kernel', 'N/A')
        if len(str(kernel)) > 50:
            kernel = str(kernel)[:47] + '...'
        summary['by_kernel'][kernel] += 1
    
    return {
        'success': len(unique_errors) == 0,
        'summary': {
            'total_matches': summary['total_matches'],
            'unique_errors': summary['unique_errors'],
            'by_type': dict(summary['by_type']),
            'by_kernel': dict(summary['by_kernel']),
        },
        'errors': unique_errors,
    }




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

