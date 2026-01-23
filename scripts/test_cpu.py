
from collections import defaultdict
import contextlib

import hashlib
import io
import os
import re
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn
import importlib.util 

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from multiprocessing import get_context

from scripts.test_kernel import _capture_import, _run_once, try_align_params
from utils.utils import _last_n_lines, _sanitize_error_message
from utils.metrics import fastp

def test_cpu(root_dir: Path, task_dir: Path, device_idx: int = 0):
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=test_kernel_cpu, args=(root_dir, task_dir, device_idx, child_conn,))
    p.start()
    try:
        child_conn.close()
    except Exception:
        pass
    p.join()
    payload = parent_conn.recv() if parent_conn.poll() else None
    try:
        parent_conn.close()
    except Exception:
        pass   
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
        else:
            metrics = {"runnable": False,"message": data}
    else:
        metrics = {"runnable": False}
    return metrics

def test_kernel_cpu(root_dir: Path, task_dir: Path, device_idx: int = 0, conn = None):
    try:
        res = _test_cpu_process(root_dir, task_dir, device_idx, conn)
        conn.send(("ok", res))
    except Exception as e:
                conn.send((
            "err",
            {
                "type": "compilation_error",
                "message": str(e),   
            }
        ))
    except ValueError as e:
        conn.send((
            "err",
            {
                "type": "value_error",
                "message": str(e),
            }
        ))
    except Exception as e:
        conn.send((
            "err",
            {
                "type": "runtime_error",
                "message": str(e),
            }
        ))

def _test_cpu_process(root_dir: Path, task_dir: Path, device_idx: int = 0,conn = None):
    dev = torch.device(f"cuda:{device_idx}")
    ref_mod , _ = _capture_import(root_dir / "spec" / "ref.py")
    test_mod , _ = _capture_import(root_dir / "spec" / "entry.py")

    RefModel = getattr(ref_mod, "Model", None)
    get_inputs = getattr(ref_mod, "get_inputs", None)
    ModelNew = getattr(test_mod, "ModelNew", None)

    if None in (RefModel, get_inputs):
        raise RuntimeError(f"Reference '{root_dir / 'spec' / 'ref.py'}' must define Model and get_inputs().")
    if ModelNew is None:
        raise RuntimeError(f"Candidate '{root_dir / 'spec' / 'entry.py'}' must define class ModelNew.")
    init_args: List[Any] = []
    init_kwargs: Dict[str, Any] = {}

    get_init_inputs_ref = getattr(ref_mod, "get_init_inputs", None)

    if callable(get_init_inputs_ref):
        init_obj = get_init_inputs_ref()
        if isinstance(init_obj, dict):
            init_kwargs = dict(init_obj)
        elif isinstance(init_obj, (list, tuple)):
            init_args = list(init_obj)
        elif init_obj is not None:
            raise TypeError("get_init_inputs() must return list/tuple (as *args) or dict (as **kwargs).")
    def _first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, torch.Tensor):
                    return t
        raise TypeError("Model forward did not return a Tensor (or a sequence containing a Tensor).")

    try:
        ctx = torch.cuda.device(0)
        with ctx:
            # Fix input randomness
            inp = get_inputs()
            if not isinstance(inp, (list, tuple)):
                inp = [inp]

            # Fix parameter initialization: set seed before constructing each side
            ref_model  = RefModel(*init_args, **init_kwargs)

            test_model = ModelNew(*init_args, **init_kwargs)

            # Parameter alignment (prefer Model→ModelNew pair-specific, then task custom, finally generic)
            align_stats = try_align_params(ref_model, test_model, ref_mod=ref_mod, test_mod=test_mod)

            # Forward pass (sync to surface errors immediately)

            ref_out,  _ = _run_once(ref_model,  inp, dev)
            test_out, _ = _run_once_cpu(test_model, inp)

            ref_out = ref_out.cpu()

            # Normalize to Tensor and ensure contiguous
            ref_out  = _first_tensor(ref_out).contiguous()
            test_out = _first_tensor(test_out).contiguous()
            if ref_out.dtype != test_out.dtype:
                test_out = test_out.to(ref_out.dtype)

            # Error & allclose
            diff = (test_out - ref_out).abs()
            max_err  = diff.max().item()
            mean_err = diff.mean().item()

            if not torch.allclose(ref_out, test_out, atol=1e-4, rtol=1e-4):
                raise ValueError(
                    f"Outputs are not close (atol={1e-4}, rtol={1e-4}). "
                    f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}"
                )

    except ValueError:
        raise
    except Exception:
        import traceback as _tb
        raise RuntimeError(_tb.format_exc()) from None   
    
def _run_once_cpu(model: nn.Module, inputs: List[Any]) -> Tuple[Any, float]:
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs)
    return outputs, 0.0