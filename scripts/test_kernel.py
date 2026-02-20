
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

from utils.utils import _last_n_lines, _sanitize_error_message
from utils.metrics import fastp
'''
come form https://github.com/OptimAI-Lab/CudaForge
'''



def test_kernel(root_dir: Path, task_dir: Path, device_idx: int = 0):
    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=test_kernel_process, args=(root_dir, task_dir, device_idx, child_conn,))
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
            ref  = np.array([metrics["ref_latency_ms"]["avg"]])
            test = np.array([metrics["test_latency_ms"]["avg"]])
            fast1 = fastp(np.array([True]), ref, test, 1, 1)
        else:
            metrics = {"runnable": False,"message": data}
    else:
        metrics = {"runnable": False}
    return metrics
            
def test_kernel_process(root_dir: Path, task_dir: Path, device_idx: int = 0, conn = None):
    try:
        torch.cuda.set_device(device_idx)
        res = _test_kernel_process(root_dir, task_dir, device_idx, conn)
        conn.send(("ok", res))
    except AssertionError as e:
        conn.send((
            "err",
            {
                "type": "parameter_alignment_error",
                "message": str(e),
            }
        ))
    except CompilationError as e:
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
    #except Exception as e:
    #    # Clean the error message if helper is available; otherwise fall back to str(e)
    #    try:
    #        msg = str(e)
    #        # cleaned = _sanitize_error_message(e)
    #        #msg = _last_n_lines(cleaned)
    #    except Exception:
    #        msg = str(e)
    #    conn.send(("err", msg))

def _test_kernel_process(root_dir: Path, task_dir: Path, device_idx: int = 0, conn = None):

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
            torch.cuda.synchronize(dev)
            ref_out,  _ = _run_once(ref_model,  inp, dev)
            test_out, _ = _run_once(test_model, inp, dev)
            torch.cuda.synchronize(dev)
            # Normalize to Tensor and ensure contiguous
            ref_out  = _first_tensor(ref_out).contiguous()
            test_out = _first_tensor(test_out).contiguous()
            if ref_out.dtype != test_out.dtype:
                test_out = test_out.to(ref_out.dtype)

            # Error & allclose
            diff = (test_out - ref_out).abs()
            max_err  = diff.max().item()
            mean_err = diff.mean().item()

            if not torch.allclose(ref_out, test_out, atol=5e-3, rtol=5e-3):
                def _fmt_trunc(t, head=3, tail=3):
                    total = t.numel()
                    if total <= head + tail:
                        return str(t.tolist())
                    h = ", ".join(f"{v:.4f}" for v in t[:head].tolist())
                    e = ", ".join(f"{v:.4f}" for v in t[-tail:].tolist())
                    return f"[{h} ... {e}] (len={total})"
                
                ref_str = _fmt_trunc(ref_out.flatten(), 4, 4)
                test_str = _fmt_trunc(test_out.flatten(), 4, 4)
                
                raise ValueError(
                    f"Outputs are not close (atol={5e-3}, rtol={5e-3}). "
                    f"max_abs_err={max_err:.3e}, mean_abs_err={mean_err:.3e}\n"
                    f"  ref:  {ref_str}\n"
                    f"  test: {test_str}"
                )

            # Timing
            ref_t  = _bench(ref_model,  inp, dev, 2, 5)
            test_t = _bench(test_model, inp, dev, 2, 5)

            torch.cuda.synchronize(dev)

    except ValueError:
        raise
    except Exception:
        import traceback as _tb
        raise RuntimeError(_tb.format_exc()) from None

    # ------------ Aggregate results -------------------------------------
    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "ref_latency_ms": {
            "avg": sum(ref_t) / len(ref_t),
            "min": min(ref_t),
            "max": max(ref_t)
        },
        "test_latency_ms": {
            "avg": sum(test_t) / len(test_t),
            "min": min(test_t),
            "max": max(test_t)
        },
        "model_init_args": init_args,
        "model_init_kwargs": init_kwargs,
        "align_stats": align_stats,  # Alignment summary (incl. whether Model→ModelNew pair-specific aligner was used)
    }
    return result

@torch.no_grad()
def align_params_generic(ref_model: nn.Module, test_model: nn.Module) -> dict[str, int]:
    ref_named = _named_tensors(ref_model)
    test_named = _named_tensors(test_model)

    copied_same, unique_shape_copied, mapped, skipped = 0, 0, 0, 0
    aligned_test: set[str] = set()
    missing_params: list[str] = []  # 新增：记录未对齐的参数

    # 1) Same name & same shape
    for name, t_dst in test_named.items():
        t_src = ref_named.get(name, None)
        if t_src is not None and _safe_copy_(t_dst, t_src):
            copied_same += 1
            aligned_test.add(name)

    # 2) Unique shape match
    shape2ref: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    shape2test: dict[tuple, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for n, t in ref_named.items():
        shape2ref[tuple(t.shape)].append((n, t))
    for n, t in test_named.items():
        if n in aligned_test:
            continue
        shape2test[tuple(t.shape)].append((n, t))

    for shp, items in shape2test.items():
        if len(items) == 1 and len(shape2ref.get(shp, [])) == 1:
            tname, t_dst = items[0]
            _, t_src = shape2ref[shp][0]
            if _safe_copy_(t_dst, t_src):
                unique_shape_copied += 1
                aligned_test.add(tname)

    # 3) Shape mapping
    for name, t_dst in test_named.items():
        if name in aligned_test:
            continue
        ok = False
        for _, t_src in ref_named.items():
            if _try_map_shape_and_copy_(t_dst, t_src):
                mapped += 1
                aligned_test.add(name)
                ok = True
                break
        if not ok:
            skipped += 1
            missing_params.append(name) 

    if missing_params:
        stats = {
            "copied_same_shape": copied_same,
            "unique_shape_copied": unique_shape_copied,
            "mapped_shape": mapped,
            "skipped": skipped,
        }
        raise ParameterAlignmentError(
            message=f"Failed to align {len(missing_params)} parameters",
            missing_params=missing_params,
            ref_model_name=ref_model.__class__.__name__,
            test_model_name=test_model.__class__.__name__,
            stats=stats
        )

    return {
        "copied_same_shape": copied_same,
        "unique_shape_copied": unique_shape_copied,
        "mapped_shape": mapped,
        "skipped": skipped,
    }


_PAIR_ALIGNERS: dict[tuple[str, str], callable] = {}

def register_pair_aligner(ref_key: str, test_key: str):
    def deco(fn):
        _PAIR_ALIGNERS[(ref_key, test_key)] = fn
        return fn
    return deco

class ParameterAlignmentError(AssertionError):
    """
    参数对齐失败时抛出的异常
    """
    def __init__(
        self,
        message: str,
        missing_params: list[str],
        ref_model_name: str,
        test_model_name: str,
        stats: dict
    ):
        super().__init__(message)
        self.missing_params = missing_params
        self.ref_model_name = ref_model_name
        self.test_model_name = test_model_name
        self.stats = stats
    
    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Parameter Alignment Failed",
            f"{'='*60}",
            f"Reference Model : {self.ref_model_name}",
            f"Test Model      : {self.test_model_name}",
            f"",
            f"Missing/Unaligned Parameters ({len(self.missing_params)}):",
        ]
        for i, param in enumerate(self.missing_params[:20], 1):
            lines.append(f"  {i}. {param}")
        if len(self.missing_params) > 20:
            lines.append(f"  ... and {len(self.missing_params) - 20} more")
        
        lines.extend([
            f"",
            f"Alignment Stats:",
            f"  - copied_same_shape: {self.stats.get('copied_same_shape', 0)}",
            f"  - unique_shape_copied: {self.stats.get('unique_shape_copied', 0)}",
            f"  - mapped_shape: {self.stats.get('mapped_shape', 0)}",
            f"  - skipped: {self.stats.get('skipped', 0)}",
            f"{'='*60}",
        ])
        return "\n".join(lines)
@torch.no_grad()
def try_align_params(ref_model: nn.Module, test_model: nn.Module,
                     ref_mod=None, test_mod=None) -> dict[str, int]:

    key_export = (getattr(ref_model, "_export_symbol", None),
                  getattr(test_model, "_export_symbol", None))
    if key_export in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_export](ref_model, test_model)
        stats["pair_key"] = f"{key_export[0]}->{key_export[1]}"
        return stats

    key_class = (ref_model.__class__.__name__, test_model.__class__.__name__)
    if key_class in _PAIR_ALIGNERS:
        stats = _PAIR_ALIGNERS[key_class](ref_model, test_model)
        stats["pair_key"] = f"{key_class[0]}->{key_class[1]}"
        return stats

    for mod in (test_mod, ref_mod):
        if mod is None:
            continue
        for fn_name in ("map_ref_to_test_params", "align_params"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                fn(ref_model, test_model)
                return {"pair_aligner": 0, "copied_same_shape": -1, "mapped_shape": -1,
                        "skipped": -1, "pair_key": "custom_fn"}
    stats = align_params_generic(ref_model, test_model)
    stats["pair_aligner"] = 0
    stats["pair_key"] = "generic"
    return stats

def _run_once(model: nn.Module, inp: List[Any], dev: torch.device) -> Tuple[Any, float]:
    model.to(dev).eval()
    inp = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in inp]


    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(dev)
    with torch.inference_mode():
        start.record()
        out = model(*inp)
        end.record()
        end.synchronize()
    return out, start.elapsed_time(end)


def _bench(model: nn.Module, inp: List[Any], dev: torch.device, warm: int, rep: int) -> List[float]:
    """Multiple forward passes; return a list of latencies in ms."""
    model.to(dev).eval()
    inp = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in inp]

    # Warmup

    with torch.inference_mode():
        for _ in range(max(0, warm)):
            model(*inp)
    torch.cuda.synchronize(dev)

    # Measure

    times: List[float] = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.inference_mode():
        for _ in range(max(1, rep)):
            s.record()
            model(*inp)
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
    return times

class CompilationError(RuntimeError):
    """Raised when dynamic import / nvcc build fails.

    The *first* argument is the full build log (Python + ninja/nvcc).
    """


# =========================== dynamic import ===============================
def _capture_import(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    mod_name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)                     # type: ignore[arg-type]
    sys.modules[mod_name] = module
    assert spec.loader is not None

    # ---- Redirect Python-level stdout/stderr to StringIO -----------------
    py_buf = io.StringIO()

    # ---- Redirect OS-level FD 1/2 (stdout/stderr) to a temp file --------
    with tempfile.TemporaryFile(mode="w+") as fd_buf, \
         contextlib.redirect_stdout(py_buf), \
         contextlib.redirect_stderr(py_buf):

        # Save current FDs so we can restore later
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            os.dup2(fd_buf.fileno(), 1)     # redirect FD 1 → temp file
            os.dup2(fd_buf.fileno(), 2)     # redirect FD 2 → temp file

            # ------------ REAL IMPORT (build/compile) --------------------
            spec.loader.exec_module(module)                             # pyright: ignore[attr-defined]

            fd_buf.flush()
            fd_buf.seek(0)
            subproc_log = fd_buf.read()

        except Exception as exc:  # ← build / link / import failed
            # Combine StringIO + temp-file logs + Exception str
            fd_buf.flush(); fd_buf.seek(0)
            subproc_log = fd_buf.read()
            full_log = "".join([py_buf.getvalue(), subproc_log, str(exc)]).strip()
            #full_log = subproc_log.strip() #only beside py
            raise CompilationError(full_log) from None

        finally:
            # Always restore original FDs
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    # ---------------- SUCCESS --------------------------------------------
    return module, py_buf.getvalue() + subproc_log


def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    named: dict[str, torch.Tensor] = {}
    for k, p in model.named_parameters(recurse=True):
        named[f"param::{k}"] = p
    for k, b in model.named_buffers(recurse=True):
        named[f"buffer::{k}"] = b
    return named

@torch.no_grad()
def _safe_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.shape != src.shape:
        return False
    dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    return True

@torch.no_grad()
def _try_map_shape_and_copy_(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """
    Shape mapping coverage:
      - Depthwise 2D:   (C,1,Kh,1)<->(C,Kh), (C,1,Kh,Kw)<->(C,Kh,Kw)
      - PW/Linear:      (Out,In,1,1)<->(Out,In)
      - Conv/ConvT 3D:  (Out,In,kD,kH,kW) <-> (In,Out,kD,kH,kW) (swap first two dims)
      - Depthwise 3D:   (C,1,kD,kH,kW) <-> (C,kD,kH,kW)
    """
    s = tuple(src.shape)
    d = tuple(dst.shape)

    # --- depthwise 2D: (C,1,Kh,1) <-> (C,Kh)
    if len(s) == 4 and s[1] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[2] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[1] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- depthwise 2D: (C,1,Kh,Kw) -> (C,Kh,Kw) and reverse
    if len(s) == 4 and s[1] == 1 and len(d) == 3 and s[0] == d[0] and s[2] == d[1] and s[3] == d[2]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 3 and len(d) == 4 and d[1] == 1 and s[0] == d[0] and s[1] == d[2] and s[2] == d[3]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    # --- PW/Linear: (Out,In,1,1) <-> (Out,In)
    if len(s) == 4 and s[2] == 1 and s[3] == 1 and len(d) == 2 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True
    if len(s) == 2 and len(d) == 4 and d[2] == 1 and d[3] == 1 and s[0] == d[0] and s[1] == d[1]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).reshape(d).contiguous())
        return True

    # --- Conv/ConvTranspose 3D: swap first two dims for 5D weights
    #     (Out, In, kD, kH, kW)  <->  (In, Out, kD, kH, kW)
    if len(s) == 5 and len(d) == 5 and s[0] == d[1] and s[1] == d[0] and s[2:] == d[2:]:
        dst.copy_(src.permute(1, 0, 2, 3, 4).contiguous().to(dtype=dst.dtype, device=dst.device))
        return True

    # --- depthwise 3D: (C,1,kD,kH,kW) -> (C,kD,kH,kW) and reverse
    if len(s) == 5 and s[1] == 1 and len(d) == 4 and s[0] == d[0] and s[2:] == d[1:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).squeeze(1).contiguous())
        return True
    if len(s) == 4 and len(d) == 5 and d[1] == 1 and s[0] == d[0] and s[1:] == d[2:]:
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device).unsqueeze(1).contiguous())
        return True

    return False

