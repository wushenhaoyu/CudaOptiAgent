import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def geometric_mean_speed_ratio_correct_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    prod = np.prod(speed_up)
    n_correct = np.sum(is_correct) # Count number of correct samples

    return prod ** (1 / n_correct) if n_correct > 0 else 0

def geometric_mean_speed_ratio_correct_and_faster_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    speed_up = np.array([x for x in speed_up if x > 1])
    prod = np.prod(speed_up)
    n_correct_and_faster = len(speed_up)

    return prod ** (1 / n_correct_and_faster) if n_correct_and_faster > 0 else 0

def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0


def extract_latency_data(result_path: Path) -> Optional[Dict]:
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data.get("runnable", False):
            return None
        
        ref = data.get("ref_latency_ms")
        test = data.get("test_latency_ms")
        
        if ref is None or test is None:
            return None
            
        return {
            "ref_avg": ref["avg"],
            "test_avg": test["avg"],
            "path": str(result_path)
        }
    except Exception as e:
        print(f"Error reading {result_path}: {e}")
        return None


def process_and_save_metrics(root_dir: str, p: float = 1.0):
    root_path = Path(root_dir)
    
    samples = []
    for file_path in root_path.rglob("result.json"):
        data = extract_latency_data(file_path)
        if data:
            samples.append(data)
    
    print(f"Found {len(samples)} valid samples")
    
    if not samples:
        print("No valid samples found!")
        return
    
    n = len(samples)
    baseline = np.array([s["ref_avg"] for s in samples])
    actual = np.array([s["test_avg"] for s in samples])
    is_correct = np.ones(n, dtype=bool)
    
    metrics = {
        "fast1": fastp(is_correct, baseline, actual, n, p),
        "geometric_mean_speed_ratio": geometric_mean_speed_ratio_correct_only(
            is_correct, baseline, actual, n
        ),
        "sample_count": n,
        "p_threshold": p,
        "speedup_mean": float(np.mean(baseline / actual))
    }
    
    metrics_path = root_path / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved to {metrics_path}: fast1={metrics['fast1']:.4f}, "
          f"geo_mean={metrics['geometric_mean_speed_ratio']:.4f}")


# 使用
if __name__ == "__main__":
    process_and_save_metrics("/home/haoyu/code/CudaOptiAgent/run/openai_gpt-5-mini_v1/level3", p=1.0)