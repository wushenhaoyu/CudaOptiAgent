import os
import json
from pathlib import Path
from typing import List, Dict


def find_result_json_files(root_dir: str) -> List[Path]:
    """递归查找所有 result.json 文件"""
    root = Path(root_dir)
    return list(root.rglob("result.json"))


def remove_fast1_from_file(result_path: Path) -> bool:
    """从单个 result.json 文件中删除 fast1 字段"""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查并删除 fast1 字段
        if "fast1" in data:
            del data["fast1"]
            
            # 写回文件
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Removed 'fast1' from: {result_path}")
            return True
        else:
            return False
            
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing {result_path}: {e}")
        return False


def remove_all_fast1(root_dir: str) -> Dict[str, bool]:
    """遍历所有 result.json，删除其中的 fast1 字段"""
    result_files = find_result_json_files(root_dir)
    results = {}
    
    print(f"Found {len(result_files)} result.json files\n")
    
    for file_path in result_files:
        removed = remove_fast1_from_file(file_path)
        results[str(file_path)] = removed
    
    # 统计
    removed_count = sum(1 for v in results.values() if v)
    print(f"\nTotal: {len(results)} files, Removed 'fast1' from: {removed_count} files")
    
    return results


# 使用
if __name__ == "__main__":
    root_directory = "./run"  # 修改为你的路径
    remove_all_fast1(root_directory)