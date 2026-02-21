import os

def delete_files(folder, filename):
    count = 0
    for root, dirs, files in os.walk(folder):
        if filename in files:
            filepath = os.path.join(root, filename)
            os.remove(filepath)
            print(f"已删除: {filepath}")
            count += 1
    print(f"共删除 {count} 个文件")

if __name__ == "__main__":
    folder_path = r"D:\data\design\code\CudaOptiAgent-clear\run"
    target_filename = "fusion_plan.json"
    delete_files(folder_path, target_filename)