import json
from pathlib import Path
from scripts.test_kernel import _test_kernel_process
from utils.utils import extract_error_report





import json
import re


def extract_json(text: str) -> dict:
    """
    从文本中提取JSON对象，支持多种格式：
    1. 标准JSON代码块 ```json ... ```
    2. 普通代码块 ``` ... ```
    3. [recommendation]...[/recommendation] 标签
    4. 裸JSON对象 {...}
    """
    # 尝试各种提取模式
    patterns = [
        # 标准JSON代码块
        r'```json\s*(\{.*?\})\s*```',
        # 普通代码块
        r'```\s*(\{.*?\})\s*```',
        # recommendation标签
        r'\[recommendation\]\s*(\{.*?\})\s*\[/recommendation\]',
        # 裸JSON对象（最宽松的匹配，放在最后）
        r'(\{[\s\S]*?"operators"[\s\S]*?\})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            try:
                # 清理可能的转义问题
                cleaned = match.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    
    # 如果以上都失败，尝试直接找最外层的大括号
    try:
        # 找到第一个 { 和最后一个匹配的 }
        start = text.find('{')
        if start == -1:
            return {}
        
        # 简单的括号计数匹配
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


# 使用示例
if __name__ == "__main__":
    # 测试用例1: 标准代码块
    text1 = '''
    这里是一些说明文字
    ```json
    {
      "operators": [
        {"name": "conv2d", "type": "nn.Conv2d", "category": "complex_out_fusable"}
      ]
    }
    ```
    '''
    
    # 测试用例2: recommendation标签
    text2 = '''
    [recommendation]
    {
      "operators": [
        {"name": "relu", "type": "torch.relu", "category": "injective"}
      ]
    }
    [/recommendation]
    '''
    
    # 测试用例3: 裸JSON
    text3 = '''{
      "operators": [
        {"name": "conv2d", "type": "nn.Conv2d", "category": "complex_out_fusable"},
        {"name": "relu", "type": "torch.relu", "category": "injective"},
        {"name": "bias_add", "type": "elementwise_add_with_broadcast", "category": "injective"}
      ],
      "fusion_groups": [
        {
          "group_id": 1,
          "operators": ["conv2d", "relu", "bias_add"],
          "fusion_type": "complex_output_fusion",
          "rules_used": [
            "injective operators can fuse to complex_out_fusable output",
            "multiple injective operators can fuse together"
          ],
          "justification": "relu and bias_add are injective and applied to conv2d output"
        }
      ],
      "fusion_boundaries": []
    }'''
    
    print("Test 1 (code block):", extract_json(text1))
    print("Test 2 (recommendation tag):", extract_json(text2))
    print("Test 3 (raw JSON):", extract_json(text3))