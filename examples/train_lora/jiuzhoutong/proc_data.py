import json
def convert_format(input_data):
    """
    将第一种格式 {"input", "target":""} 转换为第二种格式 {"messages": [...]}
    
    参数:
    input_data (dict): 输入的数据，格式为 {"input", "target":""}
    
    返回:
    dict: 转换后的数据，格式为 {"messages": [...]}
    """
    # 处理输入数据
    input_text = input_data.get("input", "")
    target_text = input_data.get("target", "")
    
    # 创建新格式
    output_data = {
        "messages": [
            {
                "from": "human",
                "value": input_text,
            },
            {
                "from": "gpt",
                "value": target_text,
            }
        ]
    }
    
    return output_data


# input_data = json.load(open("/home/jiangjin/mydata/jiuzhoutong/qwen-max/sft_data/jiuzhoutong_4_17216.jsonl", "r", encoding="utf-8"))

input_path = "/home/jiangjin/mydata/jiuzhoutong/qwen-max/sft_data/jiuzhoutong_4_17216.jsonl"

output_path = "/home/jiangjin/mydata/jiuzhoutong/jiuzhoutong_4_17216_0409.jsonl"

import json
import os
output_data = []
# input_data = json.load(open("/home/jiangjin/mydata/jiuzhoutong/yjyy_train_data.jsonl", "r", encoding="utf-8"))
# 逐行读取JSONL文件
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 跳过空行
            try:
                # 解析每一行的JSON对象
                item = json.loads(line)
                # 转换格式
                converted_item = convert_format(item)
                output_data.append(converted_item)
            except json.JSONDecodeError as e:
                print(f"跳过无效的JSON行: {e}")
                continue


# output_data = []
# for data in input_data:
#     converted_data = convert_format(data)
#     output_data.append(converted_data)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
print(f"数据已转换并保存到 {output_path}")