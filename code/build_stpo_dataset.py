
import json
import os
from tqdm import tqdm
import time
import argparse
import jsonlines
import re
from typing import Dict, Any, Tuple
import logging

from collections import defaultdict


import re
import os
from tqdm import tqdm
import time
import argparse
import jsonlines
import logging
from transformers import AutoTokenizer

def get_token_count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))



def process_json_file(json_path,json_path_2, output_path,model_path):
    """分批次处理JSON文件，逐条写入结果"""
    print(f"开始处理JSON文件: {json_path}")
    print(f"开始处理JSON-2文件: {json_path_2}")
    

    # with open(json_path, 'r') as f:
    #     data = json.load(f)
     
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    data_2 = []
    with open(json_path_2, 'r') as f:
        for line in f:
            item = json.loads(line)
            data_2.append(item)
    

    total_items = len(data[:])
    total_items_2 = len(data_2[:])
 
    print(f"总处理条目1: {total_items}")
    print(f"总处理条目2: {total_items_2}")

    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    # 生成带时间戳的输出文件名（若未指定）
    
    # 记录成功处理的条目数  

    count_final_true = 0
    success_count = 0
    # 逐条处理数据
    for i in tqdm(range(total_items), desc="处理进度"):
        item = data[i]
        item_2 = data_2[i]
       
        spilt_index = item["split_index"]
        if spilt_index == len(item["segments"]) -2 and item["mv_eval"] == True:
            continue
        if item_2["mv_eval"] == True:
            
            dict = {
                "instruction": None,
                "chosen": None,
                "rejected": None,
                "split_index":None, 
                "answer": None,
                "level": None
                }
            if "DeepSeek" in json_path:
                dict["chosen"] = item_2["model_generation"][0] + "<｜end▁of▁sentence｜>"
            elif "Qwen" in json_path:
                dict["chosen"] = item_2["model_generation"][0] + "<|im_end|>"
            qianxu_text = ""
            rejected_text = ""
            dict["split_index"] = spilt_index
            position = {
                "start": 0,
                "end": 0,
                "index": 0
            }
        
            for idx, segment in enumerate(item["segments"][:]):
                if idx <= spilt_index:
                    qianxu_text += segment
                else:
                    rejected_text += segment

            dict["answer"] = item["answer"]
            dict["rejected"] = rejected_text
            dict["instruction"] = item["question"]
            dict["level"] = item["level"]
            if item["isthink"] == True:
                if "DeepSeek" in json_path:
                    dict["rejected"] += "</think>" + item["<answer>"] + "<｜end▁of▁sentence｜>"
                elif "Qwen" in json_path:
                    dict["rejected"] += "</think>" + item["<answer>"] + "<|im_end|>"

                # # 加载分词器（根据你的模型选择合适的tokenizer）
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                instr_chosen = dict["instruction"] + dict["chosen"]
                instr_rejected = dict["instruction"] + dict["rejected"]
                tokens_chosen = get_token_count(instr_chosen, tokenizer)
                tokens_rejected = get_token_count(instr_rejected, tokenizer)
                
                
                if tokens_chosen > 6144 or tokens_rejected > 6144:   
                    print("仍然不符合要求，跳过该条数据")
                    continue
                            
                if item["mv_eval"] == True:
                    count_final_true += 1
            
                f_output.write(dict)
                # print(f"第{i+1}条数据处理成功")
                success_count += 1

        # else:
            # print(f"第{i+1}条数据不符合要求，跳过")
            # print(item_2["mv_eval"])
    print(f"处理完成，成功处理条目: {success_count}")
    print(f"原本答案正确的条目: {count_final_true}")


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='在指定GPU上运行Llama模型分析JSON文件')
    
    # 添加命令行参数

    parser.add_argument('--json_path', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/r1-14b/omni/entropy-DeepSeek-R1-Distill-Qwen-14B-omni-level237-dataset_segmented-1.9-stickbyself.jsonl",
                        help='输入JSON文件路径')
    parser.add_argument('--json_path_2', type=str, default="/home/jwangxgroup/pengzhenyue/SEAL-main/results/DeepSeek-R1-Distill-Qwen-14B/OMNI237-DeepSeek-R1-Distill-Qwen-14B-omni-level237-dataset_segmented-1.9-stickbyself/epoch1/logit-adjustment/ours_value_-1000000.0_first20000/math_eval_rewrite.jsonl",
                        help='输入JSON文件路径')
    parser.add_argument('--output_path', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/r1-14b/omni/DeepSeek-R1-Distill-Qwen-14B-omni-level237-dataset_segmented-1.9-stickbyself-simpo-nobug.jsonl",
                        help='输出JSON文件路径，不指定时自动生成带时间戳的文件名')
    parser.add_argument('--model_path', type=str, default="/home/jwangxgroup/jiewang/shared/Qwen3-8B",
                        help='输入JSON文件路径')


    
    # 解析命令行参数
    args = parser.parse_args()
    

    
    # 处理JSON文件
    # count(args.json_path ,args.json_path_2)
    process_json_file(args.json_path ,args.json_path_2 ,args.output_path,args.model_path)
    

if __name__ == "__main__":
    main()