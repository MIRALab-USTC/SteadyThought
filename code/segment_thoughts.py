"""
This script calculates the token-level entropy of a model for a given dataset.

It loads a Hugging Face model and a dataset, processes the inputs using the tokenizer's
chat template, and then computes the entropy for each token in the input sequence based
on the model's output logits.

The results, including the token-level entropies, are saved to a new Parquet file.
"""
# import deepspeed
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import json
import jsonlines
import time
import re
from torch.cuda.amp import autocast
import csv







def segment_by_entropy_3(input_path, output_path, entropy_threshold=3.0):
    # print(input_path)
    # print(output_path)
    words = ["Alternatively","think differenly","another way","another thought", "another approach","another method","another solution","another strategy","another technique"]
    words_lower = sorted([word.lower() for word in words], key=len, reverse=True)

    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    print(f"数据来源地址: {input_path}")
    data = []
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip() # 移除前后的空白字符
            if not line: # 如果行为空，则跳过
                print(f"跳过空行 {idx+1}")
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error in line {idx+1}: {e}")
    print(f"总共有 {len(data)} 个段落数据")

    content_index = set()
    for item in data:
        content_index.add(item["original_content_index"])
        # print(f'原始内容索引 {item["original_content_index"]}')
    print(f"总共有 {len(content_index)} 个不同的原始内容索引")
    content_index = list(content_index)
    segment_all = 0
    percentages = []
    for index in content_index:
        dict = {
            "split_segments" : [],
            "switch_count": None,
            "problem": None,
            "result": None,
            "steps_split_indices": [],
        }
        subset = [item for item in data if item["original_content_index"] == index]
        dict["problem"] = subset[0]["problem"]
        dict["result"] = subset[0]["result"]
        print(f"\n处理原始内容索引 {index}，包含 {len(subset)} 个段落")
        full_content = "".join([para["paragraph_text"] for para in subset])
        segment_num = 0
        segment_text = subset[0]["paragraph_text"]
        segment_steps = [0]
        for idx in range(len(subset) - 1):
            para = subset[idx+1]
            # segment_text += para["paragraph_text"]
            para_entropies = para["token_entropies"]
            if len(para_entropies) < 5:
                segment_text += para["paragraph_text"]
                segment_steps.append(para["paragraph_index"])
                continue

            para_text = para.get("paragraph_text", "")
            cleaned_text = re.sub(r'[.,;!?"\':()]', ' ', para_text)
            
            # 使用 split() 分词，然后取前10个非空单词
            # 假设一个“单词”是被空格分隔的字符串
            words_in_para = [w for w in cleaned_text.lower().split() if w]
            
            # 获取前10个单词
            first_10_words = words_in_para[:10]
            
            # 2. 将前10个单词重新组合成一个字符串，方便进行短语匹配
            # 为了处理 "think differenly" 这种短语，直接在段落的开头进行子串匹配更稳健
            first_n_chars = para_text[:100].lower() # 仅检查前N个字符
            
            should_split = False
            for trigger_word in words_lower:
                # 检查触发词语是否出现在段落的前N个字符内
                # 注意：这里我们检查的是原始文本的前N个字符，以包含标点和空格
                if trigger_word in first_n_chars:
                    # 额外检查，确保触发词语位于段落的开头附近（例如，前10个单词范围内）
                    # 对于英文，我们检查是否为句子/段落的开头
                    # 更精确的判断是：检查触发词语是否是前10个单词/token的一部分
                    # 简单起见，我们假设如果它在前100个字符内，且以它开头，则分割。
                    # 或者，如果它出现在前10个单词组合的子串中
                    
                    # 重新基于单词检查: 将前10个单词（或更多）重新连接
                    # 考虑到 words_lower 列表中的词语都是英文，我们可以简化为：
                    # 检查 para_text 的开头部分是否包含任何 trigger_word
                    if trigger_word in ' '.join(first_10_words):
                        should_split = True
                        print(f"发现触发词语: '{trigger_word}'，分割出新段落")
                        break


            result = next((i for i, x in enumerate(para_entropies[:3]) if x > entropy_threshold), None)


            para_prior = subset[idx]
            para_prior_entropies = para_prior["token_entropies"]
            result_prior = next((i for i, x in enumerate(para_prior_entropies[-3:]) if x > entropy_threshold), None)

            if  (result is not None and (result_prior is not None)) or should_split:
                # dict ={
                #     "original_content_index": "",
                #     "segment_text" : "",
                #     "segment_index": "",
                #     "full_content": ""
                # }
                # dict["original_content_index"] = index
                # dict["segment_index"] = segment_num
                # dict["segment_text"] = segment_text
                # dict["full_content"] = full_content
                # f_output.write(dict.copy())
                if (result is not None and (result_prior is not None)):
                    print(f'基于熵值分割')
                dict["split_segments"].append(segment_text)
                segment_num += 1
                segment_text = para["paragraph_text"]
                
                dict["steps_split_indices"].append(segment_steps)
                segment_steps = [para["paragraph_index"]]
                # print(f'{tokenizer.decode(para["input_ids"][result], skip_special_tokens=False)} 的熵值超过阈值，分割出新段落')
          
            else :
                segment_text += para["paragraph_text"]
                segment_steps.append(para["paragraph_index"])
                continue
        if segment_text != subset[-1]["paragraph_text"]:
            # dict ={
            #     "original_content_index": "",
            #     "segment_text" : "",
            #     "segment_index": "",
            #     "full_content": ""
            # }
            # dict["original_content_index"] = index
            # dict["segment_index"] = segment_num
            # dict["segment_text"] = segment_text
            # dict["full_content"] = full_content
            # f_output.write(dict.copy())
            dict["split_segments"].append(segment_text)
            dict["steps_split_indices"].append(segment_steps)
            segment_num += 1
        else:
            # dict ={
            #     "original_content_index": "",
            #     "segment_text" : "",
            #     "segment_index": "",
            #     "full_content": ""
            # }
            # dict["original_content_index"] = index
            # dict["segment_index"] = segment_num
            # dict["segment_text"] = subset[-1]["paragraph_text"]
            # dict["full_content"] = full_content
            # f_output.write(dict.copy())
            dict["split_segments"].append(subset[-1]["paragraph_text"])
            dict["steps_split_indices"].append([subset[-1]["paragraph_index"]])
            segment_num += 1
        dict["switch_count"] = segment_num
        f_output.write(dict.copy())
        print(f"总共分割出 {segment_num} 个片段")
        segment_all += segment_num

        if dict["split_segments"] and "".join(dict["split_segments"]):
            total_chars = len("".join(dict["split_segments"]))
            last_segment_chars = len(dict["split_segments"][-1])
            percentage = (last_segment_chars / total_chars) * 100
            # print(f"最后一个片段的字符数占总字符数的百分比: {percentage:.2f}%")
            percentages.append(percentage)
    print(f"\n所有原始内容索引总共分割出 {segment_all} 个片段")
    print(f"平均每个原始内容索引分割出 {segment_all/len(content_index):.2f} 个片段")
    if percentages:
        avg_percentage = sum(percentages) / len(percentages)
        print(f"最后一个片段字符数占总字符数百分比的平均值: {avg_percentage:.2f}%")

    return segment_all/len(content_index) ,avg_percentage









def main():
    parser = argparse.ArgumentParser(description="Calculate token-level entropy for a model and dataset.")
    parser.add_argument("--input_path", type=str, default=None, help="Path to the Hugging Face model.")
    # parser.add_argument("--output_path", type=str, default=None, help="Path to save the output Parquet file with entropies.")
    parser.add_argument("--entropy_threshold", type=float, default=2.2, help="Path to save the output Parquet file with entropies.")
    parser.add_argument('--base_path', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/r1-1.5b/omni/entropy-icml-DeepSeek-R1-Distill-Qwen-1.5B-omni-level237-dataset.jsonl",
                        help='输入JSON文件路径')
    parser.add_argument('--second_path', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/r1-14b/omni/entropy-DeepSeek-R1-Distill-Qwen-1.5B-omni-level237-dataset.jsonl",
                        help='输入JSON文件路径')
    parser.add_argument('--epochs', type=int, default=1,
                        help='输入epoch数')
    parser.add_argument('--base_flag', default=False, help='是否使用Base_path')
    parser.add_argument('--math_or_code', default="math", help='是否使用Base_path')
    parser.add_argument('--epoch_start', type=int, default=1,
                        help='输入epoch数')
    parser.add_argument('--epoch_end', type=int, default=1)

    args = parser.parse_args()
    base_path = args.base_path

    epochs = args.epochs
    
    # print(f"文件路径:{json_file_path}")


    # 统一CSV文件路径，放在base_path同目录下，文件名为base_path最后一级目录名+_result.csv
    base_dir = os.path.dirname(base_path)
    # model_name = os.path.basename(base_path)
    model_name = base_path.split("/")[-2]
    csv_file = model_name + "_" + args.second_path.split("/")[0] + "_segment_result.csv"
    csv_path = os.path.join(base_path, csv_file)
    header = ["模型名称", "epoch数", "Thoughts个数","最后一个Thought占比"]
    write_header = not os.path.exists(csv_path)

    results = []
    for epoch in range(args.epoch_start , args.epoch_end + 1):
        if args.base_flag:
            if args.math_or_code == "math":
                json_file_path = os.path.join(base_path, f"epoch{epoch}", args.second_path, "entropy-math_eval_rewrite.jsonl")
            elif args.math_or_code == "code":
                json_file_path = os.path.join(base_path, f"epoch{epoch}", args.second_path, "entropy-code_eval.jsonl")
            segmented_name = "_segmented.jsonl"
        else:
            json_file_path = base_path
            segmented_name = "_segmented-" + str(args.entropy_threshold) + ".jsonl"
        segment_count,percentage = segment_by_entropy_3(json_file_path, json_file_path.replace(".jsonl", segmented_name), args.entropy_threshold)
        
    # 组织数据
        row = [
            model_name,
            epoch,
            f"{segment_count}" if segment_count else "-",
            f"{percentage:.2f}%" if percentage else "-"
        ]
        results.append(row)

    # 一次性写入文件
    


if __name__ == "__main__":
    main()