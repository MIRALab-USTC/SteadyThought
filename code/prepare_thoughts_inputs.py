
import json
import os
from tqdm import tqdm
import time
import argparse
import jsonlines




def process_json_file(json_path, json_path_2, output_path):
    """分批次处理JSON文件，逐条写入结果"""
    print(f"开始处理JSON文件: {json_path}")
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # with open(json_path_2, 'r', encoding='utf-8') as f:
    #     data_2 = json.load(f)
    data_2 = []
    with open(json_path_2, 'r') as f:
        for line in f:
            item = json.loads(line)
            data_2.append(item)

    total_items = len(data)
    total_items_2 = len(data_2) 
    print(f"总处理条目1: {total_items}")
    print(f"总处理条目2: {total_items_2}")

    
    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    # 生成带时间戳的输出文件名（若未指定）
    if not output_path:
        base_name = os.path.basename(json_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{name}_analyzed_{timestamp}{ext}"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开输出文件（追加模式）
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write("[\n")  # 写入JSON数组开头
        
        success_count = 0
        # 逐条处理数据
        for i in tqdm(range(total_items), desc="处理进度"):
            item = data[i]
            item_2 = data_2[i]
            
            answer = item_2.get("prompt_answer")
            switch_count = item["completion_processed"]["switch_count"]

            segments_list = []
            paragraphs = ""
            if switch_count == 0:
                print(f"第{i+1}条数据无思维切换")
            else:
                for segment in item["completion_processed"]["split_segments"]:
                    paragraphs += segment
                    segments_list.append(paragraphs)


                promblem = item_2.get("promblem", "")

                dict = {
                    "instruction": None,
                    "answer": ""

                }
                
                for idx, segment in enumerate(segments_list):
                    if idx != len(segments_list) - 1:
                        prompt = prompt_product(promblem, segment, answer)
                        dict["instruction"] = prompt
                        dict["answer"] = answer
                        f_output.write(dict)
                                
            
                # 写入当前条目（JSON格式）
                # json_str = json.dumps(item, ensure_ascii=False, indent=4)
                # if i < total_items - 1:
                #     json_str += ","  # 非最后一条添加逗号
                # print(json_str)
                # f.write(prompt + "\n")
                
                
                print(f'写入第{i+1}条数据')
            
           
        
        
    
    print(f"处理完成！结果已保存至: {output_path}")
    print(f"成功分析: {success_count} 条")
    return success_count


def process_json_file_2(json_path, json_path_2, output_path):
    """分批次处理JSON文件，逐条写入结果"""
    print(f"开始处理JSON文件: {json_path}")
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # with open(json_path_2, 'r', encoding='utf-8') as f:
    #     data_2 = json.load(f)
    data_2 = []
    with open(json_path_2, 'r') as f:
        for line in f:
            item = json.loads(line)
            data_2.append(item)

    total_items = len(data)
    total_items_2 = len(data_2) 
    print(f"总处理条目1: {total_items}")
    print(f"总处理条目2: {total_items_2}")

    
    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    # 生成带时间戳的输出文件名（若未指定）
    if not output_path:
        base_name = os.path.basename(json_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{name}_analyzed_{timestamp}{ext}"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开输出文件（追加模式）
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write("[\n")  # 写入JSON数组开头
        
        success_count = 0
        # 逐条处理数据
        for i in tqdm(range(total_items), desc="处理进度"):
            item = data[i]
            item_2 = data_2[i]
            
            answer = item_2.get("answer")
            
            switch_count = item["completion_processed"]["switch_count"]

            segments_list = []
            if switch_count == 0:
                print(f"第{i+1}条数据无思维切换")
            else:
                # print(item)
                for segment in item["completion_processed"]["split_segments"]:
                    
                    segments_list.append(segment)


                problem = item_2.get("problem", "")
                
                dict = {
                    "instruction": None,
                    "answer": None,
                    "split_index": None,
                    "problem":None,
                    "segments":None,
                    "<answer>":None,
                    "isthink":None

                }
                
                for idx, segment in enumerate(segments_list):
                    if idx != len(segments_list) - 1:
                        prompt = prompt_product(problem, segment, answer)
                        dict["instruction"] = prompt
                        dict["answer"] = answer
                        dict["split_index"] = idx
                        dict["problem"] = problem
                        dict["segments"] = segments_list
                        dict["<answer>"] = item_2.get("<answer>")
                        dict["isthink"] = item_2.get("isthink")
                        f_output.write(dict)
                                
            
                # 写入当前条目（JSON格式）
                # json_str = json.dumps(item, ensure_ascii=False, indent=4)
                # if i < total_items - 1:
                #     json_str += ","  # 非最后一条添加逗号
                # print(json_str)
                # f.write(prompt + "\n")
                
                
                print(f'写入第{i+1}条数据')


def process_json_file_self(json_path, json_path_2, output_path):
    """分批次处理JSON文件，逐条写入结果"""
    print(f"开始处理JSON文件: {json_path}")
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # with open(json_path_2, 'r', encoding='utf-8') as f:
    #     data_2 = json.load(f)
    data_2 = []
    with open(json_path_2, 'r') as f:
        for line in f:
            item = json.loads(line)
            data_2.append(item)

    total_items = len(data)
    total_items_2 = len(data_2) 
    print(f"总处理条目1: {total_items}")
    print(f"总处理条目2: {total_items_2}")

    
    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    # 生成带时间戳的输出文件名（若未指定）
    if not output_path:
        base_name = os.path.basename(json_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{name}_analyzed_{timestamp}{ext}"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开输出文件（追加模式）
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write("[\n")  # 写入JSON数组开头
        
        success_count = 0
        # 逐条处理数据
        for i in tqdm(range(total_items), desc="处理进度"):
            item = data[i]
            item_2 = data_2[i]
            
            answer = item_2.get("answer")
            
            switch_count = item["completion_processed"]["switch_count"]

            segments_list = []
            if switch_count == 0:
                print(f"第{i+1}条数据无思维切换")
            else:
                # print(item)
                for segment in item["completion_processed"]["split_segments"]:
                    
                    segments_list.append(segment)


                problem = item_2.get("problem", "")
                
                dict = {
                    "question": None,
                    "answer": None,
                    "split_index": None,
                    "problem":None,
                    "segments":None,
                    "<answer>":None,
                    "isthink":None,
                    "solution":None,
                    "mv_eval":None

                }
                
                for idx, segment in enumerate(segments_list):
                    if idx != len(segments_list) - 1:
                        # prompt = prompt_product(problem, segment, answer)
                        dict["question"] = "<｜begin▁of▁sentence｜><｜User｜>Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\nQuestion: " + problem +"<｜Assistant｜><think>\n" + segment
                        dict["answer"] = answer
                        dict["split_index"] = idx
                        dict["problem"] = problem
                        dict["segments"] = segments_list
                        dict["<answer>"] = item_2.get("<answer>")
                        dict["isthink"] = item_2.get("isthink")
                        dict["solution"] = item_2.get("solution")
                        dict["mv_eval"] = item_2.get("mv_eval")
                        f_output.write(dict)
                                
            
                # 写入当前条目（JSON格式）
                # json_str = json.dumps(item, ensure_ascii=False, indent=4)
                # if i < total_items - 1:
                #     json_str += ","  # 非最后一条添加逗号
                # print(json_str)
                # f.write(prompt + "\n")
                
                
                print(f'写入第{i+1}条数据')

def process_json_file_new_dataset(json_path, json_path_2, output_path):
    """分批次处理JSON文件，逐条写入结果"""
    print(f"开始处理JSON文件: {json_path}")
    
    # 读取JSON文件
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    # with open(json_path_2, 'r', encoding='utf-8') as f:
    #     data_2 = json.load(f)
    data_2 = []
    with open(json_path_2, 'r') as f:
        for line in f:
            item = json.loads(line)
            data_2.append(item)

    total_items = len(data)
    total_items_2 = len(data_2) 
    print(f"总处理条目1: {total_items}")
    print(f"总处理条目2: {total_items_2}")

    
    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    # 生成带时间戳的输出文件名（若未指定）
    if not output_path:
        base_name = os.path.basename(json_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{name}_analyzed_{timestamp}{ext}"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开输出文件（追加模式）
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write("[\n")  # 写入JSON数组开头
        
        success_count = 0
        # 逐条处理数据

        for i in tqdm(range(total_items), desc="处理进度"):
            item = data[i]
            item_2 = data_2[i]
            
            answer = item_2.get("answer")
            
            switch_count = item["switch_count"]

            segments_list = []
            if switch_count == 1:
                print(f"第{i+1}条数据无思维切换")
            else:
                # print(item)
                for segment in item["split_segments"]:
                    
                    segments_list.append(segment)


                problem = item_2.get("problem", "")
                
                dict = {
                    "question": None,
                    "answer": None,
                    "split_index": None,
                    "problem":None,
                    "segments":None,
                    "<answer>":None,
                    "isthink":None,
                    "solution":None,
                    "mv_eval":None,
                    "level":None

                }
                
                for idx, segment in enumerate(segments_list):
                    if idx != len(segments_list) - 1:
                        if idx == 0:
                            dict["question"] = segment
                        else:
                            if "DeepSeek" in json_path:
                                dict["question"] = "<｜begin▁of▁sentence｜><｜User｜>Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\nQuestion: " + problem +"<｜Assistant｜><think>\n" + segment
                            elif "Qwen" in json_path:
                                dict["question"] = "<|im_start|>user\nAnswer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\nQuestion: " + problem + "<|im_end|>\n<|im_start|>assistant\n<think>\n"  + segment
                        dict["answer"] = item_2["answer"]
                        dict["split_index"] = idx
                        dict["problem"] = problem
                        dict["segments"] = segments_list
                        dict["<answer>"] = item_2.get("<answer>")
                        dict["isthink"] = item_2.get("isthink")
                        dict["solution"] = item_2.get("solution")
                        dict["mv_eval"] = item_2.get("result")
                        dict["level"] = item_2["level"]
                    
                        f_output.write(dict)
                        success_count += 1
                                
            
                # 写入当前条目（JSON格式）
                # json_str = json.dumps(item, ensure_ascii=False, indent=4)
                # if i < total_items - 1:
                #     json_str += ","  # 非最后一条添加逗号
                # print(json_str)
                # f.write(prompt + "\n")
                
                
                print(f'写入第{i+1}条数据')
    print(f"处理完成！结果已保存至: {output_path}")
    print(f"成功分析: {success_count} 条")


def process_json_file_new_dataset_2(json_path, json_path_2, output_path):
    """分批次处理JSON文件，逐条写入结果"""
    print(f"开始处理JSON文件: {json_path}")
    
    # 读取JSON文件
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

    # with open(json_path_2, 'r', encoding='utf-8') as f:
    #     data_2 = json.load(f)
   

    total_items = len(data) 
    total_items_2 = len(data_2)

    print(f"总处理条目1: {total_items}")
    print(f"总处理条目2: {total_items_2}")


    
    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    # 生成带时间戳的输出文件名（若未指定）
    if not output_path:
        base_name = os.path.basename(json_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{name}_analyzed_{timestamp}{ext}"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开输出文件（追加模式）
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write("[\n")  # 写入JSON数组开头
        
        success_count = 0
        # 逐条处理数据

        for i in tqdm(range(total_items), desc="处理进度"):
            item = data[i]
            item_2 = data_2[i]
            
            switch_count = item["switch_count"]

            segments_list = []
            if switch_count == 1:
                print(f"第{i+1}条数据无思维切换")
            else:
                # print(item)
                for segment in item["split_segments"]:
                    
                    segments_list.append(segment)

                problem = item.get("problem", "")
                
                dict = {
                    "question": None,
                    "answer": None,
                    "split_index": None,
                    "problem":None,
                    "segments":None,
                    "<answer>":None,
                    "isthink":None,
                    "solution":None,
                    "mv_eval":None,
                    "level":None

                }
                
                for idx, segment in enumerate(segments_list):
                    if idx != len(segments_list) - 1:
                        if idx == 0:
                            dict["question"] = segment
                        else:
                            if "DeepSeek" in json_path:
                                dict["question"] = "<｜begin▁of▁sentence｜><｜User｜>Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\nQuestion: " + problem +"<｜Assistant｜><think>\n" + segment
                            elif "Qwen" in json_path:
                                dict["question"] = "<|im_start|>user\nAnswer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\nQuestion: " + problem + "<|im_end|>\n<|im_start|>assistant\n<think>\n"  + segment
                        dict["answer"] = item_2["answer"]
                        dict["problem"] = problem
                        dict["segments"] = segments_list
                  
                      
                   
                        dict["solution"] = item_2.get("solution")
                     
                      
                    
                        f_output.write(dict)
                        success_count += 1
                                
            
                # 写入当前条目（JSON格式）
                # json_str = json.dumps(item, ensure_ascii=False, indent=4)
                # if i < total_items - 1:
                #     json_str += ","  # 非最后一条添加逗号
                # print(json_str)
                # f.write(prompt + "\n")
                
                
                print(f'写入第{i+1}条数据')
    print(f"处理完成！结果已保存至: {output_path}")
    print(f"成功分析: {success_count} 条")

def process_json_file_new_dataset_3(json_path, json_path_2, output_path):
    """分批次处理JSON文件，逐条写入结果"""
    print(f"开始处理JSON文件: {json_path}")
    
    # 读取JSON文件
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

    # with open(json_path_2, 'r', encoding='utf-8') as f:
    #     data_2 = json.load(f)
    # data = data[:100]
    # data_2 = data_2[:100]

    total_items = len(data) 
    total_items_2 = len(data_2)

    print(f"总处理条目1: {total_items}")
    print(f"总处理条目2: {total_items_2}")


    
    f_output = jsonlines.Writer(open(output_path, "w", encoding="utf-8"))
    # 生成带时间戳的输出文件名（若未指定）
    if not output_path:
        base_name = os.path.basename(json_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{name}_analyzed_{timestamp}{ext}"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开输出文件（追加模式）
    with open(output_path, 'w', encoding='utf-8') as f:
        # f.write("[\n")  # 写入JSON数组开头
        
        success_count = 0
        # 逐条处理数据

        for i in tqdm(range(total_items), desc="处理进度"):
            item = data[i]
            item_2 = data_2[i]
            
            switch_count = item["switch_count"]

            segments_list = []
            if switch_count == 1:
                print(f"第{i+1}条数据无思维切换")
            else:
                # print(item)
                for segment in item["split_segments"]:
                    
                    segments_list.append(segment)

                problem = item.get("problem", "")
                
                dict = {
                    "question": None,
                    "answer": None,
                    "split_index": None,
                    "problem":None,
                    "segments":None,
                    "<answer>":None,
                    "isthink":None,
                    "solution":None,
                    "mv_eval":None,
                    "level":None,
                    "question_index" : None

                }
                
                for idx, segment in enumerate(segments_list):
                    if idx != len(segments_list) - 1:
                        if idx == 0:
                            dict["question"] = segment
                        else:
                            if "DeepSeek" in json_path:
                                dict["question"] = "<｜begin▁of▁sentence｜><｜User｜>Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\nQuestion: " + problem +"<｜Assistant｜><think>\n" + segment
                            elif "Qwen" in json_path:
                                dict["question"] = "<|im_start|>user\nAnswer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\nQuestion: " + problem + "<|im_end|>\n<|im_start|>assistant\n<think>\n"  + segment
                        dict["answer"] = item_2["answer"]
                        dict["problem"] = problem
                        dict["segments"] = segments_list
                        dict["mv_eval"] = item_2.get("mv_eval")
                        dict["question_index"] = i
                        dict["split_index"] = idx
                   
                        dict["solution"] = item_2.get("solution")
                        dict["<answer>"] = item_2.get("<answer>")
                        dict["isthink"] = item_2.get("isthink")
                        dict["level"] = item_2["level"]
                    
                        f_output.write(dict)
                        success_count += 1
                        # break # 只写入第一条切分数据
                                
            
                # 写入当前条目（JSON格式）
                # json_str = json.dumps(item, ensure_ascii=False, indent=4)
                # if i < total_items - 1:
                #     json_str += ","  # 非最后一条添加逗号
                # print(json_str)
                # f.write(prompt + "\n")
                
                
                print(f'写入第{i+1}条数据')
    print(f"处理完成！结果已保存至: {output_path}")
    print(f"成功分析: {success_count} 条")

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='在指定GPU上运行Llama模型分析JSON文件')
    
    # 添加命令行参数

    parser.add_argument('--json_path', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/r1-1.5b/omni/entropy-icml-DeepSeek-R1-Distill-Qwen-1.5B-omni-level237-dataset_segmented-wordsonly-2.2.jsonl",
                        help='输入JSON文件路径')
    parser.add_argument('--json_path_2', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/r1-1.5b/omni/DeepSeek-R1-Distill-Qwen-1.5B-omni-level237-dataset.jsonl",
                        help='输入JSON文件路径')
    parser.add_argument('--output_path', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/r1-1.5b/omni/DeepSeek-R1-Distill-Qwen-1.5B-omni-level237-dataset_segmented-2.2-stickbyself-wordsonly-sample.jsonl",
                        help='输出JSON文件路径，不指定时自动生成带时间戳的文件名')

   
    
    # 解析命令行参数
    args = parser.parse_args()
    

    
    # 处理JSON文件
    # process_json_file_new_dataset(args.json_path, args.json_path_2, args.output_path) #需要做成训练数据
    # process_json_file_new_dataset_2(args.json_path, args.json_path_2, args.output_path) #只需要做测试数据
    process_json_file_new_dataset_3(args.json_path, args.json_path_2, args.output_path) #需要做成训练数据，且增加solution和mv_eval字段

if __name__ == "__main__":
    main()