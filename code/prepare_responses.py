import argparse
from random import random

from tqdm import tqdm
import os
import json
import math
import jsonlines
import json
import re
from collections import defaultdict
from transformers import AutoTokenizer

from transformers import AutoTokenizer



def new_dataset(input_file ,input_file_2, output_file):
    data = []
    with open(input_file, 'r') as infile:
        for line in infile:
            item = json.loads(line)
            data.append(item)
    data_2 = []
    with open(input_file_2, 'r') as infile:
        for line in infile:
            item = json.loads(line)
            data_2.append(item)
    f_output = jsonlines.Writer(open(output_file, "w", encoding="utf-8"))
    for idx,item_2 in enumerate(data_2):
        if item_2["problem"] == data[idx]["problem"]:
            dict = {
                "prompt": "",
                "model_generation": None,
                "<answer>":"",
                "answer" : "",
                "mv_eval" : "",
                "isthink": "",
                "problem": "",
                "level":  "",
                "solution": ""
            }
            if "</think>" in item_2["model_generation"][0]:
                dict["prompt"] = item_2["prompt"]
                dict["model_generation"] = item_2["model_generation"]
                dict["<answer>"] = item_2["model_generation"][0].split("</think>")[1]
                dict["isthink"] = True
            else:
                dict["prompt"] = item_2["prompt"]
                dict["model_generation"] = item_2["model_generation"]
                dict["isthink"] = False
            dict["answer"] = item_2["answer"]
            dict["level"] = data[idx]["level"]
            dict["problem"] = data[idx]["problem"]
            dict["mv_eval"] = item_2["mv_eval"]
            dict["solution"] = data[idx]["solution"]
            f_output.write(dict)




def main():
    parser = argparse.ArgumentParser(description="Prepare model responses for thought segmentation.")
    parser.add_argument("--data_path", required=True, help="Original question JSONL file.")
    parser.add_argument("--result_path", required=True, help="Evaluated inference JSONL file.")
    parser.add_argument("--output_path", required=True, help="Output response JSONL file.")
    args = parser.parse_args()
    new_dataset(args.data_path, args.result_path, args.output_path)


if __name__ == "__main__":
    main()
