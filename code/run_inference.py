
import argparse
import os
# os.environ["VLLM_USE_V1"] = "0"  # 强制使用 V0 引擎
import re
import json
import random
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial
from itertools import islice
import torch.nn.functional as F


import sys
import os
import gc

from get_math_results import main as eval_main
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# exact_match = evaluate.load("exact_match")
exact_match = evaluate.load("evaluate-main/metrics/exact_match")


# def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
#     if max_len <= 0 or len(token_ids) <= max_len:
#         logits[adjust_ids.to(logits.device)] += values
#     return logits
def print_vocab_tokens(tokenizer, max_tokens=None):
    """
    打印词汇表中所有 Token ID 及其对应的 Token
    
    参数:
    tokenizer: 分词器实例
    max_tokens: 最大打印数量，None 表示打印所有
    """
    # 获取词汇表
    vocab = tokenizer.get_vocab()
    
    # 创建 ID 到 Token 的映射
    id_to_token = {id_: token for token, id_ in vocab.items()}
    
    # 获取所有 ID 并排序
    sorted_ids = sorted(id_to_token.keys())
    
    # 限制打印数量
    if max_tokens is not None:
        sorted_ids = sorted_ids[:max_tokens]
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"打印前 {len(sorted_ids)} 个 Token:")
    print("-" * 80)
    print(f"{'ID':<10} {'Token':<30} {'Length':<10}")
    print("-" * 80)
    
    for token_id in sorted_ids:
        token = id_to_token[token_id]
        # 对特殊字符进行转义以便更好地显示
        display_token = repr(token)
        print(f"{token_id:<10} {display_token:<30} {len(token):<10}")
    
    print("-" * 80)

def search_vocab_tokens(tokenizer, search_term=None, max_results=20):
    """
    搜索词汇表中的 token
    
    参数:
    tokenizer: 分词器实例
    search_term: 搜索词，None 表示不搜索
    max_results: 最大结果显示数量
    """
    vocab = tokenizer.get_vocab()
    id_to_token = {id_: token for token, id_ in vocab.items()}
    
    if search_term is None:
        # 不搜索，直接返回所有 token 数量
        print(f"词汇表大小: {len(vocab)}")
        return
    
    print(f"搜索包含 '{search_term}' 的 token:")
    print("-" * 80)
    print(f"{'ID':<10} {'Token':<30} {'Length':<10}")
    print("-" * 80)
    
    results = []
    for token_id, token in id_to_token.items():
        # if search_term in token:
        #     results.append((token_id, token))
        if token_id == 3783 :
            results.append((token_id, token))
    
    # 限制结果数量
    results = results[:max_results]
    
    for token_id, token in results:
        display_token = repr(token)
        print(f"{token_id:<10} {display_token:<30} {len(token):<10}")
    
    print("-" * 80)
    print(f"找到 {len(results)} 个结果")
    
    return results


def logit_adjustment(token_ids, logits, tokenizer, adjust_ids, values, max_len=-1):
    
    # 检查是否满足调整条件
    condition_met = max_len <= 0 or len(token_ids) <= max_len
    
    if condition_met:
    # 确保 adjust_ids 在与 logits 相同的设备上
        device_adjust_ids = adjust_ids.to(logits.device)
        
        # # 计算调整前的概率
        # probs_before = F.softmax(logits, dim=-1)
        
        # # 找到调整前概率最高的token（即下一个token）
        # next_token_before = torch.argmax(probs_before).item()
        # next_token_prob_before = probs_before[next_token_before].item()
        
        # # 总是打印下一个token的信息
        # print(f"\n=== 下一个token信息 ===")
        # print(f"当前生成长度: {len(token_ids)}")
        # print(f"下一个token ID: {next_token_before}")
        # print(f"下一个token: '{tokenizer.decode([next_token_before])}'")
        # print(f"下一个token概率: {next_token_prob_before:.6f}")
        
        # # 检查下一个token是否是要调整的token
        # is_adjust_token = next_token_before in device_adjust_ids
        # print(f"是否是要调整的token: {is_adjust_token}")
        
        # 执行调整
        logits[device_adjust_ids] += values
        
        # 计算调整后的概率
        # probs_after = F.softmax(logits, dim=-1)
        # next_token_after = torch.argmax(probs_after).item()
        # next_token_prob_after = probs_after[next_token_after].item()
        
        # print(f"调整后下一个token ID: {next_token_after}")
        # print(f"调整后下一个token: '{tokenizer.decode([next_token_after])}'")
        # print(f"调整后下一个token概率: {next_token_prob_after:.6f}")
        
        # # 如果下一个token是要调整的token，打印详细信息
        # if is_adjust_token:s
        #     print(f"!!! 下一个token是要调整的token '{tokenizer.decode([next_token_before])}' !!!")
        #     print(f"调整前概率: {next_token_prob_before:.6f}")
        #     print(f"调整后概率: {probs_after[next_token_before].item():.6f}")
            
        #     # 如果调整改变了下一个token，打印详细信息
        #     if next_token_before != next_token_after:
        #         print(f"!!! 调整改变了下一个token: '{tokenizer.decode([next_token_before])}' -> '{tokenizer.decode([next_token_after])}' !!!")
        # else:
        #     # 如果下一个token不是要调整的token，但仍然检查是否调整改变了下一个token
        #     if next_token_before != next_token_after:
        #         print(f"!!! 调整间接改变了下一个token: '{tokenizer.decode([next_token_before])}' -> '{tokenizer.decode([next_token_after])}' !!!")
    else:
        # 不满足调整条件，直接返回原始logits
        print(f"调整条件不满足: 当前长度 {len(token_ids)}, 最大长度限制 {max_len}")
        pass

    return logits


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a

def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans


def load_jsonl_data(data_path):
    """Load a user-specified math dataset or thought-continuation JSONL file."""
    test_data = []
    with open(data_path, encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                question = example["question"] if "question" in example else example["problem"]
                test_data.append({
                    "question": question,
                    "answer": example["solution"],
                    "gt": example["answer"],
                })
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{data_path}, line {line_number}: expected question (or problem), answer, and solution"
                ) from exc
    if not test_data:
        raise ValueError(f"No data found in {data_path}")
    return test_data



def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.data_path:
        test_data = load_jsonl_data(args.data_path)
    elif args.dataset == "MATH":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            gt = extract_box(example["solution"])
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt":gt,
            })
    elif args.dataset == "MATH500":
        data_path = "math500.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":gt.replace(" ", ""),
                })
    
    elif args.dataset == "OMNI237":
        data_path = "data/omni_math/omni_math_level237.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":example["answer"],
                })
    

    elif args.dataset == "AIME24":
        data_path = "data/aime24/AIME24.json"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["solution"])
                answer = example["answer"].lstrip('0') if example["answer"].startswith('0') else example["answer"]
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":answer,
                })
    elif args.dataset == "AIME248":
        data_path = "data/aime24/expanded_aime24.json"
        with open(data_path, encoding="utf-8") as fin:
            # 读取整个文件内容
            content = fin.read()
            # 解析整个 JSON 数组
            data = json.loads(content)
            test_data = []
            # 遍历数组中的每个对象
            for example in data:
                gt = extract_box(example["solution"])
                answer = example["answer"].lstrip('0') if example["answer"].startswith('0') else example["answer"]
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt": answer,
                })

    elif args.dataset == "DEEPSCALER":
        data_path = "data/deepscaler/deepscaler_eval_50.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                # gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["answer"],
                    "gt":example["answer"],
                })
        
    elif args.dataset == "MATH_train":
        data_path = "data/MATH/train.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":gt,
                })
    elif args.dataset in ["GSM", "GSMALL"]:
        if args.dataset == "GSMALL":
            data_path = "data/gsm/test.jsonl"
        else:
            data_path = "data/gsm/test_500.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    else:
        raise ValueError("Dataset not supported")
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    
       
    test_data = test_data[:]
    all = test_data
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.9, enable_lora=args.peft is not None, max_lora_rank=128, max_model_len=args.max_tokens+2000, tensor_parallel_size = args.tensor_parallel_size, dtype="bfloat16")

    tokenizer = model.llm_engine.tokenizer.tokenizer

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    # prefix = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
    prompts = []
    for i, example in enumerate(test_data):
        if args.completion:
            # print(example)
            prompt = example["question"]

        else:
            prompt =  prefix+"Question: " + example["question"].strip()+"\nAnswer: "
            # prompt = prefix + example["question"].strip() + "put your final answer within \\boxed{}\nAssistant: <think>"
            if args.use_chat_format:
                if "DeepSeek" in args.model_name_or_path :
                    messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if args.nothink:
                        prompt += "</think>"
                elif "Qwen" in args.model_name_or_path:
                    messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    if args.nothink:
                        prompt += "</think>"
                # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                #     prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])
    print(f"数据加载完成，样例：{prompts[0]}")

    batch_size = 100
    all_prompts = prompts
    print(f"总数据量: {len(test_data)}")

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as f:
        pass # 这会创建一个空文件或清空现有文件
    for batch_start in range(0, len(test_data), batch_size):

        test_data = all[batch_start:batch_start + batch_size]
        prompts = all_prompts[batch_start:batch_start + batch_size]

        if not args.logit_adjustment:

            sampling_params = SamplingParams(n=1,
                                            temperature=args.temp,
                                            top_p=args.top_p,
                                            max_tokens=args.max_tokens)
        else:
            print(args.logit_adjustment_tokens)
            if args.logit_adjustment_tokens[0]=="ours":
                adjustment_tokens = [
                    '(any', '-any', '.Any', '.any',  ':Any', ':any', '<Any', '<any', 'Any', '[Any', '_any', 'any', 'ĠAny','(other', '-other', '.Other', '.other', 'Other', '_other', 'other',  'ĠOther', 'Ġother',
                    'Hmm', 'ĠHmm', 'Ġhmm','Alternative', 'alternative', 'ĠAlternative','Ġalternative', 'Another', 'another', 'ĠAnother', 'Ġanother','(check', '-check', '.Check', '.check', '/check','Check', '_Check',  '_check','check', 'ĉCheck', 'ĉcheck', 'ĠCheck', 'Ġcheck',
                    'Alternatively', 'ĠAlternatively', 'Ġalternatively','"But', ',but', '-but','.But', '.but', 'But', '_but', 'but',  'âĢ¦but', 'âĢĶbut', 'âĢľBut', 'ĠBut', 'Ġbut','However', 'however', 'ĠHowever', 'Ġhowever',
                    '.Verify', '.verify', 'Verify', '_verify', 'verify', 'ĉverify', 'ĠVerify', 'Ġverify','Again', '_again', 'again',  'ĠAgain', 'Ġagain','(wait', '.Wait', '.wait', '/wait', 'Wait', '_wait','wait', 'ĉwait',  'ĠWait',  'Ġwait',
                    '"Now', '(now', '-now', '.Now', '.now', 'Now', '[now', '_now', 'now',  'ĉnow', 'ĠNow', 'Ġnow','Ah', 'ah', 'aha','ĠAh', 'Ġah','"Oh', 'Oh','oh','âĢľOh', 'ĠOh','Ġoh','.Maybe', 'Maybe', 'maybe', 'ĠMaybe', 'Ġmaybe'
                ]
                # adjustment_tokens = ["Wait","alternative","Alternatively"]
            else:
                adjustment_tokens = []
            print(f"接收到的调整token数量: {len(adjustment_tokens)}")
            print(f"要调整的token: {adjustment_tokens[:]}")

        
            
            # print_vocab_tokens(tokenizer)
            # search_vocab_tokens(tokenizer," wait")
            
            # return True
            # logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in adjustment_tokens])]).to("cuda")
            # adjustment_set = set(adjustment_tokens)
            # print(adjustment_tokens)  # 先转换为集合
            vocab = tokenizer.get_vocab()
            logit_adjustment_tokens = torch.LongTensor([
                vocab[token] for token in vocab.keys() 
                if token in adjustment_tokens  # 在集合中查找
            ])
            logit_bias_map = {token_id.item(): args.logit_adjustment_value for token_id in logit_adjustment_tokens}
            print(f"Adjusting {len(logit_adjustment_tokens)} tokens: {[tokenizer.decode([tid]) for tid in logit_adjustment_tokens]} token_ids: {[tid for tid in logit_adjustment_tokens]}")
            logit_adjustment_process = partial(logit_adjustment, tokenizer=tokenizer, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
            sampling_params = SamplingParams(n=1,
                                            temperature=args.temp,
                                            top_p=args.top_p,
                                            max_tokens=args.max_tokens,
                                            # logits_processors=[logit_adjustment_process]
                                            logit_bias=logit_bias_map
                                            )
        if args.peft is not None:
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
        else:
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

        result = []
        for output in outputs:
            attempts = []
            for ith_output in output.outputs:
                attempts.append(ith_output.text)
            result.append(attempts)
        

        outputs = [[trim_output(o) for o in output] for output in result]

        predictions = [{
            "prompt": prompt,
            "problem": example["question"],
            "answer": example["gt"],
            "solution":  example["answer"],
            "model_generation": output,
        } for example, output, prompt in zip(test_data, outputs, prompts)]

        with open(os.path.join(args.save_dir, "predictions.jsonl"), "a") as fout:
            for prediction in predictions:
                fout.write(json.dumps(prediction) + "\n")
    
        del prompts, outputs, result, predictions
        gc.collect()
        torch.cuda.empty_cache()
    
        print(f"剩余样本数量: {len(all) - (batch_start + batch_size)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--completion",
        default=False,
    )
    parser.add_argument(
        "--nothink",
        default=False,
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_path",
        "--data-path",
        type=str,
        default=None,
        help="Read a local JSONL file instead of a built-in dataset path. Use --completion True for thought prefixes.",
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        nargs="*",
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1
    )


    args = parser.parse_args()
    # print(f"要调整的词语有:{args.logit_adjustment_tokens}")
    # adjustment_tokens = [token.strip() for token in args.logit_adjustment_tokens[0].split(',')]
    # print(f"接收到的调整token数量: {len(adjustment_tokens)}")
    # print(f"要调整的token: {adjustment_tokens[:]}")
    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)

    main(args)
    eval_main(os.path.join(args.save_dir, "predictions.jsonl"),args.dataset, save=True, k=None, output_dir=args.save_dir)
    



        
