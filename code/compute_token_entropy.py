import argparse
import os
import torch
import json
import jsonlines
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from tqdm import tqdm

def load_model_and_tokenizer(model_path: str):
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用 bfloat16 节省一半显存
    # 使用 sdpa (Scaled Dot Product Attention) 优化长序列
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        trust_remote_code=True,
        attn_implementation="sdpa" 
    )
    model.eval()
    return model, tokenizer

def split_content_keep_separator(content, separator="\n\n"):
    """
    按照分隔符分割内容，并保留分隔符。
    """
    if "<think>\n" in content:
        prompt = content.split("<think>\n")[0] + "<think>\n"
    else:
        prompt = "" # 兼容没有think标签的情况
    
    remaining_content = content[len(prompt):]
    if not separator or separator not in remaining_content:
        return [prompt + remaining_content] if prompt + remaining_content else []
    
    escaped_sep = re.escape(separator)
    parts = re.split(f"({escaped_sep})", remaining_content)
    
    result = []
    current_part = prompt
    for part in parts:
        if part == separator:
            current_part += part
            result.append(current_part)
            current_part = ""
        else:
            current_part += part
            
    if current_part:
        result.append(current_part)
    return result

@torch.inference_mode()
def process_single_sample(model, tokenizer, full_text, adjustment_ids_tensor):
    """
    核心计算函数：对一整条文本进行一次推理，返回熵、特定概率和IDs
    """
    device = model.device
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    
    # 限制最大长度防止崩溃
    if input_ids.shape[1] > 14336:
        input_ids = input_ids[:, :14336]
        attention_mask = inputs["attention_mask"][:, :14336]
    else:
        attention_mask = inputs["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # 预测偏移：logit[i] 预测的是 input_ids[i+1]
    # 我们取 [0, seq_len-1] 的 logits 来对应 [1, seq_len] 的预测
    shift_logits = logits[:, :-1, :].contiguous()
    
    # 1. 计算熵 (使用 LogSumExp 技巧防止溢出，直接在 tensor 上操作)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).squeeze(0).cpu().float().numpy()
    
    # 2. 提取特定 token 的概率 (只选需要的列)
    # [1, seq_len-1, len(adj_ids)]
    spec_probs = probs[:, :, adjustment_ids_tensor].squeeze(0).cpu().float().numpy()
    
    # 清理大张量
    del logits, shift_logits, log_probs, probs, outputs
    # torch.cuda.empty_cache() # 如果显存极度紧张，取消此行注释，但会变慢
    
    return entropy, spec_probs, input_ids.squeeze(0).cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--separator", type=str, default="\n\n")
    parser.add_argument('--math_or_code', default="math", choices=["math", "code"])
    parser.add_argument('--base_path', type=str, default="/home/jwangxgroup/pengzhenyue/Underthinking/data/qwen3-8b/omni/Qwen3-8B-omni-level237-dataset.jsonl")
    parser.add_argument('--epoch_start', type=int, default=1)
    parser.add_argument('--base_flag', default=False, action='store_true')
    parser.add_argument('--second_path', type=str, default="/home/jwangxgroup/pengzhenyue/SEAL-main/results/MATH500/DeepSeek-R1-Distill-Qwen-1.5B/steer-from-MATH/baseline_10000_seal_vector_500_500_layer_20_transition_reflection_steervec/coef_-1.0_remove_bos/math_eval_rewrite.jsonl")
    parser.add_argument('--epoch_end', type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    # args = parser.parse_args()
    base_path = args.base_path
    
    # 准备调整 token 的 ID
    adjustment_tokens = [
        '(any', '-any', '.Any', '.any', ':Any', ':any', '<Any', '<any', 'Any', '[Any', '_any', 'any', 'ĠAny','(other', '-other', '.Other', '.other', 'Other', '_other', 'other', 'ĠOther', 'Ġother',
        'Hmm', 'ĠHmm', 'Ġhmm','Alternative', 'alternative', 'ĠAlternative','Ġalternative', 'Another', 'another', 'ĠAnother', 'Ġanother','(check', '-check', '.Check', '.check', '/check','Check', '_Check', '_check','check', 'ĉCheck', 'ĉcheck', 'ĠCheck', 'Ġcheck',
        'Alternatively', 'ĠAlternatively', 'Ġalternatively','"But', ',but', '-but','.But', '.but', 'But', '_but', 'but', 'âĢ¦but', 'âĢĶbut', 'âĢľBut', 'ĠBut', 'Ġbut','However', 'however', 'ĠHowever', 'Ġhowever',
        '.Verify', '.verify', 'Verify', '_verify', 'verify', 'ĉverify', 'ĠVerify', 'Ġverify','Again', '_again', 'again', 'ĠAgain', 'Ġagain','(wait', '.Wait', '.wait', '/wait', 'Wait', '_wait','wait', 'ĉwait', 'ĠWait', 'Ġwait',
        '"Now', '(now', '-now', '.Now', '.now', 'Now', '[now', '_now', 'now', 'ĉnow', 'ĠNow', 'Ġnow','Ah', 'ah', 'aha','ĠAh', 'Ġah','"Oh', 'Oh','oh','âĢľOh', 'ĠOh','Ġoh','.Maybe', 'Maybe', 'maybe', 'ĠMaybe', 'Ġmaybe'
    ]
    adj_token_ids = list(set([tokenizer.convert_tokens_to_ids(t) for t in adjustment_tokens if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id]))
    adj_ids_tensor = torch.tensor(adj_token_ids, dtype=torch.long, device=model.device)


    for epoch in range(args.epoch_start , args.epoch_end + 1):
        if args.base_flag:
            if args.math_or_code == "math":
                json_file_path = os.path.join(base_path, f"epoch{epoch}", args.second_path, "math_eval_rewrite.jsonl")
            elif args.math_or_code == "code":
                json_file_path = os.path.join(base_path, f"epoch{epoch}", args.second_path, "code_eval.jsonl")
        else:
            json_file_path = args.dataset_path


    # Load dataset
    print(f"Loading dataset from {json_file_path}...")
    dataset = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            dataset.append(item)
    
    dataset = dataset[:] # 限制测试数量
    
    if args.output_path is None:
        input_dir, input_filename = os.path.split(json_file_path)
        output_filename = f"entropy-{input_filename}"
        args.output_path = os.path.join(input_dir, output_filename)
    
    # 清空/创建输出文件
    with open(args.output_path, "w") as f:
        pass 
    
    separator = args.separator.replace("\\n", "\n")

    
    # 流式处理并写入
    with jsonlines.open(args.output_path, mode='a') as writer:
        for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
            # 1. 构建全文本
            try:
                if args.math_or_code == "math":
                    gen = item.get("model_generation", [""])[0]
                    content_to_analyze = gen.split("</think>")[0] if "</think>" in gen else gen
                    full_content = item["prompt"] + content_to_analyze
                    problem = item.get("problem", "")
                    result = item.get("graded_list", [""])[0] if "graded_list" in item else ""
                else: # code
                    gen = item.get("output_list", [""])[0]
                    content_to_analyze = gen.split("</think>")[0] if "</think>" in gen else gen
                    full_content = item["prompt"] + content_to_analyze
                    problem = item.get("question_content", "")
                    result = item.get("graded_list", [""])[0]

                # 2. 推理获取整条文本的指标
                all_entropy, all_spec_probs, all_ids = process_single_sample(
                    model, tokenizer, full_content, adj_ids_tensor
                )

                # 3. 按段落切分并 Slicing
                segments = split_content_keep_separator(full_content, separator)
                current_token_pos = 0
                
                sample_output_buffer = []
                for p_idx, seg in enumerate(segments):
                    seg_ids = tokenizer(seg, add_special_tokens=False)["input_ids"]
                    seg_len = len(seg_ids)
                    
                    # 熵的切片逻辑：
                    # logit[i] 预测的是 input_id[i+1]。
                    # 对于当前段落，我们要的是能预测出该段落内 token 的那些 logits。
                    # 起点：上一个段落结束的位置 (current_token_pos - 1)
                    # 终点：当前段落结束的位置的前一个 (current_token_pos + seg_len - 1)
                    start_idx = max(0, current_token_pos - 1)
                    end_idx = current_token_pos + seg_len - 1
                    
                    p_entropy = all_entropy[start_idx : end_idx].tolist()
                    # p_spec_probs = all_spec_probs[start_idx : end_idx].tolist() # 如需概率可开启
                    p_ids = all_ids[current_token_pos : current_token_pos + seg_len].tolist()
                    
                    sample_output_buffer.append({
                        "original_content_index": idx,
                        "problem": problem,
                        "paragraph_index": p_idx,
                        "paragraph_text": seg,
                        "token_entropies": p_entropy,
                        "input_ids": p_ids,
                        "result": result
                    })
                    current_token_pos += seg_len
                
                writer.write_all(sample_output_buffer)

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

    print(f"Done. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()