import os
import json
from tqdm import tqdm, trange
from eval_math_rule.evaluation.grader import math_equal
from eval_math_rule.evaluation.parser import extract_answer, parse_ground_truth, strip_string
from collections import Counter
import jsonlines
import multiprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import queue
def bool2int(s: str) -> int:
    """true->1, false->0，忽略大小写"""
    return 1 if s.lower() == 'true' else 0

def math_equal_with_timeout(pred, gt_ans, timeout):
    def target(result_queue):
        try:
            result_queue.put(math_equal(pred, gt_ans))
        except Exception as e:
            result_queue.put(e)


    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(result_queue,))
    process.start()

    process.join(timeout)

    if process.is_alive():
        print(f"Timeout occurred for prediction: {pred}")
        process.terminate()
        process.join()    
        return False

   
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        print("Result queue timed out")
        return False

    if isinstance(result, Exception):
        print(f"Error occurred: {result}")
        return False

    return result



def parallel_math_equal(all_pred, gt_ans, timeout=20):
    results = []
    for pred in all_pred:
        results.append(math_equal_with_timeout(pred, gt_ans, timeout))
    return results



def main(res_path, data_name, save=False, k=None, output_dir=None):
    # args = parse_args()
    with open(res_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    for example in tqdm(data):
        # gt_cot, gt = parse_ground_truth(example, data_name="omni-math")
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]
        gt_cot = example["answer"]
        if data_name in ["GSM", "GSMALL"]:
            gt_ans = extract_answer(gt_cot, data_name=data_name,example=example)
        else:
            gt_ans = extract_answer(gt_cot, data_name="omni-math",example=example)
        gt_cot = str(gt_cot).strip()
        gt_ans = strip_string(gt_ans, skip_unit=False)
        all_pred = [extract_answer(p, data_name=data_name,example=example) for p in example["model_generation"]]
        all_pred = [strip_string(p, skip_unit=False) for p in all_pred]
        # all_eval = [math_equal(p, gt_ans) for p in all_pred]
        all_eval = parallel_math_equal(all_pred, gt_ans, timeout=5)
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"]) if "boxed" in o]
        if len(effective_pred) == 0:
            effective_pred = all_pred
        counter = Counter(effective_pred)
        pred = counter.most_common(1)[0][0]
        index = all_pred.index(pred)
        eval = all_eval[index]
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = eval
        example["mv_index"] = index

    acc = sum([example["mv_eval"] for example in data]) / len(data)
    print(f"Accuracy: {acc:.3f}")
    
    if save:
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        
        metric_file= os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": acc}, f)

        data = [] 
        with open(out_file, 'r') as infile:
            for line in infile:
                item = json.loads(line)
                data.append(item)
        out_file_new = os.path.join(output_dir, "math_eval_rewrite.jsonl")
        f_output = jsonlines.Writer(open(out_file_new, "w", encoding="utf-8"))
        for item in data:
            f_output.write(item)


def stra_main(res_path, data_name, save=False, k=None, output_dir=None):
    with open(res_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    y_true, y_pred = [], []          # 二分类 0/1 标签
    for example in tqdm(data, desc='eval'):
        # 1. 模型生成
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]

        # 2. 真实答案
        gt_ans = extract_answer(example["answer"], data_name=data_name,example=example)
        gt_ans = strip_string(gt_ans)
        y_true.append(bool2int(gt_ans))

        # 3. 每条生成的答案
        all_pred = [strip_string(extract_answer(p, data_name=data_name,example=example))
                    for p in example["model_generation"]]
        all_eval = parallel_math_equal(all_pred, gt_ans, timeout=5)

        # 4. 投票：优先用带 \boxed 的，否则用全部
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"])
                          if "boxed" in o]
        if not effective_pred:
            effective_pred = all_pred
        counter = Counter(effective_pred)
        mv_ans, _ = counter.most_common(1)[0]   # 投票结果
        y_pred.append(bool2int(mv_ans))

        # 5. 写回 example，方便后续 debug
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"]  = mv_ans
        example["mv_eval"]  = (mv_ans == gt_ans)

    # 6. 指标
    acc     = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1      = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    # 7. 保存
    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w", encoding='utf-8') as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        metric_file = os.path.join(output_dir, "metrics.json")
        json.dump({"acc": acc,
                   "precision": precision,
                   "recall": recall,
                   "f1": f1},
                  open(metric_file, "w", encoding='utf-8'), indent=2)

        data = [] 
        with open(out_file, 'r') as infile:
            for line in infile:
                item = json.loads(line)
                data.append(item)
        out_file_new = os.path.join(output_dir, "math_eval_rewrite.jsonl")
        f_output = jsonlines.Writer(open(out_file_new, "w", encoding="utf-8"))
        for item in data:
            f_output.write(item)

def musr_main(res_path, data_name, save=False, k=None, output_dir=None):
    # args = parse_args()
    with open(res_path, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    for example in tqdm(data):
        # gt_cot, gt = parse_ground_truth(example, data_name="omni-math")
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]
        if k is not None:
            example["model_generation"] = example["model_generation"][:k]
        gt_cot = example["answer"]
        if data_name in ["GSM", "GSMALL"]:
            gt_ans = extract_answer(gt_cot, data_name=data_name,example=example)
        else:
            gt_ans = extract_answer(gt_cot, data_name=data_name,example=example)
        gt_cot = str(gt_cot).strip()
        gt_ans = strip_string(gt_ans, skip_unit=False)
        all_pred = [extract_answer(p, data_name=data_name,example=example) for p in example["model_generation"]]
        all_pred = [strip_string(p, skip_unit=False) for p in all_pred]
        # all_eval = [math_equal(p, gt_ans) for p in all_pred]
        all_eval = parallel_math_equal(all_pred, gt_ans, timeout=5)
        effective_pred = [p for p, o in zip(all_pred, example["model_generation"]) if "boxed" in o]
        if len(effective_pred) == 0:
            effective_pred = all_pred
        counter = Counter(effective_pred)
        pred = counter.most_common(1)[0][0]
        index = all_pred.index(pred)
        eval = all_eval[index]
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["mv_pred"] = pred
        example["mv_eval"] = eval
        example["mv_index"] = index

    acc = sum([example["mv_eval"] for example in data]) / len(data)
    print(f"Accuracy: {acc:.3f}")
    
    if save:
        out_file = os.path.join(output_dir, "math_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        
        metric_file= os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump({"acc": acc}, f)
        data = [] 
        with open(out_file, 'r') as infile:
            for line in infile:
                item = json.loads(line)
                data.append(item)
        out_file_new = os.path.join(output_dir, "math_eval_rewrite.jsonl")
        f_output = jsonlines.Writer(open(out_file_new, "w", encoding="utf-8"))
        for item in data:
            f_output.write(item)