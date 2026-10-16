<h1 align="center">🧠 SteadyThought: Mitigating LLM Under-Thinking<br>via Thought-Level Preference Optimization</h1>

<p align="center"><em>Reducing unnecessary thought switching with thought-level preference optimization.</em></p>

<p align="center">
  <img src="assets/overview.svg" width="100%" alt="SteadyThought: thought identification, trajectory synthesis, preference construction, and SimPO training">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Conference-NeurIPS%202026-b31b1b?style=for-the-badge" alt="NeurIPS 2026">
  <a href="https://github.com/MIRALab-USTC/SteadyThought"><img src="https://img.shields.io/badge/Code-SteadyThought-111827?style=for-the-badge&amp;logo=github" alt="Code: SteadyThought"></a>
</p>

<p align="center"><a href="README.zh-CN.md">简体中文</a></p>

---

## 📚 Overview

**SteadyThought** identifies thought boundaries using token entropy, synthesizes controlled continuations, and pairs correct synthesized continuations with the original continuations to build thought-level preference data. The resulting dataset can be used for SimPO training in LLaMA-Factory.

## Installation

Run commands from the repository root:

```bash
conda create -n steadythought python=3.10 -y
conda activate steadythought
python -m pip install -r requirements.txt
```

## Construct STPO data

Edit the model path, dataset path and GPU IDs in `scripts/01_*.sh`–`03_*.sh` before running. Keep model and dataset settings consistent. Defaults use **DeepSeek-R1-Distill-Qwen-1.5B** and **Omni-MATH levels 2/3/7**.

```bash
# 1. Generate responses, compute entropy, and segment thoughts.
bash scripts/01_identify_thoughts.sh

# 2. Generate and evaluate controlled continuations from each thought.
bash scripts/02_synthesize_trajectories.sh

# 3. Check input alignment and build preference pairs.
bash scripts/03_build_stpo_dataset.sh
```

Or run all stages with `bash scripts/run_all.sh`. 

Final output: `outputs/<model>/<dataset>/<model>-<dataset>-stpo.jsonl`.

## Train with SimPO

The output can be used directly as an Alpaca-format preference dataset in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), without data conversion. Following its [dataset guide](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md#preference-dataset), register the file as `stpo` in `dataset_info.json`, set `ranking: true`, and map `prompt` to `instruction`, `chosen` to `chosen`, and `rejected` to `rejected`.

Install LLaMA-Factory separately and add these key settings to your complete training configuration ([SimPO options](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html#rlhf)):

```yaml
stage: dpo
pref_loss: simpo
dataset: stpo
template: empty
cutoff_len: 6144
```

Use `template: empty` because `instruction` already contains the formatted prompt and thought prefix; `chosen` and `rejected` are continuations. Do not wrap the prefix in another chat template. The scripts above construct data only; training runs in LLaMA-Factory.
