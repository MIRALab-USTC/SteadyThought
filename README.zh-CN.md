<h1 align="center">🧠 SteadyThought: Mitigating LLM Under-Thinking<br>via Thought-Level Preference Optimization</h1>

<p align="center"><em>通过 thought 级偏好优化，减少不必要的思路切换。</em></p>

<p align="center">
  <img src="assets/overview.svg" width="100%" alt="SteadyThought：识别 thought、合成轨迹、构造偏好数据，并接入 SimPO 训练">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Conference-NeurIPS%202026-b31b1b?style=for-the-badge" alt="NeurIPS 2026">
  <a href="https://github.com/MIRALab-USTC/SteadyThought"><img src="https://img.shields.io/badge/Code-SteadyThought-111827?style=for-the-badge&amp;logo=github" alt="Code: SteadyThought"></a>
</p>

<p align="center"><a href="README.md">English</a></p>

---

## 📚 概览

**SteadyThought** 利用 token 熵识别 thought 边界，合成受控续写，将正确的合成续写与原始续写组成 thought 级偏好数据。产出的数据可接入 LLaMA-Factory，使用 SimPO 进行训练。

## 安装环境

在仓库根目录执行：

```bash
conda create -n steadythought python=3.10 -y
conda activate steadythought
python -m pip install -r requirements.txt
```

## 构造 STPO 训练数据

运行前修改 `scripts/01_*.sh`–`03_*.sh` 中的模型路径、数据路径和 GPU 编号，保持各脚本的模型和数据集设置一致。默认使用 **DeepSeek-R1-Distill-Qwen-1.5B** 和 **Omni-MATH level 2/3/7**。

```bash
# 1. 生成原回答、计算熵并分割 thoughts。
bash scripts/01_identify_thoughts.sh

# 2. 从各个 thought 持续续写，并评估正确性。
bash scripts/02_synthesize_trajectories.sh

# 3. 检查输入对齐关系，构造偏好数据对。
bash scripts/03_build_stpo_dataset.sh
```

也可用 `bash scripts/run_all.sh` 一次运行全部阶段。

最终输出：`outputs/<model>/<dataset>/<model>-<dataset>-stpo.jsonl`。

## 使用 SimPO 训练

产出的数据可直接作为 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的 Alpaca 格式偏好数据，无需转换。按其[数据集说明](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md#preference-dataset)，在 `dataset_info.json` 中注册为 `stpo`，设置 `ranking: true`，将 `prompt` 映射到 `instruction`，`chosen` 和 `rejected` 映射到同名字段。

单独安装 LLaMA-Factory，在完整训练配置中加入以下关键设置（[SimPO 参数说明](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html#rlhf)）：

```yaml
stage: dpo
pref_loss: simpo
dataset: stpo
template: empty
cutoff_len: 6144
```

`instruction` 已包含格式化后的问题和 thought 前缀，`chosen`、`rejected` 是后续续写，因此使用 `template: empty`，避免再次套用聊天模板。上述脚本只制作数据，训练在 LLaMA-Factory 中执行。
