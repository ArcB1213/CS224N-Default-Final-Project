# GPT-2 Sonnet Generation with LLM-as-a-Judge & DPO

基于 Stanford CS224N Default Final Project 的课程框架，在十四行诗生成任务上扩展 **SFT → 候选生成 → LLM 偏好标注 → DPO → 盲测评估** 的后训练流程。

本项目重点展示偏好数据构造、训练目标实现、评测设计与生成退化问题分析。

[完整实验记录](docs/sonnet_dpo_experiment.md)

## 项目来源与贡献边界

| 范围 | 内容 |
| --- | --- |
| 课程提供的基础 | 项目框架、任务与数据设置、部分基础代码 |
| 课程要求 | GPT-2 关键模块与 AdamW 等补全，情感分类检查、复述检测、十四行诗生成及下游任务扩展 |
| 本仓库额外实现的具体扩展 | LLM Judge 候选排序与偏好对导出、DPO 训练、SFT/DPO 随机 A/B 盲测、重复控制及实验分析 |


## 技术贡献

### 偏好数据构造与 LLM-as-a-Judge

入口：[llm_judge_sonnet.py](llm_judge_sonnet.py)。

- 以诗歌前三行为 prompt，使用不同 temperature / top-p 组合生成候选，默认每个 prompt 生成 8 个候选。
- 调用 DeepSeek 对形式、风格、连贯性、流畅度、诗歌结构及重复退化进行评分与排序，不向 Judge 提供参考诗歌。
- 将排名第一的 continuation 与排名末尾最多 3 个候选配对，导出 `prompt / chosen / rejected` JSONL。
- 保留候选、采样参数、Judge 请求、原始响应及偏好元数据；校验排名完整性，提供请求重试和失败记录。
- 仅使用 Train 偏好对更新参数；Dev 偏好对用于验证，Test 用于生成结果评估。

### DPO 训练

入口：[dpo_train_sonnet.py](dpo_train_sonnet.py)。

- 使用同一个 SFT checkpoint 初始化可训练 policy 与冻结 reference model。
- 对 chosen / rejected 重新计算序列 log probability，通过 loss mask 排除 prompt 和 padding。
- 实现基于 policy/reference 对数概率差的 DPO loss，支持 beta、学习率和梯度裁剪配置。
- 记录训练/验证损失与偏好准确率，保存最新 checkpoint，并在提供 Dev 数据时按 Dev loss 保存最佳 checkpoint。

训练日志中的偏好准确率，是固定偏好对上
`log P(chosen) > log P(rejected)` 的比例，与生成结果的 Judge 胜率不同。

### 盲测评估与重复控制

入口：[evaluate_dpo_sonnet.py](evaluate_dpo_sonnet.py)、
[sonnet_generation.py](sonnet_generation.py)。

- SFT 与 DPO 使用相同 prompt、采样参数及对应随机种子生成结果。
- 随机分配 A/B 标签，隐藏模型身份，由 Judge 输出胜、负或平局并记录理由。
- 同时报告胜负平计数、平局计半分的得分率，以及 Dev chrF。
- 针对重复退化加入 `repetition_penalty=1.1` 与
  `no_repeat_ngram_size=3`。这是解码调整，不是重新训练；
  评测时双方均使用相同设置。

## 实验结果

以下数值来自仓库中的实验记录。

| 解码设置 | Split | 样本数 | DPO 胜 | SFT 胜 | 平局 | 平局计半分 | SFT chrF | DPO chrF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无额外重复控制 | Dev | 12 | 9 | 1 | 2 | 83.33% | 41.8070 | 41.2682 |
| 无额外重复控制 | Test | 12 | 9 | 2 | 1 | 79.17% | — | — |
| 双方均启用重复控制 | Dev | 12 | 6 | 1 | 5 | 70.83% | 41.9863 | 42.3278 |
| 双方均启用重复控制 | Test | 12 | **5** | **0** | **7** | **70.83%** | — | — |

平局计半分的得分率为：
`(DPO 胜 + 0.5 × 平局) / 总样本数`。

最终 Test 的严格获胜比例为 5/12（41.67%）；
排除平局后的 5/5 不应表述为整体“100% 胜率”。

本次小样本实验中，DPO 获得了更多 Judge 偏好；
chrF 与 Judge 偏好并不完全一致。重复控制后平局增多，
Dev chrF 高于对应 SFT，但不能据此断言整体质量或稳定性已显著提升。


## 运行与复现入口

环境入口：[setup.sh](setup.sh)、[env.yml](env.yml)。
先准备兼容环境，并在环境变量中配置 `DEEPSEEK_API_KEY`。

| 阶段 | 入口 | 主要产物 |
| --- | --- | --- |
| SFT 与 Dev 采样参数搜索 | `sonnet_generation.py` | SFT checkpoint、生成文本 |
| 候选生成与偏好标注 | `llm_judge_sonnet.py` | 候选、评审记录、偏好对 JSONL |
| DPO 训练 | `dpo_train_sonnet.py` | 最新及最佳 DPO checkpoint |
| SFT / DPO 对照评估 | `evaluate_dpo_sonnet.py` | 生成结果、逐条判断、汇总指标 |

已有 SFT checkpoint 时，可按以下 Bash 示例运行：

```bash
# 替换为实际 SFT checkpoint 路径
SFT_CKPT="/path/to/sft-checkpoint.pt"

# 构造 Train 偏好数据与 Dev 验证数据
python llm_judge_sonnet.py --use_gpu \
  --checkpoint_path "$SFT_CKPT" --splits train,dev \
  --candidate_count 8 \
  --out_dir my_results/llm_judge

# DPO 训练
python dpo_train_sonnet.py --use_gpu \
  --checkpoint_path "$SFT_CKPT" \
  --dpo_train_path my_results/llm_judge/dpo_pairs_train.jsonl \
  --dpo_dev_path my_results/llm_judge/dpo_pairs_dev.jsonl \
  --epochs 2 --out_dir my_results/dpo_sonnet

# 先评估 Dev
python evaluate_dpo_sonnet.py --use_gpu \
  --sft_checkpoint_path "$SFT_CKPT" \
  --dpo_checkpoint_path my_results/dpo_sonnet/best_dpo-sonnet.pt \
  --splits dev --temperature 0.9 --top_p 0.95 \
  --repetition_penalty 1.1 --no_repeat_ngram_size 3 \
  --out_dir my_results/dpo_eval_dev
```

配置冻结后，将 `--splits dev` 改为 `--splits test`，
并使用新的输出目录进行测试。

Judge 模型可通过
`--judge_model` 指定，更多参数见各脚本的 `--help`。

# CS 224N Default Final Project: Build GPT-2

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course
website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the GPT-2 model to
better understand its architecture.
In the second part, you will use the token embeddings produced by your GPT-2 model on two downstream tasks: paraphrase
detection and sonnet generation. You will implement extensions to improve your model's performance on these tasks.

In broad strokes, Part 1 of this project targets:

* modules/attention.py: Missing code blocks.
* modules/gpt2_layer.py: Missing code blocks.
* models/gpt2.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

To test Part 1, you will run:

* `optimizer_test.py`: To test your implementation of `optimizer.py`.
* `sanity_check.py`: To test your implementation of GPT models.
* `classifier.py` : To perform sentiment classification using your models.

In Part 2 of this project, you will use GPT2 (via cloze-style classification) detect if one sentence is a paraphrase of 
another as well as generate sonnets via autoregressive language modeling.  

To test Part 2, you will run:

* `paraphrase_detection.py`: To perform paraphrase detection. 
* `sonnet_generation.py`: To perform sonnet generation.

Important: Adjust training hyperparameters, particularly batch size, according to your GPU's specifications to optimize performance and prevent out-of-memory errors.

## Pre-testing instructions

While there are missing code blocks that you need to implement in both of these files, the main focus of this second 
part are the extensions: how you modify your GPT2 model to improve its ability to determine if one sentence is a 
paraphrase of another as well as its ability to generate sonnets. 

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).
