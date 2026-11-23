# 十四行诗后训练实验

2026-11-23。训练和评测在一张 NVIDIA A800 40 GB 上完成。服务器项目目录为 `/root/cs224n-sonnet`，独立 Python 环境为 `runtime/`。本轮使用 Python 3.12、PyTorch 2.5.1 + CUDA 12.4、Transformers 4.46.3 和 SacreBLEU 2.5.1。完整依赖版本见本地归档 `environment.txt`。

**推荐使用 GPT‑2 medium 的换行加权 SFT checkpoint。** 与历史 small SFT 相比，同一组评测下的严格格式从 23/36 提高到 34/36，整诗 chrF 从 42.39 到 42.67，续写 chrF 基本持平。

## 实验设置

- 训练集有 131 首。格式感知 SFT 使用其中恰好 14 个非空行的 129 首，并把显式 EOS 纳入损失。普通 SFT 使用全部 131 首。
- 评测使用课程 Dev 的 12 个前三行提示（ID 132–143）。模型共用温度 0.9、top‑p 0.95、最多 220 个新增 token、最多 14 个非空行和同一组每首诗的随机种子。
- 整诗 chrF 用于和课程口径对齐。续写 chrF 从参考和生成文本中去掉相同的前三行。严格格式要求输出有 14 个非空行，而且续写从新的一行开始。
- 训练 GPT‑2 medium 使用完整模型、BF16 前向、batch size 8 和学习率 5e‑6。续训读取前一阶段的模型和优化器状态。每段续训按相同的 CUDA 种子初始化，比较以实际保存的 checkpoint 为准。

## Medium SFT

| 模型 | 整诗 chrF | 续写 chrF | 14 个非空行 | 严格格式 |
| --- | ---: | ---: | ---: | ---: |
| GPT‑2 medium 预训练模型 | 41.73 | 27.51 | 27/36 | 18/36 |
| 历史 small SFT | 42.39 | 27.60 | 36/36 | 23/36 |
| Medium 普通 SFT，10 epochs | 42.30 | 27.25 | 36/36 | 28/36 |
| Medium 格式感知 SFT，10 epochs | 42.40 | 27.23 | 35/36 | 29/36 |
| Medium 普通 SFT，15 epochs | 42.68 | 27.65 | 36/36 | 28/36 |
| Medium 普通 SFT，20 epochs | 42.27 | 27.06 | 36/36 | 29/36 |

三个生成种子是 11711、15823、19489。每个种子各算一次语料 chrF，表中取三次的均值。15-epoch checkpoint 在三个种子的整诗 chrF 都高于历史 small SFT，平均高 0.29 分，续写 chrF 平均高 0.05。它也比同骨干预训练模型高 0.95 整诗 chrF，严格格式多 10/36。续训到 20 个 epoch 后续写分数回落，训练损失继续下降。格式感知 SFT 的严格格式略有改善，续写 chrF 与普通 10-epoch 组接近。

本地评测记录为 `medium-pilot-11711.json` 和三个 `medium-continuation-<seed>.json`。

## 修复偏好边界后的 small DPO

在 393 对历史训练偏好里恢复原始续写前缀，保留了 237 个 `chosen` 和 208 个 `rejected` 的前导换行。从历史 small SFT 开始训练 2 个 epoch，batch size 2、学习率 1e‑6、beta 0.1、最大长度 256。训练损失从 0.5332 降至 0.3396。

| 模型 | 整诗 chrF | 续写 chrF | 续写另起一行 | 有重复续写行的输出 |
| --- | ---: | ---: | ---: | ---: |
| 历史 small SFT | 42.34 | 27.50 | 14/24 | 0/24 |
| 历史 small DPO | 41.80 | 26.54 | 17/24 | 0/24 |
| 修复边界后重训的 small DPO | 39.89 | 23.60 | 20/24 | 3/24 |

表中 chrF 是随机种子 11711、15823 各自语料分数的均值。三个模型都达到了 24/24 的 14 行数要求。修复提高了另起一行的比例，但这组 DPO 的 chrF 更低，且出现了重复续写行。本轮保留边界修复，模型选择依据后续 SFT 结果。本地评测记录为 `dpo-11711.json` 和 `dpo-15823.json`。

## 首个续写 token 加权

131 首训练诗的首个有效续写 token 全部是换行。两个分支都从 15-epoch 普通 SFT checkpoint 出发，使用同一训练配置续训 3 个 epoch。加权组把该 token 的损失权重设为 16，其他有效 token 权重为 1。损失按有效权重总和归一化。

| 模型 | 整诗 chrF | 续写 chrF | 14 个非空行 | 严格格式 | 有重复续写行的输出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 18-epoch 对照，权重 1 | 42.66 | 27.62 | 36/36 | 29/36 | 1/36 |
| 18-epoch 换行加权，权重 16 | 42.67 | 27.63 | 36/36 | 34/36 | 0/36 |

在 12 个 Dev 提示上，温度缩放和 top‑p 过滤前，模型赋给首个换行 token 的平均概率从 **80.77%** 提高到 **90.70%**，其中 12/12 个提示的概率提高。实际生成的严格格式从 29/36 提到 34/36，整诗和续写 chrF 与同起点对照接近。与历史 small SFT 相比，严格格式多 11/36，整诗 chrF 高 0.28，续写 chrF 高 0.03。

本地评测记录为三个 `boundary-<seed>.json`，逐提示换行概率见 `newline-probability.json`。

## 生成文本抽读

在两个相同采样种子下各抽取 6 组，将同一提示的 Medium SFT、small SFT、历史 small DPO 和重训 small DPO 隐去模型名后比较。单一模型评阅在 12 组中选了 10 次 Medium SFT，另外两组分别选了历史 DPO 和重训 DPO。Medium SFT 更常沿着前三行的意象和矛盾继续写。四个模型的持续交替押韵、节拍和尾联收束仍不稳定。该对照同时涉及骨干大小和训练方式。

下一轮考虑直接从 Medium SFT 继续做 DPO，再用相同提示和解码设置比较同骨干的两份权重，并单独检查押韵、节奏、续写连贯性和尾联。

## 复现与产物

- 推荐 checkpoint 为服务器上的 `/root/cs224n-sonnet/experiments/medium-boundary-weight/medium-first-token-x16/17_sft.pt`。文件名中的 17 是从零开始的 epoch 序号，对应 18 个训练 epoch。
- 服务器 `logs/` 保留训练和安装日志。`experiments/` 记录每个条件的 checkpoint、逐首生成文本和指标。
- 本地 `my_results/post_training_2026-11-23/` 保留实验 JSON、日志和实际运行脚本。脚本分别为 `run_medium_comparison.sh`、`run_dpo_boundary.sh`、`run_medium_continuation.sh` 和 `run_boundary_weight.sh`。
- 服务器通过了 `pip check`、AdamW 数值测试和小模型的 Hugging Face 输出对齐测试。31 个模型与种子记录组的 chrF 已从逐首文本独立复算，Dev ID、随机种子和文本边界也已核对。

从服务器项目目录运行以下命令可复测推荐 checkpoint。

```bash
ROOT=/root/cs224n-sonnet
cd "$ROOT/code"
"$ROOT/runtime/bin/python" evaluate_sonnet_dev.py --use_gpu --seed 11711 \
  --checkpoint medium-boundary16 "$ROOT/experiments/medium-boundary-weight/medium-first-token-x16/17_sft.pt" \
  --out "$ROOT/experiments/medium-boundary-weight/dev-recheck-11711.json"
```

本轮比较覆盖 12 个独立 Dev 提示。三个种子用于观察采样波动。chrF 衡量与参考文本的字符重叠，严格格式衡量行数和续写边界。
