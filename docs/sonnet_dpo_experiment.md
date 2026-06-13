# Sonnet DPO 实验记录

## 实验目标

这一阶段的目标不是单纯刷高 chrF，而是沿着更实用的 Track B 改进 sonnet 生成质量：用 LLM-as-a-Judge 构造偏好数据，在已有 SFT checkpoint 的基础上做 DPO 训练，并用盲测式的 pairwise judge 评估 DPO 是否优于原来的 SFT 模型。

这个方向的核心判断是：sonnet generation 是 open-ended 任务，chrF 只能衡量和参考答案的字符 n-gram 重叠，不能充分反映风格、连贯性、押韵、语法流畅度和重复退化问题。因此本阶段把 LLM judge 的偏好结果作为主要实用指标，同时保留 chrF 作为辅助参考。

## 实验流程

1. 从已有的 SFT sonnet checkpoint 出发。
2. 对每个 prompt 生成多个候选 sonnet，使用不同的采样参数增加多样性。
3. 使用 DeepSeek 作为 LLM judge，在不提供 gold sonnet 的情况下对候选排序。
4. 根据 judge 排名导出 DPO pairs：第 1 名作为 `chosen`，靠后的候选作为 `rejected`。
5. 用 SFT checkpoint 初始化可训练 policy model，并用同一个 SFT checkpoint 作为冻结 reference model。
6. 在 DPO 数据上训练 policy model。
7. 对 SFT 和 DPO 的生成结果进行盲测 pairwise judge 评估。
8. 针对 DPO 输出中出现的重复退化问题，加入 decoding-time repetition control。

## 数据设置

- train split 的 judge 结果用于构造 DPO 训练数据。
- dev/test split 的 judge 结果用于评估，不用于构造训练偏好数据。
- DPO 数据以 `prompt / chosen / rejected` 形式保存，并保留 judge 分数和元数据，方便后续分析。
- `my_results/`、checkpoint、候选生成结果和 API 返回 JSONL 都属于运行产物，不进入 git commit。

## DPO 之前的 SFT 调试结果

在进入 LLM-as-a-Judge 和 DPO 之前，sonnet generation 已经完成了基础 SFT 训练和 dev chrF 网格搜索。最后一个 epoch 的日志保存在 `my_results/sonnet-training-log.txt`，关键结果如下：

| 指标 | 结果 |
| --- | ---: |
| Epoch | 9 |
| Train loss | 4.020 |
| 当前 epoch 最佳 dev chrF | 42.445 |
| 当前 epoch 最佳参数 | temperature=1.1, top_p=0.9 |
| 历史最佳 dev chrF | 42.748 |
| 历史最佳参数 | temperature=0.9, top_p=0.95 |

最后一个 epoch 的 dev 网格搜索结果如下：

| Temperature | Top-p | Dev chrF |
| ---: | ---: | ---: |
| 0.9 | 0.85 | 41.043 |
| 0.9 | 0.9 | 41.458 |
| 0.9 | 0.95 | 41.990 |
| 1.0 | 0.85 | 41.681 |
| 1.0 | 0.9 | 42.331 |
| 1.0 | 0.95 | 42.222 |
| 1.1 | 0.85 | 41.926 |
| 1.1 | 0.9 | 42.445 |
| 1.1 | 0.95 | 42.254 |

从 chrF 看，基础 SFT 已经能得到一个可以作为 baseline 的模型，而且采样参数对 dev chrF 有明显影响。`temperature=0.9, top_p=0.95` 是训练过程中记录到的历史最佳组合，`temperature=1.1, top_p=0.9` 是最后一个 epoch 当轮最好的组合。

从样例质量看，SFT 模型已经学到了一部分 Shakespeare sonnet 的局部风格，比如开头几行能自然接续 prompt，词汇和句法也有一定古典感。但问题也很明显：

- 长距离语义经常漂移，后半首和前三行 prompt 的主题连接会变弱。
- 有些句子语法不稳定，局部短语看起来像莎诗，但整句不一定通顺。
- 会出现不自然或离题的实体和意象，例如和 sonnet 风格不太协调的人名、动物或叙事片段。
- 部分输出存在重复、空行、格式松散或结尾收束不佳的问题。
- chrF 能反映和参考答案的重叠程度，但很难评价这些主观质量问题。

因此，DPO 之前的结论是：基础 SFT 已经完成了可用的初步实现，但如果目标是更像“人会觉得更好的 sonnet”，单靠 train loss 和 chrF 不够。后续引入 LLM-as-a-Judge 的主要动机，就是把风格、连贯性、流畅度、诗歌形式和重复退化这些 chrF 不容易覆盖的维度纳入优化和评估。

## DPO 训练机制

DPO 训练时，虽然 `chosen` 和 `rejected` 文本已经生成好了，但模型仍然要重新计算这些文本在当前 policy model 和 reference model 下的 log probability。

原因是 DPO 优化的不是“继续生成文本”，而是让 policy model 更偏好 judge 选中的回答。具体来说，每个样本都会计算：

- policy model 对 `chosen` continuation 的 log probability
- policy model 对 `rejected` continuation 的 log probability
- reference model 对 `chosen` continuation 的 log probability
- reference model 对 `rejected` continuation 的 log probability

然后 DPO loss 会优化 policy-reference 的偏好 margin，使 policy model 相比 reference model 更倾向于 `chosen` 而不是 `rejected`。

本阶段观察到前两个 epoch 的 DPO loss 明显下降，但 dev preference accuracy 已经比较早饱和，因此没有继续盲目增加 epoch。

| Epoch | Train DPO Loss | Train Preference Acc | Dev DPO Loss | Dev Policy Preference Acc |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.5633 | 0.846 | 0.3945 | 0.917 |
| 1 | 0.4141 | 0.836 | 0.2785 | 0.917 |

## 初版 DPO 评估结果

在未加入 repetition control 时，DPO 在 LLM judge 的 pairwise 评估中明显优于 SFT，但 dev chrF 略低于 SFT：

| Split | DPO Wins | SFT Wins | Ties | DPO Win Rate, Ties as Half | DPO Win Rate, Excluding Ties | SFT chrF | DPO chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 9 | 1 | 2 | 0.8333 | 0.9000 | 41.8070 | 41.2682 |
| Test | 9 | 2 | 1 | 0.7917 | 0.8182 | N/A | N/A |

这个结果说明 DPO 确实学到了 judge 偏好，但它并不必然提高 chrF。人工查看 SFT 获胜的样例后，主要问题集中在 DPO 有时会出现重复和退化，比如围绕 love、judgement、beauty 之类抽象词反复展开。

## 加入 Repetition Control 后

为了解决重复退化问题，generation 阶段加入了两个 decoding 控制项：

```bash
--repetition_penalty 1.1 --no_repeat_ngram_size 3
```

这不是重新训练模型，而是在生成时调整 token 采样分布：

- `repetition_penalty` 降低已经出现过的 token 再次被采样的概率。
- `no_repeat_ngram_size=3` 禁止生成已经出现过的 3-gram。

加入后，DPO 的明显获胜次数减少，平局变多，但整体更稳，且 dev chrF 反而超过了 SFT：

| Split | DPO Wins | SFT Wins | Ties | DPO Win Rate, Ties as Half | DPO Win Rate, Excluding Ties | SFT chrF | DPO chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 6 | 1 | 5 | 0.7083 | 0.8571 | 41.9863 | 42.3278 |
| Test | 5 | 0 | 7 | 0.7083 | 1.0000 | N/A | N/A |

## 结果解读

本阶段最重要的现象是：chrF 和 LLM judge preference 并不完全一致。

chrF 更像是“和参考答案相似程度”的指标，而 LLM judge 更关注生成文本本身是否像一首合理的 sonnet，包括风格、连贯性、语法、形式和重复程度。对于 open-ended sonnet generation 来说，后者更接近实际使用体验。

加入 repetition control 后，DPO 的优势从“经常明显赢 SFT”变成“多数情况下不差，部分情况下更好”。这会让 tie-as-half win rate 看起来没有提高，但 test 上 SFT 一次也没有赢，说明 DPO + repetition control 至少是一个更稳的版本。

dev 上 chrF 从 DPO 略低于 SFT，变成 DPO 高于 SFT，也缓解了“实用指标”和“量化指标”之间的冲突。

## 当前结论

这一阶段可以收束。当前 sonnet generation 的实用改进链路已经完成：

- LLM-as-a-Judge 生成偏好数据
- DPO pair 导出
- DPO 训练
- SFT vs DPO 盲测 pairwise 评估
- repetition penalty / no-repeat n-gram 解码控制
- 实验结果显示 DPO 相比 SFT 更符合 judge 偏好，并且 repetition control 后更加稳定

当前推荐的 DPO 评估和生成配置是：

```bash
--repetition_penalty 1.1 --no_repeat_ngram_size 3
```

后续如果继续改进，不应该再根据 test 结果调参。可以只在 dev 上尝试：

- 更好的候选生成策略
- 更稳定的 judge prompt
- 小范围 sweep DPO 的 `beta` 和 learning rate
- 对重复、押韵、行数结构分别做更细粒度的分析
