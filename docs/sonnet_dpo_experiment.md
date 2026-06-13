# Sonnet DPO Experiment Notes

## Objective

This stage explores a practical Track B objective for sonnet generation: improve human-preference quality rather than optimizing only chrF. The workflow uses an LLM-as-a-Judge pipeline to create preference data, trains a DPO model from the existing SFT checkpoint, and evaluates DPO against the SFT baseline with blind pairwise judging.

## Pipeline

1. Start from the existing SFT sonnet checkpoint.
2. Generate multiple candidate sonnets per prompt with varied sampling parameters.
3. Use DeepSeek as an LLM judge to rank candidates without seeing the gold sonnet.
4. Export DPO pairs from judge rankings: top candidate as `chosen`, bottom candidates as `rejected`.
5. Train a DPO model with the SFT checkpoint as both initial policy and frozen reference model.
6. Compare SFT and DPO generations with blind pairwise LLM judging.
7. Add decoding-time repetition control to reduce degenerate loops.

## Data

- Train preference data comes from generated candidates for training-set prompts.
- Dev and test judge data are used for evaluation, not for tuning DPO training pairs.
- DPO pairs are stored as `prompt`, `chosen`, and `rejected` records, with judge metadata preserved for analysis.
- The generation artifacts and model checkpoints are treated as run outputs and are not committed to git.

## DPO Training

The DPO trainer computes completion log probabilities for `chosen` and `rejected` continuations under both models:

- policy model: trainable DPO model
- reference model: frozen SFT model

The optimized objective is the standard DPO loss over the policy-reference preference margin. In the observed run, DPO loss decreased over the first two epochs while preference accuracy was already saturated on dev, so further training was not treated as the main next step.

Observed training snapshot:

| Epoch | Train DPO Loss | Train Preference Acc | Dev DPO Loss | Dev Policy Preference Acc |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.5633 | 0.846 | 0.3945 | 0.917 |
| 1 | 0.4141 | 0.836 | 0.2785 | 0.917 |

## Evaluation Results

### Before Repetition Control

Blind pairwise judge results showed that DPO usually beat SFT, but dev chrF was slightly lower for DPO:

| Split | DPO Wins | SFT Wins | Ties | DPO Win Rate, Ties as Half | DPO Win Rate, Excluding Ties | SFT chrF | DPO chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 9 | 1 | 2 | 0.8333 | 0.9000 | 41.8070 | 41.2682 |
| Test | 9 | 2 | 1 | 0.7917 | 0.8182 | N/A | N/A |

Manual inspection of SFT-winning cases suggested that some DPO outputs suffered from repetition loops, especially around abstract words such as love, judgement, or beauty.

### After Repetition Control

The selected decoding configuration is:

```bash
--repetition_penalty 1.1 --no_repeat_ngram_size 3
```

With this decoding control, the DPO model produced more ties and fewer clear wins, but SFT no longer beat DPO on test. Dev chrF also improved for DPO:

| Split | DPO Wins | SFT Wins | Ties | DPO Win Rate, Ties as Half | DPO Win Rate, Excluding Ties | SFT chrF | DPO chrF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 6 | 1 | 5 | 0.7083 | 0.8571 | 41.9863 | 42.3278 |
| Test | 5 | 0 | 7 | 0.7083 | 1.0000 | N/A | N/A |

## Interpretation

chrF and judge preference measure different things. chrF rewards overlap with the reference sonnet, while the judge focuses more on style, coherence, fluency, poetic form, and repetition. For this open-ended generation task, the DPO result is best interpreted as a preference-quality improvement rather than a pure chrF optimization.

The repetition-control result is a useful compromise:

- DPO remains preferred or tied against SFT in almost all pairwise comparisons.
- Test has zero SFT wins under the selected decoding configuration.
- Dev chrF improves for DPO, reducing the earlier conflict between preference quality and overlap-based evaluation.
- More ties suggest that decoding control makes outputs safer and less degenerate, though sometimes less distinctively better.

## Current Conclusion

This stage can be considered complete. The practical sonnet-generation track now has:

- LLM-as-a-Judge preference-data generation
- DPO training from the SFT checkpoint
- blind pairwise SFT-vs-DPO evaluation
- decoding-time repetition control
- evidence that DPO is at least competitive with SFT and generally preferred by the judge

The current default evaluation configuration for the DPO model should use:

```bash
--repetition_penalty 1.1 --no_repeat_ngram_size 3
```

Further work should avoid tuning on test results. If more improvement is needed, the next dev-only directions are better candidate generation, more robust judge prompts, or a small sweep over DPO `beta` and learning rate.
