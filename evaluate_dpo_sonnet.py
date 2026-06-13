"""
Blind pairwise evaluation for SFT vs DPO sonnet checkpoints.
"""

import argparse
import json
import os
import random
import time

import requests
import torch

from llm_judge_sonnet import (
  DEEPSEEK_URL,
  JudgeError,
  extract_json_object,
  load_prompts,
  parse_splits,
  set_generation_seed,
  write_jsonl_record,
)
from sonnet_generation import SonnetGPT, count_nonempty_lines, score_generated_sonnets


def load_model(checkpoint_path, device):
  saved = torch.load(checkpoint_path, weights_only=False, map_location=device)
  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  return model


@torch.no_grad()
def generate_one(model, prompt, args, seed):
  set_generation_seed(seed)
  encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(model.get_device())
  _, text = model.generate(
    encoding['input_ids'],
    temperature=args.temperature,
    top_p=args.top_p,
    max_length=args.max_gen_length,
    max_lines=args.max_sonnet_lines,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=args.no_repeat_ngram_size,
  )
  return text


def build_pairwise_messages(prompt, output_a, output_b):
  user_content = (
    "You are a blind evaluator of Shakespearean sonnet continuations.\n"
    "The model is given the first three lines and must produce a complete 14-line sonnet.\n"
    "Choose which output is better for a human reader as a Shakespearean sonnet continuation.\n"
    "Judge coherence, Shakespearean diction, grammatical fluency, poetic structure, low repetition, and form.\n"
    "Do not use any hidden reference text and do not favor an output just because it is longer.\n\n"
    f"Prompt first three lines:\n{prompt}\n\n"
    f"Output A:\n{output_a}\n\n"
    f"Output B:\n{output_b}\n\n"
    "Return JSON only with this schema:\n"
    "{\n"
    '  "winner": "A" or "B" or "tie",\n'
    '  "scores": {\n'
    '    "A": {"overall": 1-5, "form": 1-5, "style": 1-5, "coherence": 1-5, "fluency": 1-5, "repetition": 1-5},\n'
    '    "B": {"overall": 1-5, "form": 1-5, "style": 1-5, "coherence": 1-5, "fluency": 1-5, "repetition": 1-5}\n'
    "  },\n"
    '  "brief_rationale": "one short paragraph explaining the decision"\n'
    "}\n"
  )
  return [
    {
      'role': 'system',
      'content': 'You are a strict poetry judge. Return valid JSON only.',
    },
    {
      'role': 'user',
      'content': user_content,
    },
  ]


def normalize_pairwise_judgment(judgment):
  winner = str(judgment.get('winner', '')).strip().upper()
  if winner not in {'A', 'B', 'TIE'}:
    raise ValueError("Judge response must contain winner A, B, or tie")
  judgment['winner'] = 'tie' if winner == 'TIE' else winner
  if 'scores' not in judgment or not isinstance(judgment['scores'], dict):
    raise ValueError("Judge response missing scores object")
  if 'brief_rationale' not in judgment:
    judgment['brief_rationale'] = ''
  return judgment


def call_pairwise_judge(messages, args):
  api_key = os.environ.get('DEEPSEEK_API_KEY')
  if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not set. Set it before running the evaluation script.")

  payload = {
    'model': args.judge_model,
    'messages': messages,
    'temperature': 0,
    'stream': False,
  }
  headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
  }

  last_error = None
  last_raw_response = ''
  for attempt in range(1, args.max_retries + 1):
    try:
      response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=args.request_timeout)
      if response.status_code in {429, 500, 502, 503, 504}:
        raise requests.HTTPError(f"retryable HTTP {response.status_code}: {response.text}")
      response.raise_for_status()
      raw_content = response.json()['choices'][0]['message']['content']
      last_raw_response = raw_content
      judgment = normalize_pairwise_judgment(extract_json_object(raw_content))
      return judgment, raw_content
    except Exception as exc:
      last_error = exc
      if attempt == args.max_retries:
        break
      time.sleep(args.retry_base_seconds * (2 ** (attempt - 1)))

  raise JudgeError(f"Pairwise judge failed after {args.max_retries} attempts: {last_error}", last_raw_response)


def output_paths(out_dir, split):
  return {
    'generations': os.path.join(out_dir, f'pairwise_generations_{split}.jsonl'),
    'requests': os.path.join(out_dir, f'pairwise_requests_{split}.jsonl'),
    'judgments': os.path.join(out_dir, f'pairwise_judgments_{split}.jsonl'),
    'summary': os.path.join(out_dir, f'pairwise_summary_{split}.json'),
    'sft_sonnets': os.path.join(out_dir, f'sft_{split}_sonnets.txt'),
    'dpo_sonnets': os.path.join(out_dir, f'dpo_{split}_sonnets.txt'),
    'failed': os.path.join(out_dir, 'failed_pairwise_judgments.jsonl'),
  }


def ensure_outputs_available(paths, overwrite, dry_run):
  keys = ['generations', 'requests']
  if not dry_run:
    keys.extend(['judgments', 'summary', 'sft_sonnets', 'dpo_sonnets'])
  existing = [paths[key] for key in keys if os.path.exists(paths[key])]
  if existing and not overwrite:
    existing_paths = '\n'.join(existing)
    raise FileExistsError(f"Output file(s) already exist. Pass --overwrite to replace them:\n{existing_paths}")


def write_sonnet_file(path, records, key):
  with open(path, 'w', encoding='utf-8') as fp:
    fp.write("--Generated Sonnets-- \n\n")
    for record in sorted(records, key=lambda item: item['prompt_id']):
      fp.write(f"\n{record['prompt_id']}\n")
      fp.write(f"{record[key].rstrip()}\n\n")


def winner_to_model(winner, label_to_model):
  if winner == 'tie':
    return 'tie'
  return label_to_model[winner]


def summarize(records):
  judged = [record for record in records if record.get('winner_model')]
  total = len(judged)
  dpo_wins = sum(1 for record in judged if record['winner_model'] == 'dpo')
  sft_wins = sum(1 for record in judged if record['winner_model'] == 'sft')
  ties = sum(1 for record in judged if record['winner_model'] == 'tie')
  non_tie_total = max(1, dpo_wins + sft_wins)
  return {
    'total_judged': total,
    'dpo_wins': dpo_wins,
    'sft_wins': sft_wins,
    'ties': ties,
    'dpo_win_rate_including_ties_as_half': (dpo_wins + 0.5 * ties) / max(1, total),
    'dpo_win_rate_excluding_ties': dpo_wins / non_tie_total,
  }


def process_split(split, sft_model, dpo_model, args):
  prompts = load_prompts(split)
  if args.limit_prompts is not None:
    prompts = prompts[:args.limit_prompts]

  paths = output_paths(args.out_dir, split)
  ensure_outputs_available(paths, args.overwrite, args.dry_run)
  os.makedirs(args.out_dir, exist_ok=True)

  judged_records = []
  generation_records = []
  split_seed_offsets = {'dev': 101, 'test': 202}
  rng = random.Random(args.seed + split_seed_offsets[split])

  with open(paths['generations'], 'w', encoding='utf-8') as generations_fp, \
      open(paths['requests'], 'w', encoding='utf-8') as requests_fp:
    judgments_fp = None
    failed_fp = None
    try:
      if not args.dry_run:
        judgments_fp = open(paths['judgments'], 'w', encoding='utf-8')
        failed_fp = open(paths['failed'], 'a', encoding='utf-8')

      for prompt_id, prompt in prompts:
        sft_text = generate_one(sft_model, prompt, args, args.seed + prompt_id * 1000 + 11)
        dpo_text = generate_one(dpo_model, prompt, args, args.seed + prompt_id * 1000 + 11)

        if rng.random() < 0.5:
          label_to_model = {'A': 'sft', 'B': 'dpo'}
          output_a, output_b = sft_text, dpo_text
        else:
          label_to_model = {'A': 'dpo', 'B': 'sft'}
          output_a, output_b = dpo_text, sft_text

        generation_record = {
          'split': split,
          'prompt_id': prompt_id,
          'prompt': prompt,
          'sft_text': sft_text,
          'dpo_text': dpo_text,
          'sft_line_count': count_nonempty_lines(sft_text),
          'dpo_line_count': count_nonempty_lines(dpo_text),
          'label_to_model': label_to_model,
        }
        generation_records.append(generation_record)
        write_jsonl_record(generations_fp, generation_record)

        messages = build_pairwise_messages(prompt, output_a, output_b)
        request_record = {
          'split': split,
          'prompt_id': prompt_id,
          'judge_model': args.judge_model,
          'messages': messages,
          'label_to_model': label_to_model,
        }
        write_jsonl_record(requests_fp, request_record)

        if args.dry_run:
          continue

        try:
          judgment, raw_response = call_pairwise_judge(messages, args)
        except Exception as exc:
          raw_response = exc.raw_response if isinstance(exc, JudgeError) else ''
          write_jsonl_record(failed_fp, {
            'split': split,
            'prompt_id': prompt_id,
            'error': str(exc),
            'judge_model': args.judge_model,
            'raw_response': raw_response,
          })
          print(f"[{split} prompt {prompt_id}] pairwise judge failed: {exc}")
          continue

        winner_model = winner_to_model(judgment['winner'], label_to_model)
        judgment_record = {
          **generation_record,
          'judge_model': args.judge_model,
          'winner': judgment['winner'],
          'winner_model': winner_model,
          'scores': judgment.get('scores', {}),
          'brief_rationale': judgment.get('brief_rationale', ''),
          'raw_response': raw_response,
        }
        judged_records.append(judgment_record)
        write_jsonl_record(judgments_fp, judgment_record)
        print(f"[{split} prompt {prompt_id}] winner: {winner_model}")
    finally:
      for fp in [judgments_fp, failed_fp]:
        if fp is not None:
          fp.close()

  if not args.dry_run:
    write_sonnet_file(paths['sft_sonnets'], generation_records, 'sft_text')
    write_sonnet_file(paths['dpo_sonnets'], generation_records, 'dpo_text')
    summary = summarize(judged_records)

    if split == 'dev' and os.path.exists(args.dev_gold_sonnet_path):
      sft_generated = [(record['prompt_id'], f"{record['sft_text'].rstrip()}\n\n") for record in generation_records]
      dpo_generated = [(record['prompt_id'], f"{record['dpo_text'].rstrip()}\n\n") for record in generation_records]
      summary['sft_chrf'] = score_generated_sonnets(sft_generated, args.dev_gold_sonnet_path)
      summary['dpo_chrf'] = score_generated_sonnets(dpo_generated, args.dev_gold_sonnet_path)

    with open(paths['summary'], 'w', encoding='utf-8') as fp:
      json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[{split}] summary: {json.dumps(summary, ensure_ascii=False)}")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sft_checkpoint_path', type=str, required=True)
  parser.add_argument('--dpo_checkpoint_path', type=str, required=True)
  parser.add_argument('--splits', type=str, default='dev,test')
  parser.add_argument('--judge_model', type=str, default='deepseek-v4-flash')
  parser.add_argument('--out_dir', type=str, default='my_results/dpo_eval')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--limit_prompts', type=int, default=None)
  parser.add_argument('--temperature', type=float, default=0.9)
  parser.add_argument('--top_p', type=float, default=0.95)
  parser.add_argument('--max_gen_length', type=int, default=128)
  parser.add_argument('--max_sonnet_lines', type=int, default=14)
  parser.add_argument('--repetition_penalty', type=float, default=1.0)
  parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
  parser.add_argument('--dev_gold_sonnet_path', type=str, default='data/TRUE_sonnets_held_out_dev.txt')
  parser.add_argument('--max_retries', type=int, default=3)
  parser.add_argument('--retry_base_seconds', type=float, default=1.0)
  parser.add_argument('--request_timeout', type=float, default=60.0)
  return parser.parse_args()


def main():
  args = get_args()
  splits = parse_splits(args.splits)
  if 'train' in splits:
    raise ValueError("Pairwise DPO evaluation should use held-out splits only: dev,test")
  if not args.dry_run and not os.environ.get('DEEPSEEK_API_KEY'):
    raise RuntimeError("DEEPSEEK_API_KEY is not set. Set it before running pairwise evaluation.")

  if args.overwrite and not args.dry_run:
    failed_path = os.path.join(args.out_dir, 'failed_pairwise_judgments.jsonl')
    if os.path.exists(failed_path):
      os.remove(failed_path)

  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  sft_model = load_model(args.sft_checkpoint_path, device)
  dpo_model = load_model(args.dpo_checkpoint_path, device)

  for split in splits:
    process_split(split, sft_model, dpo_model, args)


if __name__ == '__main__':
  main()
