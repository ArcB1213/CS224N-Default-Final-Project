"""
Generate sonnet candidates, ask DeepSeek to judge them, and export reranked
sonnets plus DPO-style preference pairs.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import requests
import torch

from datasets import SonnetsDataset
from sonnet_generation import (
  SonnetGPT,
  count_nonempty_lines,
  score_generated_sonnets,
)


DEFAULT_TEMPERATURES = [0.8, 0.9, 1.0, 1.1]
DEFAULT_TOP_PS = [0.9, 0.95]
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


class JudgeError(RuntimeError):
  def __init__(self, message, raw_response=''):
    super().__init__(message)
    self.raw_response = raw_response


def set_generation_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_splits(value):
  splits = [item.strip() for item in value.split(',') if item.strip()]
  invalid = [split for split in splits if split not in {'train', 'dev', 'test'}]
  if invalid:
    raise ValueError(f"Unsupported split(s): {', '.join(invalid)}")
  return splits


def candidate_params(candidate_count):
  params = [(temperature, top_p) for temperature in DEFAULT_TEMPERATURES for top_p in DEFAULT_TOP_PS]
  if candidate_count < 1 or candidate_count > len(params):
    raise ValueError(f"--candidate_count must be between 1 and {len(params)}")
  return params[:candidate_count]


def split_paths(split):
  if split == 'train':
    return 'data/sonnets.txt'
  if split == 'dev':
    return 'data/sonnets_held_out_dev.txt'
  if split == 'test':
    return 'data/sonnets_held_out.txt'
  raise ValueError(f"Unsupported split: {split}")


def first_nonempty_lines(text, num_lines=3):
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  return '\n'.join(lines[:num_lines])


def load_prompts(split):
  dataset = SonnetsDataset(split_paths(split))
  if split == 'train':
    return [(sonnet_id, first_nonempty_lines(sonnet)) for sonnet_id, sonnet in dataset]
  return list(dataset)


def output_paths(out_dir, split, dry_run):
  paths = {
    'candidates': os.path.join(out_dir, f'candidates_{split}.jsonl'),
    'judge_requests': os.path.join(out_dir, f'judge_requests_{split}.jsonl'),
  }
  if not dry_run:
    paths.update({
      'judgments': os.path.join(out_dir, f'judgments_{split}.jsonl'),
      'dpo_pairs': os.path.join(out_dir, f'dpo_pairs_{split}.jsonl'),
      'reranked': os.path.join(out_dir, f'reranked_{split}_sonnets.txt'),
      'failed': os.path.join(out_dir, 'failed_judgments.jsonl'),
    })
  return paths


def ensure_outputs_available(paths, overwrite, append):
  if overwrite and append:
    raise ValueError("--overwrite and --append cannot be used together")
  if append:
    return
  existing = [path for key, path in paths.items() if key != 'failed' and os.path.exists(path)]
  if existing and not overwrite:
    existing_paths = '\n'.join(existing)
    raise FileExistsError(
      f"Output file(s) already exist. Pass --overwrite to replace them or --append to continue:\n{existing_paths}"
    )


def write_jsonl_record(fp, record):
  fp.write(json.dumps(record, ensure_ascii=False) + '\n')
  fp.flush()


def load_sonnet_model(checkpoint_path, device):
  saved = torch.load(checkpoint_path, weights_only=False, map_location=device)
  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  return model


def strip_prompt(full_text, prompt):
  if full_text.startswith(prompt):
    return full_text[len(prompt):].lstrip()
  return full_text


@torch.no_grad()
def generate_candidates_for_prompt(model, prompt_id, prompt, params, args, device):
  records = []
  for candidate_id, (temperature, top_p) in enumerate(params):
    set_generation_seed(args.seed + prompt_id * 1000 + candidate_id)
    encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    _, generated_text = model.generate(
      encoding['input_ids'],
      temperature=temperature,
      top_p=top_p,
      max_length=args.max_gen_length,
      max_lines=args.max_sonnet_lines,
      repetition_penalty=args.repetition_penalty,
      no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    records.append({
      'prompt_id': prompt_id,
      'candidate_id': candidate_id,
      'temperature': temperature,
      'top_p': top_p,
      'prompt': prompt,
      'text': generated_text,
      'completion': strip_prompt(generated_text, prompt),
      'line_count': count_nonempty_lines(generated_text),
    })
  return records


def judge_prompt_mode_for_split(split):
  if split == 'train':
    return 'train_preference'
  return 'eval_preference'


def judge_instructions(prompt_mode):
  if prompt_mode == 'train_preference':
    return (
      "You are creating preference data for DPO training of a small GPT-2 sonnet generator.\n"
      "Rank candidates by which output the model should learn to prefer in practice.\n"
      "Prioritize human-readable poetic quality: coherent continuation, Shakespearean diction, grammatical fluency,\n"
      "14-line sonnet-like form, low repetition, and no degeneration.\n"
      "Do not try to predict character n-gram overlap with a hidden reference; no reference is provided.\n"
      "Choose a strong positive example for training even if every candidate is imperfect.\n"
    )
  if prompt_mode == 'eval_preference':
    return (
      "You are a blind evaluator for held-out Shakespearean sonnet continuations.\n"
      "Rank candidates by overall quality as finished poems for a human reader.\n"
      "Use a strict evaluation standard: coherent continuation, Shakespearean diction, grammatical fluency,\n"
      "14-line sonnet-like form, low repetition, and poetic structure.\n"
      "Do not reward generic modern poetic language, and do not compare against any hidden reference text.\n"
    )
  raise ValueError(f"Unsupported judge prompt mode: {prompt_mode}")


def build_judge_messages(prompt, candidates, prompt_mode):
  candidate_blocks = []
  for candidate in candidates:
    candidate_blocks.append(
      f"Candidate {candidate['candidate_id']}:\n{candidate['text']}"
    )

  user_content = (
    judge_instructions(prompt_mode)
    + "\n"
    "You are judging candidate continuations for a Shakespearean sonnet task.\n"
    "The model is given the first three lines and must produce a complete 14-line sonnet.\n"
    "Do not compare against any hidden reference sonnet. Judge only the candidates below.\n\n"
    f"Prompt first three lines:\n{prompt}\n\n"
    "Candidates:\n\n"
    + "\n\n".join(candidate_blocks)
    + "\n\nReturn JSON only with this schema:\n"
    "{\n"
    '  "ranking": [candidate ids from best to worst],\n'
    '  "scores": {\n'
    '    "0": {\n'
    '      "form_14_lines": 1-5,\n'
    '      "shakespearean_diction": 1-5,\n'
    '      "continuation_coherence": 1-5,\n'
    '      "grammatical_fluency": 1-5,\n'
    '      "rhyme_poetic_structure": 1-5,\n'
    '      "no_repetition_or_degeneration": 1-5\n'
    "    }\n"
    "  },\n"
    '  "brief_rationale": "one short paragraph explaining the ranking"\n'
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


def extract_json_object(text):
  stripped = text.strip()
  if stripped.startswith('```'):
    stripped = stripped.strip('`')
    if stripped.startswith('json'):
      stripped = stripped[4:].strip()
  start = stripped.find('{')
  end = stripped.rfind('}')
  if start == -1 or end == -1 or end <= start:
    raise ValueError("No JSON object found in judge response")
  return json.loads(stripped[start:end + 1])


def normalize_ranking(judgment, candidate_ids):
  if 'ranking' not in judgment or not isinstance(judgment['ranking'], list):
    raise ValueError("Judge response missing ranking list")

  ranking = [int(item) for item in judgment['ranking']]
  expected = set(candidate_ids)
  if set(ranking) != expected or len(ranking) != len(candidate_ids):
    raise ValueError(f"Ranking must contain each candidate id exactly once: {sorted(expected)}")

  judgment['ranking'] = ranking
  if 'scores' not in judgment or not isinstance(judgment['scores'], dict):
    raise ValueError("Judge response missing scores object")
  if 'brief_rationale' not in judgment:
    judgment['brief_rationale'] = ''
  return judgment


def call_deepseek_judge(messages, args, candidate_ids):
  api_key = os.environ.get('DEEPSEEK_API_KEY')
  if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not set. Set it before running the judge script.")

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
      judgment = extract_json_object(raw_content)
      judgment = normalize_ranking(judgment, candidate_ids)
      return judgment, raw_content
    except Exception as exc:
      last_error = exc
      if attempt == args.max_retries:
        break
      time.sleep(args.retry_base_seconds * (2 ** (attempt - 1)))

  raise JudgeError(f"DeepSeek judge failed after {args.max_retries} attempts: {last_error}", last_raw_response)


def build_dpo_pairs(split, prompt_id, prompt, candidates_by_id, judgment, judge_model):
  ranking = judgment['ranking']
  chosen_id = ranking[0]
  rejected_ids = ranking[-min(3, len(ranking) - 1):]
  scores = judgment.get('scores', {})
  pairs = []

  for rejected_id in rejected_ids:
    chosen = candidates_by_id[chosen_id]
    rejected = candidates_by_id[rejected_id]
    pairs.append({
      'split': split,
      'prompt_id': prompt_id,
      'prompt': prompt,
      'chosen': chosen['completion'],
      'rejected': rejected['completion'],
      'chosen_id': chosen_id,
      'rejected_id': rejected_id,
      'chosen_text': chosen['text'],
      'rejected_text': rejected['text'],
      'judge_model': judge_model,
      'judge_prompt_mode': judgment.get('judge_prompt_mode', ''),
      'scores': scores,
      'brief_rationale': judgment.get('brief_rationale', ''),
    })
  return pairs


def write_reranked_sonnets(path, best_records, append=False):
  mode = 'a' if append else 'w'
  needs_header = not append or not os.path.exists(path) or os.path.getsize(path) == 0
  with open(path, mode, encoding='utf-8') as fp:
    if needs_header:
      fp.write("--Generated Sonnets-- \n\n")
    for record in sorted(best_records, key=lambda item: item['prompt_id']):
      fp.write(f"\n{record['prompt_id']}\n")
      fp.write(f"{record['text'].rstrip()}\n\n")


def process_split(split, model, args, device):
  prompts = [(prompt_id, prompt) for prompt_id, prompt in load_prompts(split)
             if prompt_id >= args.start_prompt_id]
  params = candidate_params(args.candidate_count)
  paths = output_paths(args.out_dir, split, args.dry_run)
  ensure_outputs_available(paths, args.overwrite, args.append)

  os.makedirs(args.out_dir, exist_ok=True)
  best_records = []
  max_examples = len(prompts) if args.limit_prompts is None else min(args.limit_prompts, len(prompts))
  prompt_mode = judge_prompt_mode_for_split(split)
  write_mode = 'a' if args.append else 'w'

  with open(paths['candidates'], write_mode, encoding='utf-8') as candidates_fp, \
      open(paths['judge_requests'], write_mode, encoding='utf-8') as requests_fp:
    judgments_fp = None
    dpo_fp = None
    failed_fp = None
    try:
      if not args.dry_run:
        judgments_fp = open(paths['judgments'], write_mode, encoding='utf-8')
        dpo_fp = open(paths['dpo_pairs'], write_mode, encoding='utf-8')
        failed_fp = open(paths['failed'], 'a', encoding='utf-8')

      for prompt_id, prompt in prompts[:max_examples]:
        candidates = generate_candidates_for_prompt(model, prompt_id, prompt, params, args, device)
        for candidate in candidates:
          candidate_record = dict(candidate)
          candidate_record['split'] = split
          candidate_record['judge_prompt_mode'] = prompt_mode
          write_jsonl_record(candidates_fp, candidate_record)

        messages = build_judge_messages(prompt, candidates, prompt_mode)
        request_record = {
          'split': split,
          'prompt_id': prompt_id,
          'judge_model': args.judge_model,
          'judge_prompt_mode': prompt_mode,
          'messages': messages,
        }
        write_jsonl_record(requests_fp, request_record)

        if args.dry_run:
          continue

        candidate_ids = [candidate['candidate_id'] for candidate in candidates]
        candidates_by_id = {candidate['candidate_id']: candidate for candidate in candidates}
        try:
          judgment, raw_response = call_deepseek_judge(messages, args, candidate_ids)
        except Exception as exc:
          raw_response = exc.raw_response if isinstance(exc, JudgeError) else ''
          write_jsonl_record(failed_fp, {
            'split': split,
            'prompt_id': prompt_id,
            'error': str(exc),
            'judge_model': args.judge_model,
            'judge_prompt_mode': prompt_mode,
            'raw_response': raw_response,
          })
          print(f"[{split} prompt {prompt_id}] judge failed: {exc}")
          continue

        judgment_record = {
          'split': split,
          'prompt_id': prompt_id,
          'judge_model': args.judge_model,
          'judge_prompt_mode': prompt_mode,
          'ranking': judgment['ranking'],
          'scores': judgment.get('scores', {}),
          'brief_rationale': judgment.get('brief_rationale', ''),
          'raw_response': raw_response,
        }
        judgment['judge_prompt_mode'] = prompt_mode
        write_jsonl_record(judgments_fp, judgment_record)

        best_id = judgment['ranking'][0]
        best_records.append(candidates_by_id[best_id])
        for pair in build_dpo_pairs(split, prompt_id, prompt, candidates_by_id, judgment, args.judge_model):
          write_jsonl_record(dpo_fp, pair)

        print(f"[{split} prompt {prompt_id}] best candidate: {best_id}")
    finally:
      for fp in [judgments_fp, dpo_fp, failed_fp]:
        if fp is not None:
          fp.close()

  if not args.dry_run:
    write_reranked_sonnets(paths['reranked'], best_records, append=args.append)
    if split == 'dev' and os.path.exists(args.dev_gold_sonnet_path):
      if args.append:
        print("Skipping reranked dev chrF in append mode; score the merged file separately after completion.")
      else:
        generated = [(record['prompt_id'], f"{record['text'].rstrip()}\n\n") for record in best_records]
        score = score_generated_sonnets(generated, args.dev_gold_sonnet_path)
        print(f"reranked dev chrF: {score:.3f}")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_path', type=str, required=True)
  parser.add_argument('--splits', type=str, default='train,dev,test')
  parser.add_argument('--candidate_count', type=int, default=8)
  parser.add_argument('--judge_model', type=str, default='deepseek-v4-flash')
  parser.add_argument('--out_dir', type=str, default='my_results/llm_judge')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--append', action='store_true')
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--start_prompt_id', type=int, default=0)
  parser.add_argument('--limit_prompts', type=int, default=None)
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
  if not args.dry_run and not os.environ.get('DEEPSEEK_API_KEY'):
    raise RuntimeError("DEEPSEEK_API_KEY is not set. Set it before running the judge script.")
  if args.overwrite and not args.dry_run:
    failed_path = os.path.join(args.out_dir, 'failed_judgments.jsonl')
    if os.path.exists(failed_path):
      os.remove(failed_path)
  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  model = load_sonnet_model(args.checkpoint_path, device)

  for split in splits:
    process_split(split, model, args, device)


if __name__ == '__main__':
  main()
