"""Compare sonnet checkpoints and cached GPT-2 bases on the 12 dev prompts."""

import argparse
import json
import os
from pathlib import Path
import re

# Do not download model assets or contact a judge during evaluation.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from sacrebleu.metrics import CHRF

from evaluate_dpo_sonnet import generate_one, load_model
from sonnet_generation import SonnetGPT, add_arguments


DEV_DIR = Path(__file__).resolve().parent / 'data'
DEV_IDS = list(range(132, 144))


def read_poems(path):
  parts = re.split(r'\n[ \t]*(\d+)[ \t]*\n', path.read_text(encoding='utf-8'))
  ids = [int(value) for value in parts[1::2]]
  if ids != DEV_IDS:
    raise ValueError(f'{path} must contain each dev sonnet ID from 132 to 143 in order')
  return dict(zip(ids, (poem.strip() for poem in parts[2::2])))


def nonempty_lines(text):
  return [line.strip() for line in text.splitlines() if line.strip()]


def extract_continuation(text, prompt):
  if text.startswith(prompt):
    return text[len(prompt):].strip()
  return None


def load_dev():
  prompts = read_poems(DEV_DIR / 'sonnets_held_out_dev.txt')
  references = read_poems(DEV_DIR / 'TRUE_sonnets_held_out_dev.txt')
  dev = []
  for sonnet_id in DEV_IDS:
    prompt = prompts[sonnet_id]
    gold = references[sonnet_id]
    gold_continuation = extract_continuation(gold, prompt)
    if (len(nonempty_lines(prompt)) != 3 or len(nonempty_lines(gold)) != 14
        or not gold.startswith(prompt + '\n')):
      raise ValueError(f'Dev prompt and reference do not align for sonnet {sonnet_id}')
    dev.append((sonnet_id, prompt, gold, gold_continuation))
  return dev


def evaluate(model, dev, args):
  rows = []
  for prompt_index, (sonnet_id, prompt, gold, gold_continuation) in enumerate(dev):
    seed = args.seed + prompt_index * 1000 + 11
    text = generate_one(model, prompt, args, seed).strip()
    continuation = extract_continuation(text, prompt)
    lines = nonempty_lines(continuation or '')
    normalized_lines = [' '.join(line.casefold().split()) for line in lines]
    rows.append({
      'sonnet_id': sonnet_id,
      'seed': seed,
      'prompt': prompt,
      'gold': gold,
      'gold_continuation': gold_continuation,
      'text': text,
      'continuation': continuation,
      'prompt_preserved': continuation is not None,
      'starts_on_new_line': text.startswith(prompt + '\n'),
      'line_count': len(nonempty_lines(text)),
      'continuation_line_count': len(lines),
      'repeated_continuation_lines': len(lines) - len(set(normalized_lines)),
    })

  chrf = CHRF()
  return {
    'metrics': {
      'full_chrf': chrf.corpus_score([row['text'] for row in rows], [[row['gold'] for row in rows]]).score,
      'continuation_chrf': chrf.corpus_score(
        [row['continuation'] or '' for row in rows], [[row['gold_continuation'] for row in rows]],
      ).score,
      'prompt_preservation_rate': sum(row['prompt_preserved'] for row in rows) / len(rows),
      'newline_after_prompt_rate': sum(row['starts_on_new_line'] for row in rows) / len(rows),
      'fourteen_line_rate': sum(row['line_count'] == 14 for row in rows) / len(rows),
      'mean_continuation_lines': sum(row['continuation_line_count'] for row in rows) / len(rows),
      'repeated_line_poem_rate': sum(row['repeated_continuation_lines'] > 0 for row in rows) / len(rows),
    },
    'outputs': rows,
  }


def get_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', nargs=2, action='append', default=[], metavar=('NAME', 'PATH'))
  parser.add_argument('--base', nargs=2, action='append', default=[], metavar=('NAME', 'MODEL_SIZE'),
                      help='Cached pretrained model, such as gpt2 or gpt2-medium.')
  parser.add_argument('--out', type=Path, required=True)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--temperature', type=float, default=0.9)
  parser.add_argument('--top_p', type=float, default=0.95)
  parser.add_argument('--max_gen_length', type=int, default=220)
  parser.add_argument('--repetition_penalty', type=float, default=1.0)
  parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
  parser.set_defaults(max_sonnet_lines=14)
  args = parser.parse_args()

  names = [name for name, _ in args.checkpoint + args.base]
  if not names or len(names) != len(set(names)):
    parser.error('provide at least one checkpoint or base and use unique model names')
  if any(size not in {'gpt2', 'gpt2-medium', 'gpt2-middle', 'gpt2-large'} for _, size in args.base):
    parser.error('base model size must be gpt2, gpt2-medium, gpt2-middle or gpt2-large')
  if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.max_gen_length <= 0:
    parser.error('temperature and generation length must be positive and top_p must be in (0, 1]')
  if args.repetition_penalty < 1 or args.no_repeat_ngram_size < 0:
    parser.error('repetition penalty must be at least 1 and n-gram size must be nonnegative')
  if args.use_gpu and not torch.cuda.is_available():
    parser.error('CUDA is unavailable')
  return args


def main():
  args = get_args()
  device = torch.device('cuda' if args.use_gpu else 'cpu')
  dev = load_dev()
  report = {
    'config': {key: value for key, value in vars(args).items() if key != 'out'},
    'device': str(device),
    'models': {},
  }
  for name, size in args.base:
    model = SonnetGPT(add_arguments(argparse.Namespace(model_size=size))).to(device).eval()
    report['models'][name] = evaluate(model, dev, args)
    del model
    print(f'{name}: {report["models"][name]["metrics"]}', flush=True)
  for name, path in args.checkpoint:
    model = load_model(path, torch.device('cpu')).to(device)
    report['models'][name] = evaluate(model, dev, args)
    del model
    print(f'{name}: {report["models"][name]["metrics"]}', flush=True)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
  main()
