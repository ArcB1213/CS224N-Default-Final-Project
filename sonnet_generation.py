'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from sacrebleu.metrics import CHRF

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


def count_nonempty_lines(text):
  return sum(1 for line in text.splitlines() if line.strip())


def trim_to_nonempty_lines(text, max_lines):
  lines = []
  nonempty_count = 0
  for line in text.splitlines():
    if line.strip():
      nonempty_count += 1
    if nonempty_count > max_lines:
      break
    lines.append(line)
  return '\n'.join(lines).rstrip()


def apply_repetition_penalty(logits, token_ids, repetition_penalty):
  if repetition_penalty <= 1.0:
    return

  for batch_idx in range(logits.size(0)):
    seen_tokens = set(token_ids[batch_idx].tolist())
    for token_id in seen_tokens:
      if logits[batch_idx, token_id] < 0:
        logits[batch_idx, token_id] *= repetition_penalty
      else:
        logits[batch_idx, token_id] /= repetition_penalty


def banned_ngram_tokens(token_ids, no_repeat_ngram_size):
  if no_repeat_ngram_size <= 0 or token_ids.size(1) + 1 < no_repeat_ngram_size:
    return []

  tokens = token_ids[0].tolist()
  prefix_len = no_repeat_ngram_size - 1
  current_prefix = tuple(tokens[-prefix_len:])
  banned = []
  for i in range(len(tokens) - no_repeat_ngram_size + 1):
    ngram = tokens[i:i + no_repeat_ngram_size]
    if tuple(ngram[:-1]) == current_prefix:
      banned.append(ngram[-1])
  return banned


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    local_path = f'{args.model_size}_pretrained'
    model_name = local_path if os.path.isdir(local_path) else args.model_size
    self.gpt = GPT2Model.from_pretrained(model=model_name, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2_pretrained')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Autoregressive LM: produce a logit for EVERY token in the sequence,
    not just the last one. This lets the model learn the full language
    distribution of sonnets during training.
    """
    gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = gpt_output['last_hidden_state']  # [bs, seq_len, hidden_size]
    logits = self.gpt.hidden_state_to_token(hidden_states)  # [bs, seq_len, vocab_size]
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128, max_lines=14,
               repetition_penalty=1.0, no_repeat_ngram_size=0):
    """
    Generates an original sonnet using top-p (nucleus) sampling and softmax temperature.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      decoded_so_far = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
      line_count = count_nonempty_lines(decoded_so_far)
      if line_count > max_lines:
        break

      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature
      if line_count < max_lines:
        logits_last_token[:, self.tokenizer.eos_token_id] = float('-inf')
      apply_repetition_penalty(logits_last_token, token_ids, repetition_penalty)
      for banned_token_id in banned_ngram_tokens(token_ids, no_repeat_ngram_size):
        logits_last_token[:, banned_token_id] = float('-inf')

      # Top-p (nucleus) sampling
      sorted_logits, sorted_indices = torch.sort(logits_last_token, descending=True)
      sorted_probs = torch.softmax(sorted_logits, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

      # Remove tokens with cumulative probability above the threshold
      # Keep the first token that exceeds the threshold (so we always have at least 1)
      sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
      sorted_logits[sorted_indices_to_remove] = float('-inf')

      # Sample from the filtered distribution
      probs = torch.softmax(sorted_logits, dim=-1)
      sampled_index = torch.multinomial(probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    generated_output = trim_to_nonempty_lines(generated_output, max_lines)
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def parse_float_list(value, fallback):
  if value is None or value.strip() == '':
    return [fallback]
  return [float(item.strip()) for item in value.split(',') if item.strip()]


@torch.no_grad()
def generate_sonnets(model, held_out_sonnet_dataset, device, temperature, top_p, max_length, max_lines,
                     repetition_penalty=1.0, no_repeat_ngram_size=0):
  generated_sonnets = []
  for sonnet_id, prompt in held_out_sonnet_dataset:
    encoding = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    _, decoded_output = model.generate(
      encoding['input_ids'],
      temperature=temperature,
      top_p=top_p,
      max_length=max_length,
      max_lines=max_lines,
      repetition_penalty=repetition_penalty,
      no_repeat_ngram_size=no_repeat_ngram_size
    )
    generated_sonnets.append((sonnet_id, f'{decoded_output}\n\n'))
  return generated_sonnets


def score_generated_sonnets(generated_sonnets, gold_path):
  true_sonnets = [x[1].strip() for x in SonnetsDataset(gold_path)]
  predicted_sonnets = [x[1].strip() for x in generated_sonnets]
  max_len = min(len(true_sonnets), len(predicted_sonnets))
  chrf_score = CHRF().corpus_score(predicted_sonnets[:max_len], [true_sonnets[:max_len]])
  return float(chrf_score.score)


@torch.no_grad()
def evaluate_generation_grid(model, dev_dataset, args, device, temperatures, top_ps):
  if not args.dev_gold_sonnet_path or not os.path.exists(args.dev_gold_sonnet_path):
    print(f"Skipping dev chrF: gold file not found at {args.dev_gold_sonnet_path}")
    return None

  best_score = None
  best_temperature = temperatures[0]
  best_top_p = top_ps[0]
  best_sonnets = None

  for temperature in temperatures:
    for top_p in top_ps:
      generated_sonnets = generate_sonnets(
        model,
        dev_dataset,
        device,
        temperature=temperature,
        top_p=top_p,
        max_length=args.max_gen_length,
        max_lines=args.max_sonnet_lines,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
      )
      score = score_generated_sonnets(generated_sonnets, args.dev_gold_sonnet_path)
      print(f"dev chrF @ temperature={temperature:g}, top_p={top_p:g}: {score:.3f}")
      if best_score is None or score > best_score:
        best_score = score
        best_temperature = temperature
        best_top_p = top_p
        best_sonnets = generated_sonnets

  return best_score, best_temperature, best_top_p, best_sonnets


def train(args):
  """Fine-tune GPT-2 for sonnet continuation."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Dev prompts have the first 3 lines and paired gold sonnets for local tuning.
  dev_sonnet_dataset = SonnetsDataset(args.dev_held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  temperatures = parse_float_list(args.temperature_grid, args.temperature)
  top_ps = parse_float_list(args.top_p_grid, args.top_p)
  best_dev_chrf = None
  best_temperature = args.temperature
  best_top_p = args.top_p
  best_checkpoint_path = None

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_target_mask = batch['target_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      b_target_mask = b_target_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      # Shift: logits predict next token, so align predictions with targets
      shift_logits = logits[:, :-1].contiguous()  # [bs, seq_len-1, vocab]
      shift_labels = b_ids[:, 1:].contiguous()     # [bs, seq_len-1]
      shift_mask = b_target_mask[:, 1:].contiguous()  # ignore padding and the conditioning prompt

      loss = F.cross_entropy(
        rearrange(shift_logits, 'b t d -> (b t) d'),
        shift_labels.flatten(),
        reduction='none',
      )
      # Mask out padding positions before averaging
      loss = (loss * shift_mask.flatten()).sum() / shift_mask.flatten().sum()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    model.eval()
    dev_result = None
    if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
      dev_result = evaluate_generation_grid(model, dev_sonnet_dataset, args, device, temperatures, top_ps)

    if dev_result is not None:
      dev_chrf, dev_temperature, dev_top_p, dev_sonnets = dev_result
      if best_dev_chrf is None or dev_chrf > best_dev_chrf:
        best_dev_chrf = dev_chrf
        best_temperature = dev_temperature
        best_top_p = dev_top_p
        best_checkpoint_path = f'best_{args.filepath}'
        args.temperature = best_temperature
        args.top_p = best_top_p
        save_model(model, optimizer, args, best_checkpoint_path)
      print(
        f"Epoch {epoch}: best dev chrF this epoch :: {dev_chrf:.3f} "
        f"(temperature={dev_temperature:g}, top_p={dev_top_p:g})."
      )
      print(
        f"Best dev chrF so far :: {best_dev_chrf:.3f} "
        f"(temperature={best_temperature:g}, top_p={best_top_p:g})."
      )
      print('Sample dev generations:')
      for _, sonnet in dev_sonnets[:args.dev_preview_count]:
        print(f'{sonnet}\n')

    # Save only the latest checkpoint (delete previous to save disk space)
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')
    if epoch > 0:
      prev_path = f'{epoch-1}_{args.filepath}'
      if os.path.exists(prev_path):
        os.remove(prev_path)

  if best_checkpoint_path is not None:
    args.checkpoint_path = best_checkpoint_path
    args.temperature = best_temperature
    args.top_p = best_top_p
  return best_temperature, best_top_p


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  checkpoint_path = getattr(args, 'checkpoint_path', f'{args.epochs-1}_{args.filepath}')
  saved = torch.load(checkpoint_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = generate_sonnets(
    model,
    held_out_sonnet_dataset,
    device,
    temperature=args.temperature,
    top_p=args.top_p,
    max_length=args.max_gen_length,
    max_lines=args.max_sonnet_lines,
    repetition_penalty=args.repetition_penalty,
    no_repeat_ngram_size=args.no_repeat_ngram_size
  )
  for _, sonnet in generated_sonnets:
    print(sonnet)

  sonnet_out_dir = os.path.dirname(args.sonnet_out)
  if sonnet_out_dir:
    os.makedirs(sonnet_out_dir, exist_ok=True)
  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--dev_held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--dev_gold_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--temperature_grid", type=str, default="",
                      help="Comma-separated temperatures for dev chrF search. Defaults to --temperature only.")
  parser.add_argument("--top_p_grid", type=str, default="",
                      help="Comma-separated top-p values for dev chrF search. Defaults to --top_p only.")
  parser.add_argument("--max_gen_length", type=int, default=128)
  parser.add_argument("--max_sonnet_lines", type=int, default=14)
  parser.add_argument("--repetition_penalty", type=float, default=1.0)
  parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
  parser.add_argument("--eval_every", type=int, default=1)
  parser.add_argument("--dev_preview_count", type=int, default=2)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)
