"""
DPO training for the sonnet generator using LLM-judged preference pairs.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from optimizer import AdamW
from sonnet_generation import SonnetGPT, save_model


TQDM_DISABLE = False


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class DpoPreferenceDataset(Dataset):
  def __init__(self, path):
    self.examples = []
    with open(path, encoding='utf-8') as fp:
      for line in fp:
        if line.strip():
          self.examples.append(json.loads(line))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    return self.examples[idx]


def encode_prompt_completion(tokenizer, prompt, completion, max_length):
  prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
  full_ids = tokenizer(prompt + completion, add_special_tokens=False, truncation=True, max_length=max_length)['input_ids']

  if len(full_ids) <= len(prompt_ids):
    full_ids = tokenizer(prompt + '\n' + completion, add_special_tokens=False,
                         truncation=True, max_length=max_length)['input_ids']
  prompt_len = min(len(prompt_ids), max(0, len(full_ids) - 1))
  return full_ids, prompt_len


def collate_dpo_batch(examples, tokenizer, max_length):
  encoded = []
  for example in examples:
    prompt = example['prompt']
    chosen_ids, chosen_prompt_len = encode_prompt_completion(tokenizer, prompt, example['chosen'], max_length)
    rejected_ids, rejected_prompt_len = encode_prompt_completion(tokenizer, prompt, example['rejected'], max_length)
    encoded.append((chosen_ids, chosen_prompt_len, rejected_ids, rejected_prompt_len))

  max_seq_len = max(max(len(item[0]), len(item[2])) for item in encoded)
  pad_id = tokenizer.pad_token_id
  batch_size = len(encoded)

  chosen_input_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
  chosen_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
  chosen_loss_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
  rejected_input_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
  rejected_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
  rejected_loss_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float)

  for i, (chosen_ids, chosen_prompt_len, rejected_ids, rejected_prompt_len) in enumerate(encoded):
    chosen_len = len(chosen_ids)
    rejected_len = len(rejected_ids)

    chosen_input_ids[i, :chosen_len] = torch.tensor(chosen_ids, dtype=torch.long)
    chosen_attention_mask[i, :chosen_len] = 1
    chosen_loss_mask[i, chosen_prompt_len:chosen_len] = 1

    rejected_input_ids[i, :rejected_len] = torch.tensor(rejected_ids, dtype=torch.long)
    rejected_attention_mask[i, :rejected_len] = 1
    rejected_loss_mask[i, rejected_prompt_len:rejected_len] = 1

  return {
    'chosen_input_ids': chosen_input_ids,
    'chosen_attention_mask': chosen_attention_mask,
    'chosen_loss_mask': chosen_loss_mask,
    'rejected_input_ids': rejected_input_ids,
    'rejected_attention_mask': rejected_attention_mask,
    'rejected_loss_mask': rejected_loss_mask,
  }


def sequence_log_probs(model, input_ids, attention_mask, loss_mask):
  logits = model(input_ids, attention_mask)
  shift_logits = logits[:, :-1, :].contiguous()
  shift_labels = input_ids[:, 1:].contiguous()
  shift_mask = loss_mask[:, 1:].contiguous()

  token_log_probs = F.log_softmax(shift_logits, dim=-1)
  selected_log_probs = token_log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
  return (selected_log_probs * shift_mask).sum(dim=-1)


def preference_accuracy(chosen_logps, rejected_logps):
  return (chosen_logps > rejected_logps).float().mean().item()


def load_models(checkpoint_path, device):
  saved = torch.load(checkpoint_path, weights_only=False, map_location=device)

  policy_model = SonnetGPT(saved['args'])
  policy_model.load_state_dict(saved['model'])
  policy_model = policy_model.to(device)

  reference_model = SonnetGPT(saved['args'])
  reference_model.load_state_dict(saved['model'])
  reference_model = reference_model.to(device)
  reference_model.eval()
  for param in reference_model.parameters():
    param.requires_grad = False

  return policy_model, reference_model, saved['args']


@torch.no_grad()
def evaluate_dpo(dataloader, policy_model, reference_model, beta, device):
  policy_model.eval()
  total_loss = 0.0
  total_policy_acc = 0.0
  total_ref_acc = 0.0
  num_batches = 0

  for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
    batch = {key: value.to(device) for key, value in batch.items()}
    policy_chosen = sequence_log_probs(
      policy_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['chosen_loss_mask'])
    policy_rejected = sequence_log_probs(
      policy_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['rejected_loss_mask'])
    ref_chosen = sequence_log_probs(
      reference_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['chosen_loss_mask'])
    ref_rejected = sequence_log_probs(
      reference_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['rejected_loss_mask'])

    policy_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    loss = -F.logsigmoid(beta * (policy_logratio - ref_logratio)).mean()

    total_loss += loss.item()
    total_policy_acc += preference_accuracy(policy_chosen, policy_rejected)
    total_ref_acc += preference_accuracy(ref_chosen, ref_rejected)
    num_batches += 1

  return {
    'loss': total_loss / max(1, num_batches),
    'policy_acc': total_policy_acc / max(1, num_batches),
    'reference_acc': total_ref_acc / max(1, num_batches),
  }


def train(args):
  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  policy_model, reference_model, checkpoint_args = load_models(args.checkpoint_path, device)
  tokenizer = policy_model.tokenizer

  train_dataset = DpoPreferenceDataset(args.dpo_train_path)
  train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=lambda examples: collate_dpo_batch(examples, tokenizer, args.max_length),
  )

  dev_dataloader = None
  if args.dpo_dev_path and os.path.exists(args.dpo_dev_path):
    dev_dataset = DpoPreferenceDataset(args.dpo_dev_path)
    dev_dataloader = DataLoader(
      dev_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      collate_fn=lambda examples: collate_dpo_batch(examples, tokenizer, args.max_length),
    )

  optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  best_dev_loss = None
  os.makedirs(args.out_dir, exist_ok=True)

  for epoch in range(args.epochs):
    policy_model.train()
    total_loss = 0.0
    total_policy_acc = 0.0
    total_margin = 0.0
    num_batches = 0

    for batch in tqdm(train_dataloader, desc=f'dpo-train-{epoch}', disable=TQDM_DISABLE):
      batch = {key: value.to(device) for key, value in batch.items()}
      optimizer.zero_grad()

      policy_chosen = sequence_log_probs(
        policy_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['chosen_loss_mask'])
      policy_rejected = sequence_log_probs(
        policy_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['rejected_loss_mask'])
      with torch.no_grad():
        ref_chosen = sequence_log_probs(
          reference_model, batch['chosen_input_ids'], batch['chosen_attention_mask'], batch['chosen_loss_mask'])
        ref_rejected = sequence_log_probs(
          reference_model, batch['rejected_input_ids'], batch['rejected_attention_mask'], batch['rejected_loss_mask'])

      policy_logratio = policy_chosen - policy_rejected
      ref_logratio = ref_chosen - ref_rejected
      logits = args.beta * (policy_logratio - ref_logratio)
      loss = -F.logsigmoid(logits).mean()

      loss.backward()
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
      optimizer.step()

      total_loss += loss.item()
      total_policy_acc += preference_accuracy(policy_chosen, policy_rejected)
      total_margin += (policy_logratio - ref_logratio).mean().item()
      num_batches += 1

    train_loss = total_loss / max(1, num_batches)
    train_acc = total_policy_acc / max(1, num_batches)
    train_margin = total_margin / max(1, num_batches)
    print(
      f"Epoch {epoch}: train dpo loss :: {train_loss:.4f}, "
      f"policy preference acc :: {train_acc:.3f}, avg margin :: {train_margin:.3f}."
    )

    latest_path = os.path.join(args.out_dir, args.output_checkpoint_name)
    save_model(policy_model, optimizer, checkpoint_args, latest_path)

    if dev_dataloader is not None:
      dev_metrics = evaluate_dpo(dev_dataloader, policy_model, reference_model, args.beta, device)
      print(
        f"Epoch {epoch}: dev dpo loss :: {dev_metrics['loss']:.4f}, "
        f"policy preference acc :: {dev_metrics['policy_acc']:.3f}, "
        f"reference preference acc :: {dev_metrics['reference_acc']:.3f}."
      )
      if best_dev_loss is None or dev_metrics['loss'] < best_dev_loss:
        best_dev_loss = dev_metrics['loss']
        best_path = os.path.join(args.out_dir, 'best_' + args.output_checkpoint_name)
        save_model(policy_model, optimizer, checkpoint_args, best_path)


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_path', type=str, required=True)
  parser.add_argument('--dpo_train_path', type=str, default='my_results/llm_judge/dpo_pairs_train.jsonl')
  parser.add_argument('--dpo_dev_path', type=str, default='my_results/llm_judge/dpo_pairs_dev.jsonl')
  parser.add_argument('--out_dir', type=str, default='my_results/dpo_sonnet')
  parser.add_argument('--output_checkpoint_name', type=str, default='dpo-sonnet.pt')
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--epochs', type=int, default=3)
  parser.add_argument('--batch_size', type=int, default=2)
  parser.add_argument('--lr', type=float, default=1e-6)
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--beta', type=float, default=0.1)
  parser.add_argument('--max_length', type=int, default=256)
  parser.add_argument('--max_grad_norm', type=float, default=1.0)
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  seed_everything(args.seed)
  train(args)
