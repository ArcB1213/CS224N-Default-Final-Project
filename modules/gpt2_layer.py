from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    Residual connection helper: project output, apply dropout, then add to input.
    Layer norm is NOT applied here — it's handled separately in forward (pre-norm).
    """
    output = dense_layer(output)
    output = dropout(output)
    return input + output

  def forward(self, hidden_states, attention_mask):
    # --- Attention sub-layer (pre-norm) ---
    # LayerNorm → Self-Attention → Dense projection + Dropout + Residual
    normed = self.attention_layer_norm(hidden_states)
    attn_out = self.self_attention(normed, attention_mask)
    hidden_states = self.add(hidden_states, attn_out, self.attention_dense, self.attention_dropout)

    # --- Feed-forward sub-layer (pre-norm) ---
    # LayerNorm → FFN (up-project → GELU) → Dense projection + Dropout + Residual
    normed = self.out_layer_norm(hidden_states)
    ffn_out = self.interm_dense(normed)
    ffn_out = self.interm_af(ffn_out)
    hidden_states = self.add(hidden_states, ffn_out, self.out_dense, self.out_dropout)

    return hidden_states

