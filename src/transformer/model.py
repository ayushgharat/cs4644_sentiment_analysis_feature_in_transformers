"""
Small decoder-only transformer for next-token prediction.

Architecture (matching course spec):
  - 4 layers, hidden size 256, 8 attention heads, max seq len 128
  - RMSNorm, RoPE, SwiGLU FFN, tied embeddings
  - Cross-entropy next-token loss (no sentiment labels)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TransformerConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / norm


def precompute_rope(d_head: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 10000.0 ** (-torch.arange(0, d_head, 2).float() / d_head)
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, theta)          # (T, d_head/2)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, d_head)
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    # Rotate pairs: [-x1, x0, -x3, x2, ...]
    x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1)
    x2 = x2.flatten(-2)
    return x * cos + x2 * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        cos, sin = precompute_rope(self.d_head, config.max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)                    # each (B, T, n_heads, d_head)
        q = q.transpose(1, 2)                       # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                 # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_f = RMSNorm(config.d_model)
        # Tied output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Causal mask
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[torch.Tensor] | None]:
        x = self.emb_dropout(self.tok_emb(input_ids))

        hidden_states = []
        for block in self.blocks:
            x = block(x, self.causal_mask)
            if return_hidden_states:
                hidden_states.append(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, hidden_states if return_hidden_states else None

    @torch.no_grad()
    def get_layer_activations(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Return residual stream activations after layer `layer_idx`."""
        x = self.emb_dropout(self.tok_emb(input_ids))
        for i, block in enumerate(self.blocks):
            x = block(x, self.causal_mask)
            if i == layer_idx:
                return x
        return x

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits, _, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
