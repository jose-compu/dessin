"""
Decoder-only GPT (GPT-2 / nanoGPT-style): causal Transformer + language-model head.

Shared by :mod:`dessin.llm.micro_gpt_trainer` (tiny preset) and :mod:`dessin.llm.gpt2_nano_trainer`.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    def __init__(self, text: str):
        self.raw_text = text
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def text_hash(self) -> str:
        return hashlib.sha256(self.raw_text.encode()).hexdigest()


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        nh = self.n_head
        hd = C // nh
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyGPT(nn.Module):
    """GPT-2-class decoder-only LM (nanoGPT-compatible module layout)."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            Block(n_embd, n_head, block_size) for _ in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


TinyCharGPT = DecoderOnlyGPT


def flatten_params_float64(model: nn.Module) -> Tuple[str, List[Tuple[int, ...]], int]:
    parts: List[torch.Tensor] = []
    shapes: List[Tuple[int, ...]] = []
    for p in model.parameters():
        flat = p.detach().cpu().reshape(-1).to(torch.float64)
        parts.append(flat)
        shapes.append((flat.numel(),))
    vec = torch.cat(parts, dim=0)
    raw = vec.numpy().tobytes()
    return raw.hex(), shapes, len(raw)


def load_params_from_hex(model: nn.Module, weights_hex: str) -> None:
    """Load model parameters from hex string (inverse of flatten_params_float64)."""
    raw = bytes.fromhex(weights_hex)
    vec = torch.from_numpy(np.frombuffer(raw, dtype=np.float64).copy())
    
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        if idx + numel > vec.numel():
            raise ValueError(f"Checkpoint size mismatch: expected {vec.numel()} values, exhausted at param with {numel} elements")
        flat = vec[idx:idx + numel].to(p.dtype).to(p.device)
        p.data.copy_(flat.reshape(p.shape))
        idx += numel
    
    if idx != vec.numel():
        raise ValueError(f"Checkpoint size mismatch: consumed {idx} values but checkpoint has {vec.numel()}")


def eval_lm_loss(
    model: DecoderOnlyGPT,
    data: CharTokenizer,
    device: torch.device,
    batch_size: int,
    block_size: int,
    seed: int,
) -> float:
    model.eval()
    g = torch.Generator()
    g.manual_seed(int(seed))
    n = len(data.data) - block_size - 1
    if n <= 0:
        return 0.0
    ix = torch.randint(0, n, (batch_size,), generator=g)
    x = torch.stack([data.data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data.data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    with torch.no_grad():
        _, loss = model(x, y)
    return float(loss.item()) if loss is not None else 0.0
