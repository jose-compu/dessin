"""
LoRA Fine-Tuning for DecoderOnlyGPT
====================================

Implements Low-Rank Adaptation (LoRA) in **pure PyTorch** — no external
``peft`` or ``accelerate`` library required.  Works with the existing
:class:`~dessin.llm.decoder_only_gpt.DecoderOnlyGPT` architecture that is
already used for PoGO training.

Design
------
LoRA replaces a frozen :class:`~torch.nn.Linear` weight matrix ``W`` with:

    output = x @ W.T  +  x @ A.T @ B.T * (alpha / rank)

where ``A`` and ``B`` are small trainable matrices.  The original ``W`` is
**frozen** (``requires_grad = False``).  Only adapter weights are updated
during fine-tuning.

Integration with PoGO
---------------------
The training function :func:`train_lora` returns a standard
:class:`~dessin.llm.simple_trainer.TrainingResult` with the **merged**
weights (base + LoRA delta) serialised as hex-encoded float64.  This means
the output checkpoint is fully compatible with the existing DeSSIN block /
BitTorrent distribution pipeline.

Usage
-----
::

    from dessin.llm.lora_trainer import (
        apply_lora_to_model, train_lora, LoRAConfig, strip_lora
    )

    model = DecoderOnlyGPT(vocab_size=65, block_size=32, n_layer=2,
                           n_head=2, n_embd=64)
    cfg   = LoRAConfig(rank=4, alpha=8.0, target_modules={"qkv", "proj"})
    apply_lora_to_model(model, cfg)

    result = train_lora(model, corpus_text, cfg, steps=200)
    # result.model_weights_hex  ← merged, drop-in compatible with load_params_from_hex
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_only_gpt import (
    CharTokenizer,
    DecoderOnlyGPT,
    eval_lm_loss,
    flatten_params_float64,
    load_params_from_hex,
)
from .simple_trainer import TrainingResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adaptation.

    Parameters
    ----------
    rank:
        Low-rank dimension ``r``.  Higher = more capacity, more params.
        Typical values: 4, 8, 16, 32.
    alpha:
        LoRA scaling factor.  The effective scale is ``alpha / rank``.
        A common heuristic: ``alpha = 2 * rank``.
    target_modules:
        Names of :class:`~torch.nn.Linear` layers to wrap.  For
        :class:`~dessin.llm.decoder_only_gpt.DecoderOnlyGPT` the relevant
        names are ``"qkv"`` (combined Q/K/V projection),
        ``"proj"`` (attention output), and the two MLP linears
        (accessible via ``"0"`` and ``"2"`` under each block's ``mlp``).
    dropout:
        Applied to the LoRA path input before multiplication.
    merge_on_export:
        When ``True``, :func:`train_lora` merges adapters into the base
        weights before exporting.  The output is then compatible with
        ``load_params_from_hex`` on a plain ``DecoderOnlyGPT``.
    """

    rank: int = 8
    alpha: float = 16.0
    target_modules: Set[str] = field(default_factory=lambda: {"qkv", "proj"})
    dropout: float = 0.0
    merge_on_export: bool = True


# ---------------------------------------------------------------------------
# LoRALinear — drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    ``W_out = W_frozen + B @ A * scale``

    where ``A ∈ R^{rank × d_in}``, ``B ∈ R^{d_out × rank}``.

    Initialisation follows the original LoRA paper:
    - ``A`` initialised from ``N(0, 0.01)``
    - ``B`` initialised to **zero** → zero-init means the adapter starts as
      a no-op, so training begins from the same point as a non-adapted model.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.scale = alpha / rank

        # Frozen base weights (not trainable)
        self.weight = linear.weight  # shared; do NOT deepcopy
        self.bias = linear.bias

        # Trainable adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_in = self.dropout(x) if self.dropout is not None else x
        # lora_in @ A.T → (batch, seq, rank), then @ B.T → (batch, seq, d_out)
        delta = F.linear(F.linear(lora_in, self.lora_A), self.lora_B) * self.scale
        return base + delta

    def merge(self) -> nn.Linear:
        """
        Return a plain :class:`~torch.nn.Linear` with the LoRA delta
        added into the weight matrix.  Used before exporting weights.
        """
        merged_w = self.weight + (self.lora_B @ self.lora_A) * self.scale
        linear = nn.Linear(self.d_in, self.d_out, bias=self.bias is not None)
        linear.weight = nn.Parameter(merged_w.detach().clone())
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.detach().clone())
        return linear

    @property
    def merged_weight(self) -> torch.Tensor:
        return self.weight + (self.lora_B @ self.lora_A) * self.scale


# ---------------------------------------------------------------------------
# Apply / strip helpers
# ---------------------------------------------------------------------------

def apply_lora_to_model(
    model: DecoderOnlyGPT,
    cfg: LoRAConfig,
) -> Dict[str, LoRALinear]:
    """
    Replace targeted ``nn.Linear`` sub-modules in ``model`` with
    :class:`LoRALinear` in-place.

    Freezes **all** base-model parameters; only the LoRA adapters are
    left with ``requires_grad = True``.

    Returns
    -------
    dict
        Mapping of module path → :class:`LoRALinear` (for inspection /
        adapter-only serialisation).
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    adapted: Dict[str, LoRALinear] = {}

    def _replace(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and name in cfg.target_modules:
                lora = LoRALinear(child, cfg.rank, cfg.alpha, cfg.dropout)
                setattr(parent, name, lora)
                adapted[full_name] = lora
            else:
                _replace(child, full_name)

    _replace(model, "")
    return adapted


def strip_lora(model: DecoderOnlyGPT) -> None:
    """
    Merge all :class:`LoRALinear` modules back into plain
    :class:`~torch.nn.Linear` in-place and re-enable all gradients.

    After this call the model is identical to a freshly trained
    ``DecoderOnlyGPT`` — ready for serialisation with
    :func:`~dessin.llm.decoder_only_gpt.flatten_params_float64`.
    """
    def _merge(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merge())
            else:
                _merge(child)

    _merge(model)
    for p in model.parameters():
        p.requires_grad_(True)


def lora_param_count(model: nn.Module) -> Tuple[int, int]:
    """Return ``(trainable_params, total_params)``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class LoRATrainingResult:
    """
    Extended result from :func:`train_lora`.

    ``training_result`` is a standard :class:`~dessin.llm.simple_trainer.TrainingResult`
    with the merged weights — fully compatible with the PoGO block pipeline.
    ``adapter_weights`` is a raw state-dict of the LoRA matrices only
    (before merging), useful for sharing adapters separately.
    ``initial_loss`` / ``final_loss`` are scalar floats.
    ``loss_curve`` is the per-step loss list for PoGO verification.
    ``trainable_params`` / ``total_params`` help verify the savings.
    """

    training_result: TrainingResult
    adapter_weights: Dict[str, torch.Tensor]
    initial_loss: float
    final_loss: float
    loss_curve: List[float]
    trainable_params: int
    total_params: int
    rank: int
    alpha: float

    @property
    def param_efficiency(self) -> float:
        """Fraction of parameters actually trained (lower = more efficient)."""
        if self.total_params == 0:
            return 0.0
        return self.trainable_params / self.total_params


def train_lora(
    model: DecoderOnlyGPT,
    corpus_text: str,
    cfg: LoRAConfig,
    *,
    steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    device: Optional[torch.device] = None,
    checkpoint_weights_hex: Optional[str] = None,
    seed: int = 42,
) -> LoRATrainingResult:
    """
    Fine-tune ``model`` with LoRA adapters using the char LM objective.

    The base model is kept **frozen**; only LoRA adapter parameters are
    updated.  After training the adapters are optionally merged back into
    the base weights (controlled by ``cfg.merge_on_export``).

    Parameters
    ----------
    model:
        A ``DecoderOnlyGPT`` instance (already constructed and optionally
        pre-loaded with a checkpoint via ``checkpoint_weights_hex``).
        :func:`apply_lora_to_model` is called internally — do **not** call
        it before passing the model here.
    corpus_text:
        Plain-text corpus to fine-tune on (character-level tokenisation).
    cfg:
        LoRA hyper-parameters.
    steps:
        Number of gradient update steps.
    batch_size:
        Sequences per gradient step.
    learning_rate:
        Adam learning rate for adapter weights.
    device:
        Target device (default: ``cpu``).
    checkpoint_weights_hex:
        Optional hex checkpoint to load into ``model`` before adapting.
    seed:
        Reproducibility seed.

    Returns
    -------
    LoRATrainingResult
        Contains a standard ``TrainingResult`` (merged weights as hex) plus
        adapter-only weights and training diagnostics.
    """
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)

    # Optional checkpoint restore
    if checkpoint_weights_hex:
        load_params_from_hex(model, checkpoint_weights_hex)

    model = model.to(device)

    # Build tokeniser
    tokenizer = CharTokenizer(corpus_text)
    block_size = model.block_size

    # Apply LoRA (freezes base weights)
    adapted = apply_lora_to_model(model, cfg)
    trainable, total = lora_param_count(model)

    # Collect only trainable (adapter) parameters
    adapter_params: List[nn.Parameter] = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(adapter_params, lr=learning_rate)

    g = torch.Generator()
    g.manual_seed(seed)

    def _get_batch() -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(tokenizer.data) - block_size - 1
        ix = torch.randint(0, max(1, n), (batch_size,), generator=g)
        x = torch.stack([tokenizer.data[i: i + block_size] for i in ix]).to(device)
        y = torch.stack([tokenizer.data[i + 1: i + block_size + 1] for i in ix]).to(device)
        return x, y

    # Evaluate initial loss (frozen base)
    initial_loss = eval_lm_loss(model, tokenizer, device, batch_size, block_size, seed)

    loss_curve: List[float] = []
    model.train()

    for step in range(steps):
        xb, yb = _get_batch()
        _, loss = model(xb, yb)
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_curve.append(float(loss.item()))

    model.eval()
    final_loss = eval_lm_loss(model, tokenizer, device, batch_size, block_size, seed + 1)

    # Snapshot adapter weights before merge (A and B have different shapes)
    adapter_snapshot: Dict[str, torch.Tensor] = {}
    for path, lora in adapted.items():
        adapter_snapshot[f"{path}.lora_A"] = lora.lora_A.detach().clone()
        adapter_snapshot[f"{path}.lora_B"] = lora.lora_B.detach().clone()

    # Merge adapters into base weights for export
    if cfg.merge_on_export:
        strip_lora(model)

    # Serialise merged model using the standard PoGO hex format
    import json

    weights_hex, pogo_layer_shapes, nbytes = flatten_params_float64(model)

    arch = {
        "schema_version": "1.0",
        "family": "decoder_only_transformer",
        "variant": "dessin_native_lora",
        "n_layer": len(model.blocks),
        "n_head": model.blocks[0].attn.n_head,
        "n_embd": model.blocks[0].attn.n_embd,
        "block_size": model.block_size,
        "vocab_size": model.token_emb.num_embeddings,
        "lora_rank": cfg.rank,
        "lora_alpha": cfg.alpha,
        "lora_target_modules": sorted(cfg.target_modules),
        "merged": cfg.merge_on_export,
        "pogo_layer_shapes": [list(s) for s in pogo_layer_shapes],
    }
    tdata = {
        "kind": "lora_character_lm",
        "corpus_hash": tokenizer.text_hash(),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "target_modules": sorted(cfg.target_modules),
    }

    training_result = TrainingResult(
        loss_before=initial_loss,
        loss_after=final_loss,
        model_weights_hex=weights_hex,
        model_size_bytes=nbytes,
        training_steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        training_data_hash=tokenizer.text_hash(),
        pogo_layer_shapes=pogo_layer_shapes,
        model_architecture_json=arch,
        training_data_json=tdata,
    )

    return LoRATrainingResult(
        training_result=training_result,
        adapter_weights=adapter_snapshot,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_curve=loss_curve,
        trainable_params=trainable,
        total_params=total,
        rank=cfg.rank,
        alpha=cfg.alpha,
    )
