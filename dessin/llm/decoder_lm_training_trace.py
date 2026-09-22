"""
Char-LM training that emits a :class:`~dessin.consensus.spot_check_verification.TrainingTrace`.

Mirrors :mod:`dessin.llm.decoder_lm_training` step-for-step but, at every gradient step,
records:

- the pre-step weights (hex, float64-flattened),
- the gradient (hex, float64-flattened),
- the gradient L2 norm,
- the per-step minibatch loss.

Used for the spot-check verification path. The verifier replays only the few step
indices selected by a shared post-publication VRF seed (see
:func:`dessin.consensus.spot_check_verification.derive_challenge_step_indices`).

Optimizer choice — **plain SGD on purpose** (see also ``docs/SPOT_CHECK_VERIFICATION.md``):

The update rule is ``θ_j = θ_{j-1} - lr · g_j``. Replaying a single step ``j`` in
isolation needs only ``(state_hex_before, data_seed, batch_size, learning_rate)``;
**no per-parameter optimizer state is required** in the witness.

Switching to AdamW would:

1. Triple the witness footprint (each step would also need ``m`` and ``v`` buffers and
   the step counter ``t`` for bias correction; with decoupled weight decay there is an
   additional coupled term).
2. Require :class:`~dessin.consensus.spot_check_verification.TrainingTracePrivate` to carry a
   per-step optimizer-state commitment (hash chain) so a verifier can replay step
   ``j`` from the *exact* ``(θ_{j-1}, m_{j-1}, v_{j-1}, t)`` the miner used; otherwise
   an adversary could forge ``θ_j`` by tweaking ``m, v``.
3. Stress determinism: AdamW under different PyTorch/CUDA/BLAS combos can differ at
   the last few bits. Replay would either need to be pinned to CPU + ``float64`` or
   declare a documented tolerance, and the verifier would have to enforce that.

Until those follow-ups land, this trainer must use SGD; do not silently switch the
optimizer here without also extending the trace schema and replay function.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .decoder_lm_training import BUNDLED_SHAKESPEARE_CHAR_CORPUS, _resolve_device, _resolve_hparams
from .decoder_only_gpt import (
    CharTokenizer,
    DecoderOnlyGPT,
    eval_lm_loss,
    flatten_params_float64,
    load_params_from_hex,
)
from ..models.llm_model_spec import (
    DecoderOnlyTransformerSpec,
    LLMVariant,
    TokenizerKind,
    TokenizerProfile,
    resolve_char_lm_tokenizer_options,
)
from .simple_trainer import TrainingResult
from ..consensus.spot_check_verification import (
    ReplayOutcome,
    TrainingTrace,
    TrainingTracePrivate,
    compute_grad_hash_chain,
)

__all__ = [
    "TraceTrainingArtifacts",
    "train_decoder_only_char_lm_with_trace",
    "make_replay_fn",
]


@dataclass
class TraceTrainingArtifacts:
    """Bundle returned by :func:`train_decoder_only_char_lm_with_trace`."""

    result: TrainingResult
    trace: TrainingTrace
    private: TrainingTracePrivate


def _flatten_grads_float64(model: torch.nn.Module) -> Tuple[bytes, float]:
    parts = []
    sq = 0.0
    for p in model.parameters():
        g = p.grad
        if g is None:
            g = torch.zeros_like(p)
        flat = g.detach().cpu().reshape(-1).to(torch.float64)
        parts.append(flat)
        sq += float((flat * flat).sum().item())
    if not parts:
        return b"", 0.0
    vec = torch.cat(parts, dim=0)
    return vec.numpy().tobytes(), float(sq) ** 0.5


def _state_dict_snapshot(model: torch.nn.Module) -> str:
    hex_str, _, _ = flatten_params_float64(model)
    return hex_str


def train_decoder_only_char_lm_with_trace(
    variant: LLMVariant,
    *,
    model_id: str,
    data_seed: int,
    training_steps: int,
    batch_size: int,
    learning_rate: float,
    env_prefix: str,
    device: Optional[str] = None,
    n_layer: Optional[int] = None,
    n_head: Optional[int] = None,
    n_embd: Optional[int] = None,
    block_size: Optional[int] = None,
    checkpoint_weights_hex: Optional[str] = None,
    tokenizer_kind: Optional[TokenizerKind] = None,
    tokenizer_profile: Optional[TokenizerProfile] = None,
) -> TraceTrainingArtifacts:
    """
    Train + emit trace. ``training_steps`` is exact (no salt retries here — the spot-check
    path explicitly does not require per-block monotonicity; usefulness is enforced by the
    model-level strike ledger over time).
    """
    device_t = _resolve_device(env_prefix, device)
    tk, tprof = resolve_char_lm_tokenizer_options(
        env_prefix, tokenizer_kind=tokenizer_kind, tokenizer_profile=tokenizer_profile,
    )
    if tk != TokenizerKind.CHARACTER:
        raise ValueError(
            "Trace trainer supports only TokenizerKind.CHARACTER; got "
            f"{tk!r}. Use a BPE-capable trainer when one is wired."
        )
    nl, nh, ne, bs = _resolve_hparams(
        variant, env_prefix, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    )

    torch.manual_seed(int(data_seed) % (2**31 - 1))
    tok = CharTokenizer(BUNDLED_SHAKESPEARE_CHAR_CORPUS)
    model = DecoderOnlyGPT(
        vocab_size=tok.vocab_size, block_size=bs, n_layer=nl, n_head=nh, n_embd=ne,
    ).to(device_t)
    if checkpoint_weights_hex:
        load_params_from_hex(model, checkpoint_weights_hex)

    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # NOTE: SGD is required for the spot-check witness to remain "weights-only".
    # Do not switch to AdamW (or any stateful optimizer) without first extending
    # ``TrainingTracePrivate`` with per-step optimizer-state commitments and updating
    # ``make_replay_fn`` accordingly. See module docstring + docs/SPOT_CHECK_VERIFICATION.md.

    eval_seed = (int(data_seed) * 7919 + 1) % (2**31 - 1)
    loss_before = eval_lm_loss(model, tok, device_t, batch_size, bs, eval_seed)

    n = len(tok.data) - bs - 1
    if n <= 0:
        raise RuntimeError("Corpus too short for block_size")

    state_hex_before_steps = []
    grad_hex_steps = []
    grad_bytes_steps = []
    grad_norms = []
    per_step_loss = []

    state_0_hex = _state_dict_snapshot(model)
    state_0_hash = hashlib.sha256(bytes.fromhex(state_0_hex)).hexdigest()

    model.train()
    for step in range(training_steps):
        pre_state_hex = _state_dict_snapshot(model)
        state_hex_before_steps.append(pre_state_hex)

        g_gen = torch.Generator()
        g_gen.manual_seed(int(data_seed) + (step + 1) * 10007)
        ix = torch.randint(0, n, (batch_size,), generator=g_gen)
        x = torch.stack([tok.data[i : i + bs] for i in ix]).to(device_t)
        y = torch.stack([tok.data[i + 1 : i + bs + 1] for i in ix]).to(device_t)

        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Loss was None during traced training")
        per_step_loss.append(float(loss.item()))
        loss.backward()
        gbytes, gnorm = _flatten_grads_float64(model)
        grad_bytes_steps.append(gbytes)
        grad_hex_steps.append(gbytes.hex())
        grad_norms.append(gnorm)
        opt.step()

    state_N_hex = _state_dict_snapshot(model)
    state_N_hash = hashlib.sha256(bytes.fromhex(state_N_hex)).hexdigest()
    loss_after = eval_lm_loss(model, tok, device_t, batch_size, bs, eval_seed)

    grad_chain = compute_grad_hash_chain(grad_bytes_steps)

    weights_hex = state_N_hex
    nbytes = len(bytes.fromhex(weights_hex))
    spec = DecoderOnlyTransformerSpec(
        variant=variant, tokenizer_kind=tk,
        n_layer=nl, n_head=nh, n_embd=ne, block_size=bs, vocab_size=tok.vocab_size,
        tokenizer_profile=tprof, corpus_id="shakespeare_sonnet1_excerpt_x80",
    )
    arch = spec.to_architecture_dict(pogo_layer_shapes=[[p.numel()] for p in model.parameters()])

    result = TrainingResult(
        loss_before=loss_before,
        loss_after=loss_after,
        model_weights_hex=weights_hex,
        model_size_bytes=nbytes,
        training_steps=training_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        training_data_hash=tok.text_hash(),
        pogo_layer_shapes=[(p.numel(),) for p in model.parameters()],
        model_architecture_json=arch,
        training_data_json={
            "kind": "character_lm",
            "text_hash_sha256": tok.text_hash(),
            "data_seed": int(data_seed),
            "samples_bound": int(n),
            "tokenizer_profile": tprof.value,
            "spot_check": True,
        },
    )

    loss_curve = (loss_before, *per_step_loss[1:], loss_after) if per_step_loss else (loss_before, loss_after)
    if len(loss_curve) != training_steps + 1:
        loss_curve = (loss_before, *per_step_loss, loss_after)
        if len(loss_curve) > training_steps + 1:
            loss_curve = loss_curve[: training_steps + 1]

    trace = TrainingTrace(
        model_id=model_id,
        num_steps=training_steps,
        state_0_hash=state_0_hash,
        state_N_hash=state_N_hash,
        loss_curve=tuple(float(x) for x in loss_curve),
        gradient_hash_chain=tuple(grad_chain),
        gradient_norms=tuple(float(x) for x in grad_norms),
        data_seed=int(data_seed),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
    )
    private = TrainingTracePrivate(
        state_hex_before=tuple(state_hex_before_steps),
        grad_hex=tuple(grad_hex_steps),
    )
    return TraceTrainingArtifacts(result=result, trace=trace, private=private)


def make_replay_fn(
    *,
    variant: LLMVariant,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    data_seed: int,
    batch_size: int,
    learning_rate: float,
    device: Optional[str] = None,
):
    """
    Build a deterministic step replayer matching
    :func:`train_decoder_only_char_lm_with_trace`. Used by verifiers to spot-check a
    single step ``j`` from a witnessed pre-state hex.
    """
    dev = torch.device(device or "cpu")
    tok = CharTokenizer(BUNDLED_SHAKESPEARE_CHAR_CORPUS)
    n = len(tok.data) - block_size - 1
    if n <= 0:
        raise RuntimeError("Corpus too short for block_size")

    def _replay(j: int, pre_state_hex: str) -> ReplayOutcome:
        m = DecoderOnlyGPT(
            vocab_size=tok.vocab_size, block_size=block_size,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(dev)
        load_params_from_hex(m, pre_state_hex)
        opt = torch.optim.SGD(m.parameters(), lr=learning_rate)
        # Single-step SGD replay: depends only on (θ_{j-1}, batch, lr). If the trainer
        # ever moves to AdamW this MUST also accept (m, v, t) from the witness.

        g_gen = torch.Generator()
        g_gen.manual_seed(int(data_seed) + j * 10007)
        ix = torch.randint(0, n, (batch_size,), generator=g_gen)
        x = torch.stack([tok.data[i : i + block_size] for i in ix]).to(dev)
        y = torch.stack([tok.data[i + 1 : i + block_size + 1] for i in ix]).to(dev)

        opt.zero_grad(set_to_none=True)
        _, loss = m(x, y)
        if loss is None:
            raise RuntimeError("Loss was None during replay")
        loss.backward()
        gbytes, gnorm = _flatten_grads_float64(m)
        opt.step()
        post_state_hex, _, _ = flatten_params_float64(m)
        return ReplayOutcome(grad_bytes=gbytes, grad_norm=gnorm, state_hex_after=post_state_hex)

    _ = variant
    return _replay
