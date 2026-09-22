"""
Spot-check verification (shared-VRF, no retraining of the full block).

Per-block training is split into ``num_steps`` deterministic gradient steps. The miner
publishes:

- ``state_0_hash`` (anchor: must equal the previous block's final state hash)
- ``state_N_hash`` (final model)
- full ``loss_curve`` (length ``num_steps + 1``)
- ``gradient_hash_chain`` (length ``num_steps``; rolling chain over per-step gradient hashes)
- ``gradient_norms``

All intermediate states / gradients are kept locally and revealed only on challenge.

Verification combines:

1. **Structural** checks (mandatory, free)
2. **Statistical** heuristics on the published curves (cheap, every verifier)
3. **Spot replay** of ``k`` step indices selected from a *post-publication* shared VRF
   seed (every honest verifier replays the same ``k`` steps and votes; ``≥66%`` agreement
   ⇒ accepted)
4. **Model-level strike** book-keeping (no per-step monotonicity required)

This module is intentionally chaincraft-/torch-agnostic: it operates on hex / hashes /
floats and accepts a ``replay_fn`` callback for step replay so trainers stay decoupled.
See :mod:`dessin.llm.decoder_lm_training_trace` for a torch-based reference implementation.

Optimizer assumption (v1):
    The witness schema (:class:`TrainingTracePrivate`: per-step pre-state + gradient)
    is sized for a **stateless** optimizer such as plain SGD, where step ``j``'s update
    depends only on ``(θ_{j-1}, lr, g_j)``. Stateful optimizers like AdamW additionally
    depend on running moments ``(m_{j-1}, v_{j-1}, t)`` and would require:

    1. Extending :class:`TrainingTracePrivate` with per-step optimizer state and a
       hash chain over those buffers (so they cannot be forged).
    2. Updating any ``replay_fn`` to reconstruct that state for the challenged step.
    3. Pinning device/dtype (CPU + float64 typically) or declaring a tolerance, since
       AdamW updates can drift bit-wise across BLAS / CUDA backends.

    See ``docs/SPOT_CHECK_VERIFICATION.md`` for the migration plan.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


__all__ = [
    "TrainingTrace",
    "TrainingTracePrivate",
    "StatisticalThresholds",
    "SpotCheckResult",
    "ReplayOutcome",
    "ChainStrikeState",
    "compute_grad_hash_chain",
    "derive_challenge_step_indices",
    "verify_structural",
    "verify_statistical",
    "verify_spot_replay",
    "accumulate_strikes",
]


_GRAD_CHAIN_DOMAIN = b"DESSIN/spot-check/grad-chain/v1"
_CHALLENGE_DOMAIN = b"DESSIN/spot-check/challenge-step/v1"


@dataclass(frozen=True)
class TrainingTrace:
    """
    Public, on-chain (or on-block-sidecar) commitments for a training block.

    All hashes are lowercase hex SHA-256.

    ``loss_curve`` has length ``num_steps + 1`` and is interpreted as: ``loss_curve[i]`` is
    the loss measured on the **step ``i+1`` minibatch** (using ``data_seed`` and step index)
    against the model state ``state_i``; ``loss_curve[num_steps]`` is the post-final-eval
    loss on the same eval seed used for ``loss_curve[0]``.

    ``gradient_hash_chain[i]`` (``i = 0..num_steps - 1``) is::

        sha256(_GRAD_CHAIN_DOMAIN || prev_chain || sha256(grad_bytes_i))

    with ``prev_chain`` being ``32 * 0x00`` for ``i == 0`` and ``gradient_hash_chain[i-1]``
    bytes otherwise.
    """

    model_id: str
    num_steps: int
    state_0_hash: str
    state_N_hash: str
    loss_curve: Tuple[float, ...]
    gradient_hash_chain: Tuple[str, ...]
    gradient_norms: Tuple[float, ...]
    data_seed: int
    learning_rate: float
    batch_size: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id,
            "num_steps": int(self.num_steps),
            "state_0_hash": self.state_0_hash,
            "state_N_hash": self.state_N_hash,
            "loss_curve": list(self.loss_curve),
            "gradient_hash_chain": list(self.gradient_hash_chain),
            "gradient_norms": list(self.gradient_norms),
            "data_seed": int(self.data_seed),
            "learning_rate": float(self.learning_rate),
            "batch_size": int(self.batch_size),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TrainingTrace":
        return cls(
            model_id=str(data["model_id"]),
            num_steps=int(data["num_steps"]),
            state_0_hash=str(data["state_0_hash"]),
            state_N_hash=str(data["state_N_hash"]),
            loss_curve=tuple(float(x) for x in data["loss_curve"]),  # type: ignore[arg-type]
            gradient_hash_chain=tuple(str(x) for x in data["gradient_hash_chain"]),  # type: ignore[arg-type]
            gradient_norms=tuple(float(x) for x in data["gradient_norms"]),  # type: ignore[arg-type]
            data_seed=int(data["data_seed"]),  # type: ignore[arg-type]
            learning_rate=float(data["learning_rate"]),  # type: ignore[arg-type]
            batch_size=int(data["batch_size"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TrainingTracePrivate:
    """
    Off-chain witness held by the miner. Revealed on challenge for the chosen step indices.

    ``state_hex_before[i]`` is the hex weights at the START of step ``i+1`` (so
    ``state_hex_before[0]`` reconstructs ``state_0_hash``, ``state_hex_before[num_steps - 1]``
    is the input to the LAST step, and the post-state of step ``num_steps`` reconstructs
    ``state_N_hash``).

    ``grad_hex[i]`` is the hex of the (flattened, float64) gradient used at step ``i+1``.
    """

    state_hex_before: Tuple[str, ...]  # length: num_steps
    grad_hex: Tuple[str, ...]  # length: num_steps


@dataclass
class StatisticalThresholds:
    """Heuristic bounds for the published curves; tune via env / config."""

    min_grad_norm: float = 1e-9
    max_grad_norm: float = 1e6
    max_loss_jump_ratio: float = 50.0  # |L_{i+1} - L_i| / max(|L_i|, 1) bound per step
    require_loss_curve_finite: bool = True
    # If both gradient norms and |Δloss| have spread, an honest training run usually shows
    # *positive* correlation between the two (large step ⇒ larger loss change). Strongly
    # negative correlation across the curve is a red flag; we only flag *very* negative
    # correlation to avoid false positives on tiny budgets.
    min_grad_loss_corr: float = -0.9


@dataclass(frozen=True)
class SpotCheckResult:
    accepted: bool
    structural_ok: bool
    statistical_ok: bool
    replays_ok: bool
    challenge_step_indices: Tuple[int, ...]
    failures: Tuple[str, ...]


@dataclass(frozen=True)
class ReplayOutcome:
    """What a callback returns after re-running step ``j`` from ``state_hex_before``."""

    grad_bytes: bytes
    grad_norm: float
    state_hex_after: str


@dataclass
class ChainStrikeState:
    """Per-model strike counter state — update one entry per accepted block."""

    strikes_by_model: Dict[str, int] = field(default_factory=dict)
    frozen_models: set = field(default_factory=set)

    def is_frozen(self, model_id: str) -> bool:
        return model_id in self.frozen_models

    def reset(self, model_id: str) -> None:
        self.strikes_by_model[model_id] = 0
        self.frozen_models.discard(model_id)


def compute_grad_hash_chain(grad_bytes_per_step: List[bytes]) -> List[str]:
    """Reference rolling SHA-256 chain over per-step flattened gradients."""
    chain: List[str] = []
    prev = bytes(32)
    for g in grad_bytes_per_step:
        gh = hashlib.sha256(g).digest()
        h = hashlib.sha256(_GRAD_CHAIN_DOMAIN + prev + gh).digest()
        prev = h
        chain.append(h.hex())
    return chain


def _hkdf_like_int_stream(seed: bytes, info: bytes, count: int) -> List[int]:
    """Deterministic 64-bit unsigned ints derived from ``seed``+``info`` via HMAC-SHA256."""
    out: List[int] = []
    counter = 0
    while len(out) < count:
        block = hmac.new(seed, info + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        for i in range(0, len(block), 8):
            chunk = block[i : i + 8]
            if len(chunk) < 8:
                break
            out.append(int.from_bytes(chunk, "big"))
            if len(out) == count:
                break
        counter += 1
    return out


def derive_challenge_step_indices(
    seed: bytes,
    *,
    num_steps: int,
    k: int = 2,
) -> List[int]:
    """
    Deterministically draw ``k`` step indices in ``[1, num_steps]`` from a shared seed.

    ``seed`` MUST be a *post-publication* commitment (e.g. hash of the block published
    after this training block) so the miner cannot bias which steps get challenged.
    Indices may repeat — they're independent draws and that is fine for spot-check
    soundness.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    raw = _hkdf_like_int_stream(seed, _CHALLENGE_DOMAIN, k)
    return [int(r % num_steps) + 1 for r in raw]


def verify_structural(
    trace: TrainingTrace,
    *,
    expected_num_steps: int,
    prev_block_state_hash: str,
    claimed_state_N_hash: str,
) -> Tuple[bool, List[str]]:
    """Free, deterministic checks on the published commitments."""
    failures: List[str] = []

    if trace.num_steps != expected_num_steps:
        failures.append(
            f"num_steps mismatch: trace={trace.num_steps} expected={expected_num_steps}"
        )

    if len(trace.loss_curve) != trace.num_steps + 1:
        failures.append(
            f"loss_curve length {len(trace.loss_curve)} != num_steps+1 ({trace.num_steps + 1})"
        )

    if len(trace.gradient_hash_chain) != trace.num_steps:
        failures.append(
            f"gradient_hash_chain length {len(trace.gradient_hash_chain)} != num_steps ({trace.num_steps})"
        )

    if len(trace.gradient_norms) != trace.num_steps:
        failures.append(
            f"gradient_norms length {len(trace.gradient_norms)} != num_steps ({trace.num_steps})"
        )

    if trace.state_0_hash != prev_block_state_hash:
        failures.append(
            f"state_0_hash {trace.state_0_hash[:12]}… does not anchor to previous block "
            f"{prev_block_state_hash[:12]}…"
        )

    if trace.state_N_hash != claimed_state_N_hash:
        failures.append(
            f"state_N_hash {trace.state_N_hash[:12]}… != claimed final state "
            f"{claimed_state_N_hash[:12]}…"
        )

    for h in [trace.state_0_hash, trace.state_N_hash, *trace.gradient_hash_chain]:
        if not isinstance(h, str) or len(h) != 64:
            failures.append(f"malformed sha256 hex: {h!r}")
            break

    return (not failures), failures


def verify_statistical(
    trace: TrainingTrace,
    thresholds: Optional[StatisticalThresholds] = None,
) -> Tuple[bool, List[str]]:
    """Cheap heuristics that flag obviously synthetic curves."""
    th = thresholds or StatisticalThresholds()
    failures: List[str] = []

    if th.require_loss_curve_finite:
        for i, L in enumerate(trace.loss_curve):
            if not math.isfinite(L):
                failures.append(f"loss_curve[{i}]={L!r} is not finite")

    for i, gn in enumerate(trace.gradient_norms):
        if not math.isfinite(gn):
            failures.append(f"gradient_norms[{i}]={gn!r} is not finite")
            continue
        if gn < th.min_grad_norm:
            failures.append(
                f"gradient_norms[{i}]={gn:.3e} below min {th.min_grad_norm:.3e}"
            )
        if gn > th.max_grad_norm:
            failures.append(
                f"gradient_norms[{i}]={gn:.3e} above max {th.max_grad_norm:.3e}"
            )

    if len(trace.loss_curve) >= 2:
        for i in range(len(trace.loss_curve) - 1):
            a = trace.loss_curve[i]
            b = trace.loss_curve[i + 1]
            denom = max(abs(a), 1.0)
            if math.isfinite(a) and math.isfinite(b):
                jump = abs(b - a) / denom
                if jump > th.max_loss_jump_ratio:
                    failures.append(
                        f"loss jump {jump:.2f} between steps {i}->{i+1} "
                        f"exceeds max_loss_jump_ratio={th.max_loss_jump_ratio}"
                    )

    n = len(trace.gradient_norms)
    if n >= 3 and len(trace.loss_curve) == n + 1:
        deltas = [
            abs(trace.loss_curve[i + 1] - trace.loss_curve[i]) for i in range(n)
        ]
        gn = list(trace.gradient_norms)
        if max(gn) - min(gn) > 0 and max(deltas) - min(deltas) > 0:
            corr = _pearson(gn, deltas)
            if corr < th.min_grad_loss_corr:
                failures.append(
                    f"grad-norm vs |Δloss| correlation {corr:.3f} below "
                    f"min_grad_loss_corr={th.min_grad_loss_corr}"
                )

    return (not failures), failures


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _grad_chain_step(prev_chain_hex_or_zero: str, grad_bytes: bytes) -> str:
    if prev_chain_hex_or_zero == "":
        prev = bytes(32)
    else:
        prev = bytes.fromhex(prev_chain_hex_or_zero)
    gh = hashlib.sha256(grad_bytes).digest()
    return hashlib.sha256(_GRAD_CHAIN_DOMAIN + prev + gh).hexdigest()


def verify_spot_replay(
    trace: TrainingTrace,
    private: TrainingTracePrivate,
    *,
    challenge_step_indices: List[int],
    replay_fn: Callable[[int, str], ReplayOutcome],
    state_hash_fn: Callable[[str], str] = lambda h: hashlib.sha256(bytes.fromhex(h)).hexdigest(),
    grad_hash_chain_recompute: bool = True,
    grad_norm_tolerance: float = 1e-3,
) -> Tuple[bool, List[str]]:
    """
    For each challenge step index ``j``, replay step ``j`` from the witnessed pre-state
    and verify it reconstructs the published state, gradient hash chain entry, and norm.

    ``replay_fn(j, state_hex_before)`` MUST be a deterministic re-run of step ``j`` of
    the original training (same ``data_seed``, ``learning_rate``, ``batch_size`` from
    ``trace``). It returns the gradient bytes (flattened, float64), its L2 norm, and
    the resulting state hex after applying the optimizer.
    """
    failures: List[str] = []
    n = trace.num_steps
    if len(private.state_hex_before) != n or len(private.grad_hex) != n:
        return False, [
            f"private witness lengths mismatch (state={len(private.state_hex_before)} "
            f"grad={len(private.grad_hex)} num_steps={n})"
        ]

    for j in challenge_step_indices:
        if j < 1 or j > n:
            failures.append(f"challenge step index {j} out of range [1,{n}]")
            continue

        idx = j - 1
        pre_state_hex = private.state_hex_before[idx]
        pre_state_hash = state_hash_fn(pre_state_hex)

        if idx == 0 and pre_state_hash != trace.state_0_hash:
            failures.append(
                f"step {j}: provided pre-state does not match state_0_hash"
            )
            continue

        outcome = replay_fn(j, pre_state_hex)

        observed_grad_hash = hashlib.sha256(outcome.grad_bytes).hexdigest()
        claimed_grad_hash = hashlib.sha256(bytes.fromhex(private.grad_hex[idx])).hexdigest()
        if observed_grad_hash != claimed_grad_hash:
            failures.append(
                f"step {j}: replayed gradient hash {observed_grad_hash[:12]}… "
                f"!= witnessed {claimed_grad_hash[:12]}…"
            )
            continue

        if grad_hash_chain_recompute:
            prev_chain = "" if idx == 0 else trace.gradient_hash_chain[idx - 1]
            recomputed_chain_entry = _grad_chain_step(prev_chain, outcome.grad_bytes)
            if recomputed_chain_entry != trace.gradient_hash_chain[idx]:
                failures.append(
                    f"step {j}: rolling chain entry mismatch "
                    f"(replay={recomputed_chain_entry[:12]}… "
                    f"trace={trace.gradient_hash_chain[idx][:12]}…)"
                )
                continue

        claimed_norm = trace.gradient_norms[idx]
        if abs(outcome.grad_norm - claimed_norm) > grad_norm_tolerance * max(claimed_norm, 1.0):
            failures.append(
                f"step {j}: replayed grad norm {outcome.grad_norm:.4f} "
                f"!= claimed {claimed_norm:.4f}"
            )
            continue

        post_state_hash = state_hash_fn(outcome.state_hex_after)
        if idx + 1 < n:
            expected_post_hash = state_hash_fn(private.state_hex_before[idx + 1])
        else:
            expected_post_hash = trace.state_N_hash
        if post_state_hash != expected_post_hash:
            failures.append(
                f"step {j}: replayed post-state hash {post_state_hash[:12]}… "
                f"!= expected {expected_post_hash[:12]}…"
            )
            continue

    return (not failures), failures


def accumulate_strikes(
    state: ChainStrikeState,
    *,
    model_id: str,
    trace: TrainingTrace,
    max_strikes: int = 3,
) -> Tuple[int, bool]:
    """
    Model-level **usefulness** ledger: if the 100-step run never dipped below ``L_0``,
    increment the strike counter; on 3 strikes the model is frozen until refresh.
    Any block whose loss curve dips below ``L_0`` resets the counter.

    Returns ``(new_strike_count, became_frozen_now)``.
    """
    if not trace.loss_curve:
        return state.strikes_by_model.get(model_id, 0), False

    L0 = trace.loss_curve[0]
    dipped = any(L < L0 for L in trace.loss_curve[1:]) if math.isfinite(L0) else False

    cur = state.strikes_by_model.get(model_id, 0)
    became_frozen = False
    if dipped:
        state.strikes_by_model[model_id] = 0
    else:
        cur += 1
        state.strikes_by_model[model_id] = cur
        if cur >= max_strikes and model_id not in state.frozen_models:
            state.frozen_models.add(model_id)
            became_frozen = True
    return state.strikes_by_model[model_id], became_frozen
