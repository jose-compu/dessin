# Block Verification Strategy

## Current Status

> **Important:** DeSSIN currently uses **spot-check verification** (VRF-selected gradient step replays) as its security primitive. A quantization-consistency shortcut was designed but its actual comparison logic is not yet wired in production — the current `verify_block` path performs hash-presence and format checks only, not cryptographic proof of training. This document explains the full roadmap and what is active today.

---

## What Is Active Today

### Step 1 — Structural checks (always on)

Free and mandatory. Verified immediately when a block arrives:

- `state_0_hash` matches the previous block's `state_N_hash` (anchor check).
- `state_N_hash` matches the published model artifact.
- `loss_curve` length equals `num_steps + 1`.
- `gradient_hash_chain` length matches `num_steps`.
- Merkle root fields (`hash_full_model_32`, `hash_quant_4`) are present and non-trivial.

These catch malformed blocks before any compute is spent.

### Step 2 — Statistical heuristics (always on)

Cheap checks on the published loss curve and gradient norms:

- Gradient norms must be in `[spot_check_min_grad_norm, spot_check_max_grad_norm]`.
- Per-step loss jumps must be below `spot_check_max_loss_jump_ratio`.
- Gradient-vs-Δloss correlation must exceed `spot_check_min_grad_loss_corr`.

These flag obvious fabrications (uniform-random gradient norms, impossibly clean loss trajectories) at near-zero cost. They are heuristics — they raise the cost of cheating but do not prove honest training on their own.

### Step 3 — Spot-Check Replay (the security proof)

Verifiers replay $k$ VRF-selected gradient steps **after** block publication (so the miner cannot predict which steps are challenged):

```
seed = VRF(miner_private_key, block_hash)
{i₁, i₂, …, iₖ} = DrawWithoutReplacement(seed, N, k)   # k = 2 by default
```

For each selected step $i_j$:

1. Fetch $\theta_{i_j - 1}$ (weights before that step) from the miner's local trace.
2. Re-run the forward + backward pass with the same batch `data_seed[i_j]`.
3. Check that the resulting gradient hash matches the committed `gradient_hash_chain[i_j]`.

Block is accepted when $\geq 66\%$ of verifiers confirm all replays match.

**Catch probability** (without statistical heuristics):

$$P(\text{cheater caught}) = 1 - \left(\frac{m}{N}\right)^k$$

For $N = 100, m = 2, k = 2$: $P \approx 99.96\%$.

### Enabling spot-check in your node

```python
from dessin.runtime.config import DessinConfig

config = DessinConfig.default()
config.consensus.enable_spot_check_verification = True   # off by default
config.consensus.spot_check_count = 2                    # k — number of step replays
config.consensus.spot_check_steps_per_block = 100        # N — steps miner commits to
config.consensus.spot_check_attestation_threshold = 0.66 # required voter fraction
```

Or via environment variables:

```bash
DESSIN_ENABLE_SPOT_CHECK_VERIFICATION=true \
DESSIN_SPOT_CHECK_COUNT=2 \
python -m dessin.runtime.node
```

---

## Planned: Quantization-Consistency Shortcut

The original design included a faster verification pass:

```
verify: quantize(32-bit weights) ≈ claimed 4-bit weights   (O(n), not O(n × steps))
```

**Why it was attractive:**
- Deterministic: `quantize(x)` always produces the same `y` for the same `x`.
- O(n) vs O(n × steps) for full replay.
- Combined with random Merkle leaf challenges it raises the cost of tampered-layer attacks.

**Current status:**
The miner already computes both Merkle roots (`hash_full_model_32`, `hash_quant_4`) during block production. The verifier currently only checks that these fields are non-zero. The actual quantization consistency comparison (`quantize(32bit) == 4bit?`) is not yet executed in `verify_block` — the call is present but commented out pending DA (data availability) integration so both weight artifacts can be fetched reliably.

When this lands, it will sit **between** the statistical checks and the spot-check replay as a cheap pre-filter — not a replacement for spot-check.

```
[Structural] → [Statistical] → [Quantization consistency] → [Spot replay]
                                      ↑ planned; not yet active
```

---

## Configuration Reference

All fields live in `ConsensusConfig` (`dessin.runtime.config`):

| Field | Default | Description |
|---|---|---|
| `enable_spot_check_verification` | `False` | Gate for the spot-replay path |
| `spot_check_count` | `2` | `k` — steps replayed per verifier |
| `spot_check_steps_per_block` | `100` | `N` — steps the miner commits to |
| `spot_check_attestation_threshold` | `0.66` | Required voter fraction |
| `spot_check_max_strikes` | `3` | Model freezes after this many unproductive blocks |
| `enable_efficient_verification` | `True` | Enables structural + format checks (always recommended) |
| `merkle_challenge_count` | `5` | Merkle leaf challenges (used when quantization path activates) |

---

## Implementation Locations

| Component | File |
|---|---|
| `SpotCheckVerifier` | `dessin/consensus/spot_check_verification.py` |
| Reference trace trainer | `dessin/llm/decoder_lm_training_trace.py` |
| `QuantizationVerifier` (planned path) | `dessin/protocol/quantization_verifier.py` |
| `verify_block` (main consensus) | `dessin/consensus/consensus.py` |
| Config fields | `dessin/runtime/config.py` — `ConsensusConfig` |

---

## Related Documents

- [`SPOT_CHECK_VERIFICATION.md`](SPOT_CHECK_VERIFICATION.md) — full spec of the spot-replay path (mining contract, verifier steps, strike ledger, SGD witness design).
- [`POGO_PROTOCOL_SPECIFICATION.md`](POGO_PROTOCOL_SPECIFICATION.md) — complete protocol spec including block fields and two-phase window.
- [`E2E_NETWORK_TESTS.md`](E2E_NETWORK_TESTS.md) — the four-node E2E test that enables `enable_spot_check_verification = True`.
