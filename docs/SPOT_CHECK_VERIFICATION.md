# Shared-VRF spot-check verification

This document describes the lightweight "no full retraining" verification path
implemented in `dessin/consensus/spot_check_verification.py` and the reference torch trainer
`dessin/llm/decoder_lm_training_trace.py`.

## Mining contract per training block

Each block runs **exactly `spot_check_steps_per_block` gradient steps** (default
**100**) on a deterministic batch schedule derived from `data_seed` (which is
itself anchored to the block's VRF output). The miner publishes:

- `state_0_hash`: SHA-256 of the float64-flattened weights at the **start** of
  the block (must equal the previous block's `state_N_hash` — anchor check).
- `state_N_hash`: SHA-256 of the float64-flattened weights at the **end** of the
  block (must equal the published model artifact).
- `loss_curve`: length `num_steps + 1`; the per-minibatch loss measured at each
  step plus the final eval loss.
- `gradient_hash_chain`: rolling SHA-256 chain over per-step gradients.
- `gradient_norms`: per-step L2 norm of the (flattened, float64) gradient.
- `data_seed`, `learning_rate`, `batch_size`, `num_steps`.

Per-step pre-states and gradients are **kept locally** by the miner (the
`TrainingTracePrivate` witness) and revealed only when challenged.

## Verification (every full node, no watchtowers)

1. **Structural** (free, mandatory): step counts, lengths, `state_0_hash` anchor
   to the previous block, `state_N_hash` matches the published model. See
   `verify_structural`.
2. **Statistical** heuristics on the published curves (cheap): finite values,
   gradient-norm band, max per-step loss-jump ratio, grad-vs-Δloss correlation
   floor. See `verify_statistical` + `StatisticalThresholds`. These flag obvious
   forgeries (e.g. uniform-random gradient norms, impossibly clean loss
   trajectories) but are heuristics — they raise the cost of cheating, they do
   not prove honest training on their own.
3. **Spot replay** of `spot_check_count` step indices (default **2**) drawn from
   `derive_challenge_step_indices(seed, num_steps, k)`. The seed **must come
   from chain state published *after* this training block** (e.g. the next
   block's hash, mixed with a randomness beacon if available) so the miner
   cannot grind for predictable challenge step indices. Each verifier replays
   **the same** chosen steps.
4. **Attestation**: a verifier votes "approved" if (1) ∧ (2) ∧ (3) all pass.
   The block is accepted when **`spot_check_attestation_threshold`** of the
   stake (default **0.66**) approves.

## Model-level strike ledger

Per accepted block, `accumulate_strikes(state, model_id, trace, max_strikes)`:

- If any `loss_curve[i>0] < loss_curve[0]` ⇒ strike count resets to 0.
- Else strike count `+= 1`. On reaching `max_strikes` (default **3**) the model
  is added to the frozen set; new training blocks against it are rejected until
  the owner submits the existing `ModelTrainingDataRefreshTransaction` (paid
  refresh) which calls `state.reset(model_id)`.

The strike system enforces *usefulness at the model granularity* without
requiring per-block monotonic loss decrease at the consensus layer.

## Why two challenges are typically enough

Without knowing `(j, k)` at mining time, a miner who only honestly computes
`m` of `N` steps passes the spot replay with probability `(m/N)^k`. For
`N = 100, m = 2, k = 2` ≈ **0.04%** (≈99.96% catch rate from spot replay
alone). Statistical heuristics add an independent layer: even an adversary
who guesses `(j, k)` must still produce a globally consistent loss curve,
gradient-norm trajectory, and chain hashes across all 100 steps. Joint
escape probability collapses to negligible.

## Optimizer assumption (v1: SGD only)

The trace + replay path in this version assumes **plain SGD**:

```text
θ_j = θ_{j-1} - lr · g_j
```

A single-step replay needs only `(θ_{j-1}, data_seed, batch_size, lr)`. No
per-parameter optimizer state goes into the witness.

### Why SGD now

- Witness stays "weights + gradient" only (small).
- Determinism is straightforward (CPU/float64 already pin replay).
- Replay function is ~20 lines.

### Caveats: switching to AdamW (or other stateful optimizers) later

AdamW's update at step `j` depends on per-parameter running moments and the
step counter:

```text
m_j = β1·m_{j-1} + (1-β1)·g_j
v_j = β2·v_{j-1} + (1-β2)·g_j²
θ_j = θ_{j-1} - lr · (m̂_j / (√v̂_j + ε))   (with bias correction in m̂, v̂)
```

To preserve spot-check soundness with AdamW we would need to:

1. **Extend `TrainingTracePrivate`** with `optimizer_state_hex_before[j]`
   (containing flattened `m_{j-1}, v_{j-1}` and the step counter `t`) plus a
   hash chain so an adversary cannot forge `θ_j` by tweaking moments.
2. **Extend `make_replay_fn`** to instantiate an AdamW optimizer, load the
   committed `(m, v, t)` for step `j`, run one step, and check that the
   resulting `(θ_j, m_j, v_j)` reconstruct the next chain entry / state hash.
3. **Pin determinism**: AdamW under different PyTorch / CUDA / BLAS versions
   can differ at the last few bits. Either enforce **CPU + float64** for the
   replay path, or declare a documented numerical tolerance and check
   distances rather than exact equality.
4. **Trace-size impact**: per-step witness becomes ≈ 3× larger (weights, `m`,
   `v`). For the default tiny micro-GPT this is still kilobytes per step; for
   a real GPT-2-class model it is meaningful and may push us to commit only
   hashes of `(m, v)` per step on chain and stream the buffers off-chain on
   challenge.

Until these land, do **not** silently swap the optimizer in
`dessin/llm/decoder_lm_training_trace.py` — the witness schema is sized for SGD.

## Config knobs (`ConsensusConfig`)

| Field | Env override | Default |
| --- | --- | --- |
| `enable_spot_check_verification` | `DESSIN_ENABLE_SPOT_CHECK_VERIFICATION` | `False` |
| `spot_check_steps_per_block` | `DESSIN_SPOT_CHECK_STEPS_PER_BLOCK` | `100` |
| `spot_check_count` | `DESSIN_SPOT_CHECK_COUNT` | `2` |
| `spot_check_attestation_threshold` | `DESSIN_SPOT_CHECK_ATTESTATION_THRESHOLD` | `0.66` |
| `spot_check_max_strikes` | `DESSIN_SPOT_CHECK_MAX_STRIKES` | `3` |
| `spot_check_min_grad_norm` | (none) | `1e-9` |
| `spot_check_max_grad_norm` | (none) | `1e6` |
| `spot_check_max_loss_jump_ratio` | (none) | `50.0` |
| `spot_check_min_grad_loss_corr` | (none) | `-0.9` |

The gate (`enable_spot_check_verification`) is off by default; the trace path
is opt-in until block-schema fields for `state_0_hash` / trace URI are wired
into `PogoBlock`. Tests cover the verifier surface end-to-end via the callback
API today.

## Tests and demo

- Unit: `tests/test_spot_check_verification.py`,
  `tests/test_decoder_lm_training_trace.py`.
- E2E (no four-node mesh): `e2e/tests/test_spot_check_pipeline.py` —
  honest miner approval, two cheating-miner rejection scenarios (corrupted loss
  curve / faked gradient on a VRF-challenged step), and the strike → freeze →
  refresh flow.
- Walkthrough script: `e2e/scripts/spot_check_demo.py`.

## Related documents

- [`POGO_PROTOCOL_SPECIFICATION.md`](POGO_PROTOCOL_SPECIFICATION.md): consensus,
  blocks, attestations, quantization, two-phase verification windows.
- [`EFFICIENT_VERIFICATION.md`](EFFICIENT_VERIFICATION.md): the on-by-default
  quantization-consistency fast path that this document extends.
- [`E2E_NETWORK_TESTS.md`](E2E_NETWORK_TESTS.md): broader E2E test battery
  layout and environment knobs.
