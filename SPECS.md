# DeSSIN chain — operations & economics

This document describes how the DeSSIN node and token layer behave from a **Node Operator** (runs infrastructure, mining, pricing) and an **end User** (model owner, query consumer, token holder) perspective. **Tokenomics, balances, and fees** are the focus; implementation details live in `dessin/config.py`, `dessin/economic_system.py`, `dessin/enhanced_pogo_consensus.py`, and `dessin/transactions.py`.

---

## 1. Roles

### Node Operator

- Runs a **DeSSIN node** (`DessinNode`): Chaincraft P2P, PoGO / enhanced consensus, optional mining loop when `block_time_minutes > 0`.
- Controls **local configuration** (ports, peers, consensus timings, model cache paths).
- May tune **dynamic economics** via `DynamicParameterManager` / node operator API (e.g. storage price, block reward targets within safety bounds).
- **Mining**: proposes blocks when eligible; receives **block rewards** and **training payments** when economics finalize.

### User (application layer)

- **Model owner**: uploads/registers models; pays **training**, **storage**, and possibly **escrow** so miners train against registered workloads.
- **End user / querier**: pays **query/compute** pricing when inference or queries are billed against balances.
- **Token holder**: holds **liquid** balance, **staked** balance, or (during bootstrap) **virtual** stake for ranking.

---

## 2. Token & supply (tokenomics)

| Concept | Default / behavior |
|--------|---------------------|
| **Symbol** | `DESSIN` (see `TokenConfig.token_symbol`) |
| **Decimals** | 18 (for display and integration; internal economics use `float` for amounts in many paths) |
| **Initial supply** | Carried in a **genesis** account; **block rewards and materialization** mint from genesis up to available balance; `total_tokens_emitted` tracks emissions. |
| **Base mining reward** | `TokenConfig.mining_reward` (e.g. 10 DESSIN per block when not using annualized emission) |
| **Block reward formula** | `EconomicSystem.distribute_block_reward`: `base_reward` (emission- or fixed-based) **+** `training_steps * 0.01` (training bonus). |
| **Annualized emission (optional)** | If `bootstrap_period_blocks > 0`, **bootstrap** and **post-bootstrap** use `annual_emission_rate_bootstrap` / `annual_emission_rate_post_bootstrap` with `avg_block_time_seconds` / `blocks_per_year` to size per-block mint. |
| **Attestation rewards** | Verifiers can receive `EconomicEventType.ATTESTATION_REWARD` when attestation distribution runs. |

### Where new tokens go

- **Non-bootstrap** (`bootstrap_period_blocks == 0`): block rewards are minted to the miner’s **liquid** balance by default (`auto_stake_reward=False` in that mode).
- **Bootstrap enabled** (`bootstrap_period_blocks > 0`): while the chain is in the bootstrap window and after, **block rewards are auto-staked** (`auto_stake_reward=True`): rewards accrue in `staked_amounts[miner]` unless/until unstaked (when implemented) or moved by protocol rules.

---

## 3. Account balances (three buckets)

The economic layer tracks, **per address**:

1. **Liquid** — `balances[address]` — transferable (subject to tx success); receives block rewards when not auto-staking; pays fees and owner-side costs.
2. **Staked** — `staked_amounts[address]` — non-liquid stake used for security / ranking mechanics (bootstrap reward stake, post-bootstrap auto-stake rewards, **materialized** bootstrap proceeds).
3. **Bootstrap virtual** — `bootstrap_virtual_locked[address]` — **non-liquid, non-transferable ranking weight** during bootstrap only; cleared at slash or **materialization**.

**Holdings** shown in operator logs are typically:

`holdings = liquid + staked + virtual`

**Protocol rewards (audit sum)** — `total_protocol_rewards_received(address)` sums economic events of types **block reward**, **attestation reward**, and **bootstrap materialization** credited `to_address == address`.

---

## 4. Bootstrap lifecycle (when `bootstrap_period_blocks > 0`)

| Phase | Behavior |
|-------|----------|
| **Registration** | Eligible miners call into `register_bootstrap_participant`; receive **virtual stake** (`bootstrap_virtual_stake`) for ranking. |
| **Mining during bootstrap** | Block rewards **auto-stake**; virtual + staked rewards define **effective bootstrap stake** for active-set ranking (`effective_bootstrap_stake`). |
| **Ranking / slashing** | Inactive or out-of-top-N participants may be **slashed** (virtual removed; reward stake cleared per `slash_bootstrap_participant`). |
| **End of bootstrap** | At the cap block, **survivors** are **materialized**: virtual claim **+** materialization grant **+** carried reward stake are minted from genesis into **`staked_amounts`** (not liquid). Event type: `BOOTSTRAP_MATERIALIZATION`. |
| **Post-bootstrap mining** | Eligibility can require minimum **holdings fraction** of emitted supply (`post_bootstrap_min_holdings_fraction`); block rewards continue **auto-staked** while bootstrap mode is configured. |

If **`bootstrap_period_blocks == 0`** (default in many dev setups): no virtual stake, no materialization; rewards follow the non-bootstrap path (typically **liquid**).

---

## 5. Fees & payments

### Transaction fees

- Transactions carry a **`fee`** field (DESSIN). Processing paths call `EconomicSystem.collect_transaction_fee` → transfer from payer to **`system`** (`TRANSACTION_FEE`).

### Training payments

- Per-step price: `training_payment_per_step` (economic system); scaled by a **model complexity factor** (size-dependent).
- Flow: **`process_training_payment`** pays the **miner** from **per-model escrow** first (`model_training_escrow[model_id]`), then from the **model owner’s liquid balance** for the remainder.

### Training escrow

- Owners deposit with `deposit_model_training_escrow` — moves liquid DESSIN into **`escrow:<model_id>`** accounting (`TRAINING_ESCROW_DEPOSIT`).

### Storage & queries

- **Storage**: priced per GB per block (`base_storage_price` × multipliers); tracked via storage transactions and economic events (`STORAGE_*`).
- **Queries**: `query_fee_per_token`–style pricing on token usage for inference/query paths.

### Owner refresh / training-data updates

- Transactions such as **`ModelTrainingDataRefreshTransaction`** charge a **minimum fee** (`training_refresh_min_fee`) and may require **escrow** when a model was frozen (`training_refresh_min_escrow_when_frozen`). Used to register new training data and **unfreeze** training after strikes.

---

## 6. Operator-visible tuning (environment)

Key variables (non-exhaustive; see `DessinConfig.from_env`):

- **Mining cadence**: `DESSIN_BLOCK_TIME_MINUTES` (legacy: `DESSIN_BLOCK_TIME_HOURS` × 60).
- **Training cadence / dynamic timing**: `DESSIN_TRAINING_BLOCK_TIME_MINUTES`, min/max block seconds, ramp percentages.
- **Bootstrap**: `DESSIN_BOOTSTRAP_PERIOD_BLOCKS`, `DESSIN_BOOTSTRAP_VIRTUAL_STAKE`, `DESSIN_BOOTSTRAP_MATERIALIZATION_GRANT`, etc.
- **Merkle / verification**: `DESSIN_MERKLE_PROOF_COUNT`, phase windows, sample attempts.

---

## 7. User journey (short)

1. **Fund liquid DESSIN** (genesis-funded demos or transfers).
2. **Upload / register** a model; pay **storage** and optional **training escrow**.
3. Miners train on blocks; **owner** pays **training cost** to miners (escrow + balance); **miner** earns **block reward** (+ bonus by steps) and attestation rewards where applicable.
4. **Queries** consume balance according to operator/query pricing.
5. Under **bootstrap**, rank and rewards differ until **materialization** moves survivor stake into **staked** balances.

---

## 8. E2E tests

- Live four-node battery: `e2e/tests/test_four_node_live_network.py`.
- Run with **stdout**: `pytest -svv e2e/tests`, optional env shorten (`DESSIN_E2E_DURATION_SEC`, …).
- Use a Python interpreter that has **chaincraft==0.5.1** (PyPI) and ML deps installed, or the repo’s **`lib/pythonX.Y/site-packages`** vendored path (see `e2e/conftest.py` and `docs/E2E_NETWORK_TESTS.md`).
- Skip all: `DESSIN_SKIP_E2E=1`. Optional heavy GPT-2-class test: `DESSIN_E2E_INCLUDE_GPT2_NANO=1`.
- **Spot-check pipeline** (no full retraining): `e2e/tests/test_spot_check_pipeline.py` covers honest miner approval, two cheating-miner rejection scenarios (corrupted loss curve / faked gradient on a VRF-challenged step), and the model-level strike system. Walkthrough script: `e2e/scripts/spot_check_demo.py`. Design and config knobs: `docs/SPOT_CHECK_VERIFICATION.md`.

---

## 9. Verification paths

Two coexisting verification approaches are tracked in code and docs; both avoid full retraining on validators:

| Path | Where | Purpose |
| --- | --- | --- |
| **Quantization consistency** (`enable_efficient_verification`, default **on**) | `dessin/quantization_verifier.py`, `dessin/consensus.py::verify_block` | Validators check that disclosed full-precision weights re-quantize to the claimed quantized artifact (plus Merkle leaf challenges). See [`docs/EFFICIENT_VERIFICATION.md`](docs/EFFICIENT_VERIFICATION.md). |
| **Shared-VRF spot-check** (`enable_spot_check_verification`, default **off**, opt-in) | `dessin/spot_check_verification.py`, `dessin/decoder_lm_training_trace.py` | Miners publish `state_0_hash` / `state_N_hash` anchors, full loss curve, gradient hash chain. Verifiers replay only `spot_check_count` (default **2**) step indices drawn from a **post-publication** VRF seed; ≥`spot_check_attestation_threshold` (default **0.66**) of stake must replay-and-agree. Model-level **strike** system freezes a model after `spot_check_max_strikes` (default **3**) consecutive blocks whose loss never dips below `L_0`; refresh via `ModelTrainingDataRefreshTransaction` clears strikes. See [`docs/SPOT_CHECK_VERIFICATION.md`](docs/SPOT_CHECK_VERIFICATION.md). |

The spot-check path is wired through a callback API today (`replay_fn`) and gated by config; block-schema fields for the trace anchors will land in a follow-up so the four-node battery can opt into it. Tests in `tests/test_spot_check_verification.py`, `tests/test_decoder_lm_training_trace.py`, and `e2e/tests/test_spot_check_pipeline.py`.

---

## 10. Disclaimer

Parameters and formulas evolve with the codebase; verify against **`ConsensusConfig`**, **`TokenConfig`**, and **`EconomicSystem`** for your checkout. This document reflects the intended economic split between **liquid**, **staked**, **virtual bootstrap**, **fees**, and **materialization to stake**.
