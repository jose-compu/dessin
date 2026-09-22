# Four-node live E2E battery

Runs four real processes in-process (three miners, one non-mining node), meshes Chaincraft UDP peers, mines briefly, then asserts tip agreement and chain growth.

The battery configures **1-minute** mining/training slots, **block-based** two-phase completion (`verification_phase_completion_in_blocks`), and modest phase windows (defaults below)—verification advances by **block height**, not multi-minute wall clocks tied to production defaults.

**Tests:** `test_four_node_live_consensus_battery_micro_llm` always runs (unless skipped): **micro GPT char** (`DESSIN_LLM_VARIANT=micro_gpt_char`). Optional heavy run: `test_four_node_live_consensus_battery_gpt2_nano` — set **`DESSIN_E2E_INCLUDE_GPT2_NANO=1`** to exercise the **GPT-2-class** path (`gpt2_nano` via `DESSIN_LLM_VARIANT`; same Shakespeare corpus until BPE lands).

**Training:** The four-node pytest module **forces** decoder LM training (`DESSIN_MICRO_GPT_TRAINING=1` at import) and registers the configured LM model id (default `micro_gpt_char`) on **every** node before mining, so all miners share the same catalog and cannot auto-bootstrap a divergent local model id. Logs like `Trained weight payload: 855040 bytes variant=micro_gpt_char` are expected: that byte count is the **float64-flattened micro-GPT weight tensor** for the default tiny architecture. The legacy `simple_neural_network` MLP catalog has been removed; `node._create_dummy_model()` now always produces a decoder-LM catalog entry resolved from `DESSIN_LLM_VARIANT`.

**Block trace:** With stdout visible (`pytest -s`) and `DESSIN_E2E_QUIET` unset, each new block prints `[e2e] BLOCK idx=… leader=node_K … loss_before/loss_after …` using the longest local chain as the canonical view (`chain_view=node_i`).

**Timing:** Each `[e2e]` line appends `⏱ elapsed=…s` and `⌛ remaining≈…s` (rough wall-clock budget: warmup + soak + converge + fixed slack). Soak iterations also log `soak_phase_remaining` for the sleep window.

**Colors / emoji:** When stdout is a TTY, lines use ANSI colors and category emojis (blocks 🧱, mesh 🔗, soak 🌊, etc.; same helpers as core `dessin.pretty_console`). Set `NO_COLOR=1`, `DESSIN_NO_FANCY=1`, or `DESSIN_E2E_NO_FANCY=1` for plain text. Use `FORCE_COLOR=1` under captured runners if you still want colors.

## Run (verbose console output)

**Interpreter:** Run pytest with a Python version that has **chaincraft**, **torch**, **transformers**, and related deps available. Install **`chaincraft==0.5.1`** from PyPI (`pip install chaincraft==0.5.1`) or via **`requirements.txt`**. The repo may ship vendored wheels under `lib/pythonX.Y/site-packages`; **`e2e/conftest.py`** prepends that directory for the active interpreter (e.g. use **`python3.13`** if `lib/python3.13/site-packages` exists).

Pytest hides stdout unless you disable capture; use `-s` (or `--capture=no`) plus `-vv` for detail:

Quick smoke (few blocks; overrides defaults **30** new blocks / **420** s soak):

```bash
cd /Users/joseignacio/Documents/GitHub/dessin && \
DESSIN_E2E_DURATION_SEC=35 \
DESSIN_E2E_WARMUP_SEC=18 \
DESSIN_E2E_MIN_NEW_BLOCKS=1 \
DESSIN_E2E_MIN_PEERS=2 \
DESSIN_E2E_CONVERGE_TIMEOUT_SEC=120 \
./bin/python -m pytest e2e/tests/test_four_node_live_network.py \
  -svv --tb=short --color=yes \
  --log-cli-level=INFO 2>&1 | tee /tmp/dessin_e2e_pytest.log
```

Full default battery (no env): expects **≥30** new blocks after warmup; allow enough wall time for your machine.

Inspect the full log without truncation (`tail` drops earlier failures):

```bash
tail -200 /tmp/dessin_e2e_pytest.log
```

Quiet harness banners only (still shows node/mining prints from the library):

```bash
DESSIN_E2E_QUIET=1 ./bin/python -m pytest e2e/tests/test_four_node_live_network.py -svv --tb=short
```

Skip the battery:

```bash
DESSIN_SKIP_E2E=1 ./bin/python -m pytest e2e/tests/
```

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `DESSIN_E2E_DURATION_SEC` | `1500` | Soak after mesh (~25 min); sized for default 20×1-min slots with headroom |
| `DESSIN_E2E_WARMUP_SEC` | `25` | Budget for peer mesh stabilization |
| `DESSIN_E2E_MIN_NEW_BLOCKS` | `20` | Requested min new blocks vs baseline (battery uses `bootstrap_period_blocks=10`: ≈10 in-window + ≈10 post-bootstrap); **capped** by `DURATION_SEC` × slot cadence |
| `DESSIN_E2E_MIN_PEERS` | `2` | Every node must report at least this many outbound peers |
| `DESSIN_E2E_CONVERGE_TIMEOUT_SEC` | `120` | Wait for matching tips across nodes |
| `DESSIN_E2E_MAX_HEIGHT_SPREAD` | `2` | Max allowed height difference between nodes |
| `DESSIN_E2E_QUIET` | unset | Set to `1` to reduce `[e2e]` progress lines |
| `DESSIN_E2E_MESH_SETTLE_SEC` | `1.75` | Extra delay (seconds) after all nodes start before peer mesh attempts |
| `DESSIN_DEBUG_PEER_BANS` | unset | Set to `1` to log Chaincraft invalid-message strikes (3 → ban) and peer bans |
| `DESSIN_E2E_INCLUDE_GPT2_NANO` | unset | Set to `1`/`true` to collect and run the optional **gpt2_nano** four-node battery (slower; separate test function) |
| `DESSIN_E2E_PHASE1_BLOCKS` | `1` | PoGO phase 1 span in blocks (quantized verification window after training block) |
| `DESSIN_E2E_PHASE2_BLOCKS` | `1` | Phase 2 span in blocks (leaf challenges) |
| `DESSIN_E2E_FINALIZATION_WINDOW_BLOCKS` | `3` | Blocks until finalization phase boundary (matches `ConsensusConfig.finalization_window`) |
| `DESSIN_E2E_BOOTSTRAP_PERIOD_BLOCKS` | `10` | PoGO bootstrap window length (`ConsensusConfig.bootstrap_period_blocks` in the four-node battery) |
| `DESSIN_E2E_INITIAL_LIQUID_BONUS` | `1000` | Liquid DESSIN moved from genesis to **each** node address on **every** replica before `start()` (avoids early affordability skips while rewards are still auto-staked) |
| `DESSIN_E2E_SKIP_INFERENCE` | unset | Set to `1` to skip post-battery greedy Shakespeare continuation (uses latest mined JSON under the test temp dir) |
| `DESSIN_E2E_INFER_MAX_NEW` | `96` | Max greedy char tokens appended after the fixed prompt |

### Micro-GPT (mining / PoGO blocks)

Set `DESSIN_MICRO_GPT_TRAINING=1` for char-LM blocks (default for this pytest module via `setdefault`). Override sizes like a nanoGPT smoke run:

| Variable | Default | Meaning |
|----------|---------|---------|
| `DESSIN_MICRO_GPT_TRAINING` | `1` in E2E test | `1`/`true`: Transformer LM; `0`: MLP |
| `DESSIN_LLM_VARIANT` | `micro_gpt_char` | `gpt2_nano` for GPT-2-class (nanoGPT-scale defaults); aliases `gpt2`, `nanogpt` |
| `DESSIN_MICRO_GPT_STEPS` | consensus/env | Optimizer steps per mined block |
| `DESSIN_MICRO_GPT_BATCH` | consensus/env | Batch size |
| `DESSIN_MICRO_GPT_LR` | consensus/env | AdamW learning rate |
| `DESSIN_MICRO_GPT_DEVICE` | `cpu` | `cpu` or `cuda` |
| `DESSIN_MICRO_GPT_N_LAYER` | `2` | Transformer layers |
| `DESSIN_MICRO_GPT_N_HEAD` | `2` | Attention heads |
| `DESSIN_MICRO_GPT_N_EMBD` | `64` | Embedding width |
| `DESSIN_MICRO_GPT_BLOCK_SIZE` | `32` | Context length |

When `DESSIN_LLM_VARIANT=gpt2_nano`, architecture metadata uses variant `gpt2_nano` (see `dessin.llm_model_spec`). Override sizes with `DESSIN_GPT2_N_LAYER`, `DESSIN_GPT2_N_HEAD`, `DESSIN_GPT2_N_EMBD`, `DESSIN_GPT2_BLOCK_SIZE` (defaults: 6 / 6 / 192 / 128), and device with `DESSIN_GPT2_DEVICE` (falls back to `DESSIN_MICRO_GPT_DEVICE` in the torrent trainer log path).

External references: [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat) — this harness is an in-repo minimal analogue for CI/E2E.

## Timeout

The test is marked `timeout(1500)` when `pytest-timeout` is installed (`pip install -e ".[dev]"`).

## Spot-check verification pipeline

The shared-VRF spot-check verification path (no full retraining; verifiers replay
only `k` step indices drawn from a post-publication seed) ships an isolated e2e
test independent of the four-node mesh:

```bash
./bin/python -m pytest e2e/tests/test_spot_check_pipeline.py -svv --tb=short
```

It exercises one honest miner block (≥ 0.66 approval ratio), two cheating-miner
scenarios (corrupted loss curve → statistical reject; faked gradient on a
challenged step → replay reject), and the model-level strike system (3
unproductive blocks → frozen, requires the existing
`ModelTrainingDataRefreshTransaction` to unfreeze).

A walkthrough script with verbose per-verifier prints lives under
`e2e/scripts/spot_check_demo.py`:

```bash
./bin/python e2e/scripts/spot_check_demo.py
```

Design and config knobs are documented in
[`docs/SPOT_CHECK_VERIFICATION.md`](SPOT_CHECK_VERIFICATION.md). The path is
gated by `enable_spot_check_verification` (off by default) until block-schema
fields land in `PogoBlock`; the four-node battery above continues to use the
existing PoGO consensus path.
