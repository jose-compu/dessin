# Chapter 8 — Developer Quick-Start

## What this chapter covers

By the end of this chapter you will be able to:

1. Set up a local Python environment and run the full test suite.
2. Start a local single-node devnet.
3. Train a `DESSIN_NATIVE` model end-to-end with the built-in LoRA trainer.
4. Post a training task to the fee market and observe slot selection.
5. Upload a dataset and manage its storage lease.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | `python3 --version` |
| PyTorch | ≥ 2.0 | CPU-only is fine for tests |
| Git | any | |

No GPU is required to run the tests or the devnet. A GPU speeds up larger models but the built-in `DecoderOnlyGPT` nano-model trains in seconds on CPU.

---

## 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/dessin.git
cd dessin

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

---

## 2. Running the Tests

### All tests (unit + integration)

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected output: `632 passed` (or higher as features are added), zero failures.

### Specific test modules

```bash
# Training market (fee scheduling, urgency, VRF)
.venv/bin/python -m pytest tests/test_training_market.py -v

# LoRA fine-tuning (real gradient steps)
.venv/bin/python -m pytest tests/test_lora_trainer.py -v

# Model format registry and submission transactions
.venv/bin/python -m pytest tests/test_model_submission.py -v

# Storage lease lifecycle
.venv/bin/python -m pytest tests/test_storage_lease.py -v

# Full PoGO protocol (block production + verification)
.venv/bin/python -m pytest tests/test_full_pogo_protocol.py -v
```

### E2E network tests

The E2E tests spin up real multi-node networks in-process:

```bash
.venv/bin/python -m pytest e2e/ -v
```

> These take longer (30–120 seconds) because they simulate actual block production cycles. They do not require any external services.

---

## 3. Training a Model Locally

The simplest end-to-end training path uses the built-in `DecoderOnlyGPT` and the bundled Shakespeare corpus.

### 3a. Full PoGO training (baseline)

```python
from dessin.llm.decoder_only_gpt import DecoderOnlyGPT
from dessin.llm.simple_trainer import train_model
from dessin.llm.decoder_lm_training import BUNDLED_SHAKESPEARE_CHAR_CORPUS

model = DecoderOnlyGPT(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64)

result = train_model(
    model=model,
    training_data=BUNDLED_SHAKESPEARE_CHAR_CORPUS,
    steps=200,
    batch_size=4,
    learning_rate=1e-3,
)

print(f"Loss: {result.loss_before:.3f} → {result.loss_after:.3f}")
print(f"Weights hex (first 40 chars): {result.model_weights_hex[:40]}...")
```

### 3b. LoRA fine-tuning (parameter-efficient)

```python
from dessin.llm.lora_trainer import LoRAConfig, train_lora
from dessin.llm.decoder_only_gpt import DecoderOnlyGPT
from dessin.llm.decoder_lm_training import BUNDLED_SHAKESPEARE_CHAR_CORPUS

model = DecoderOnlyGPT(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64)

cfg = LoRAConfig(
    rank=8,
    alpha=16.0,
    target_modules={"qkv", "proj"},   # which Linear layers get adapters
)

result = train_lora(
    model=model,
    training_data=BUNDLED_SHAKESPEARE_CHAR_CORPUS,
    config=cfg,
    steps=300,
    batch_size=4,
    learning_rate=3e-4,
)

print(f"Loss: {result.initial_loss:.3f} → {result.final_loss:.3f}")
print(f"Trainable params: {result.trainable_params:,} / {result.total_params:,} "
      f"({result.param_efficiency:.1%})")

# The merged weights are PoGO-compatible:
print(result.training_result.model_weights_hex[:40])
```

The `LoRATrainingResult.training_result` field is a standard `TrainingResult` — the same object produced by full PoGO training — so it slots directly into the block production pipeline.

---

## 4. Using the Training Market

### 4a. Submit a task

```python
from dessin.consensus.training_market import TrainingMarket, TrainingTaskSpec
from dessin.runtime.config import TrainingMarketConfig

market = TrainingMarket(TrainingMarketConfig())

task = TrainingTaskSpec(
    task_id="dev_task_001",
    model_id="my_nano_gpt",
    owner="0xDev",
    deposit=0.12,          # base_fee(0.01) + tip(0.11)
    base_fee=0.01,
    tip=0.11,
    cooldown_blocks=5,
    max_slots=1,
    submitted_block=1,
    expiry_block=100,
)

ok, msg = market.submit_task(task, current_block=1)
print(ok, msg)   # True, "ok"
```

### 4b. Select tasks for a block

```python
import hashlib

vrf_seed = hashlib.sha256(b"miner_secret_block1").digest()
slots = market.select_tasks_for_block(current_block=1, vrf_seed=vrf_seed)

for slot in slots:
    print(f"  → task {slot.task_id}  tip_paid={slot.tip_paid:.4f}  base_fee_burned={slot.base_fee_burned:.4f}")
```

### 4c. Finalize the block and watch the base fee adjust

```python
market.finalize_block(block_number=1, scheduled_slots=slots)
print(f"New base fee: {market.base_fee:.6f}")   # rises if block was full
```

---

## 5. Registering a Model

```python
import json
from dessin.models.model_format_registry import (
    dessin_native_preset, ModelFamily,
    TrainingPipelineSpec, PipelineStageSpec, ModelSubmissionSpec
)
from dessin.models.model_submission_transactions import RegisterModelTransaction
from dessin.economics.compute_fee_schedule import ComputeFeeSchedule

# Build format spec
fmt = dessin_native_preset(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=65)
assert fmt.is_trainable  # always True for DESSIN_NATIVE

# Build pipeline
pipeline = TrainingPipelineSpec(name="lora-quickstart")
pipeline.add_stage(PipelineStageSpec(
    stage_index=0,
    method="lora",
    dataset_id="my_dataset_sha256",
    max_steps=500,
    lora_rank=8,
    lora_alpha=16,
))

# Build submission spec
spec = ModelSubmissionSpec(
    model_name="QuickstartNanoGPT",
    owner="0xDev",
    family=ModelFamily.NANO_GPT,
    format_spec=fmt,
    pipeline=pipeline,
)

errors = spec.validate()
assert errors == [], errors

# Estimate fees
sched = ComputeFeeSchedule()
deposit = sched.estimate_pipeline_fee(
    size_mb=fmt.size_mb,
    stages=[s.to_dict() for s in pipeline.stages],
    is_quantized=fmt.is_quantized,
)
print(f"Estimated deposit: {deposit:.6f} DESSIN")

# Build transaction (fill in real sender/sig fields for on-chain use)
tx = RegisterModelTransaction(
    sender="0xDev", fee=0.001, timestamp=0,
    public_key="", signature="", tx_id="reg_001",
    model_name=spec.model_name,
    family=spec.family.value,
    format=spec.format_spec.format.value,
    size_mb=spec.format_spec.size_mb,
    parameter_count_millions=spec.format_spec.parameter_count_millions,
    content_hash=spec.content_hash(),
    submission_spec_json=json.dumps(spec.to_dict()),
    required_deposit=deposit,
    storage_blocks=100,
    storage_payment=0.001,
)
print(f"Content hash: {tx.content_hash}")
```

---

## 6. Uploading Training Data

```python
from dessin.models.model_submission_transactions import UploadTrainingDataTransaction
from dessin.economics.compute_fee_schedule import ComputeFeeSchedule
from dessin.economics.storage_lease import StorageLeaseManager, StorageLeaseConfig

sched = ComputeFeeSchedule()
size_mb = 10.0

upload_fee    = sched.estimate_upload_fee(size_mb)
storage_fee   = sched.estimate_storage_renewal_fee(size_mb, blocks=200)

print(f"Upload fee:  {upload_fee:.4f} DESSIN")
print(f"Storage fee: {storage_fee:.4f} DESSIN (200 blocks)")

tx = UploadTrainingDataTransaction(
    sender="0xDev", fee=0.001, timestamp=0,
    public_key="", signature="", tx_id="upload_001",
    file_id="sha256_of_my_dataset",
    model_id="QuickstartNanoGPT",
    file_name="corpus.txt",
    file_hash="sha256_of_my_dataset",
    size_mb=size_mb,
    dataset_type="pretrain",
    storage_blocks=200,
    storage_payment=storage_fee,
)

# Simulate creating the lease
mgr = StorageLeaseManager(StorageLeaseConfig())
lease = mgr.create_lease(
    file_id=tx.file_id,
    owner=tx.sender,
    size_mb=tx.size_mb,
    current_block=1,
    storage_blocks=tx.storage_blocks,
    payment=tx.storage_payment,
)
print(f"Lease status: {lease.status.value}")       # active
print(f"Paid through: block {lease.paid_through_block}")  # 201
```

---

## 7. Module Map

```
dessin/
├── consensus/
│   ├── training_market.py   ← fee market, mempool, slot selection
│   └── transactions.py      ← all transaction types
├── llm/
│   ├── decoder_only_gpt.py  ← DecoderOnlyGPT architecture
│   ├── simple_trainer.py    ← full PoGO training loop
│   └── lora_trainer.py      ← LoRA fine-tuning (pure PyTorch)
├── models/
│   ├── model_format_registry.py      ← ModelFormat, ModelFormatSpec, presets
│   └── model_submission_transactions.py  ← RegisterModel, UploadData, RenewLease
├── economics/
│   ├── compute_fee_schedule.py  ← fee heuristics
│   └── storage_lease.py         ← StorageLeaseManager
└── runtime/
    ├── config.py    ← DessinConfig, TrainingMarketConfig
    └── node.py      ← full node orchestration
```

---

## 8. Environment Variables

The node reads configuration from environment variables. Useful for testing:

| Variable | Default | Effect |
|---|---|---|
| `DESSIN_BLOCK_TIME` | `7200` | Block time in seconds |
| `DESSIN_MARKET_MAX_TASKS_PER_BLOCK` | `4` | Slot budget per block |
| `DESSIN_MARKET_INITIAL_BASE_FEE` | `0.01` | Starting base fee |
| `DESSIN_MARKET_MIN_TIP` | `0.0` | Minimum tip required |
| `DESSIN_MARKET_URGENCY_RATE` | `0.10` | Urgency growth per block |
| `DESSIN_MARKET_URGENCY_CAP` | `5.0` | Maximum urgency multiplier |

Example:

```bash
DESSIN_MARKET_MAX_TASKS_PER_BLOCK=8 \
DESSIN_BLOCK_TIME=15 \
.venv/bin/python -m pytest tests/ -v
```

---

## Checkpoint

After this chapter you should be able to:
- Set up the dev environment and run the full test suite (632 tests, all green).
- Train a `DESSIN_NATIVE` model with full PoGO or LoRA.
- Submit tasks to the training market and observe scheduling.
- Register a model and upload training data with a storage lease.

---

*You have now completed the guided tour. The codebase lives under `dessin/` — consult the module map above or the source docstrings for deeper details.*
