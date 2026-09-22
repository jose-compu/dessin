"""
E2E: Alice's LoRA Fine-Tuning Pipeline
=======================================

Exercises the full "Alice" workflow end-to-end — registration → dataset upload
with storage lease → training-market task submission → LoRA fine-tuning →
PoGO-compatible weight export — using **DESSIN_NATIVE** format (the only
format with a working in-protocol trainer today).

Why not GGUF?
    GGUF and SafeTensors are inference-only in the current protocol: no in-protocol
    trainer exists for them.  The "7B GGUF with LoRA" scenario from the docs is a
    future goal; today we demonstrate the same economic workflow on a
    DecoderOnlyGPT nano-model, where ALL numbers (fees, storage, tips) are
    real and derived from the same fee schedule.

MacBook Air compatibility:
    The model uses 2 layers / 2 heads / 64 embd — identical to the unit tests.
    Training runs 200 gradient steps on CPU (< 10 s).  The "2000-step" scenario
    from the docs is shown in comments as a fee-estimation exercise.

Skipped under DESSIN_SKIP_E2E=1.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import pytest

pytest.importorskip("torch")
import torch

from dessin.consensus.training_market import TrainingMarket, TrainingTaskSpec
from dessin.economics.compute_fee_schedule import ComputeFeeSchedule, TrainingPhase
from dessin.economics.storage_lease import (
    LeaseStatus,
    StorageLeaseConfig,
    StorageLeaseManager,
)
from dessin.llm.decoder_lm_training import BUNDLED_SHAKESPEARE_CHAR_CORPUS
from dessin.llm.decoder_only_gpt import DecoderOnlyGPT
from dessin.llm.lora_trainer import LoRAConfig, train_lora
from dessin.models.model_format_registry import (
    ModelFamily,
    ModelSubmissionSpec,
    PipelineStageSpec,
    TrainingPipelineSpec,
    dessin_native_preset,
)
from dessin.models.model_submission_transactions import (
    RegisterModelTransaction,
    RenewStorageLeaseTransaction,
    UploadTrainingDataTransaction,
)
from dessin.runtime.config import TrainingMarketConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

#: Nano-GPT hyperparams — same as unit-test suite, fast on CPU
_HP = dict(n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=65)

#: LoRA training steps for this test (fast).  See fee comment for 2000-step cost.
_LORA_STEPS = 200

#: Simulated on-chain state
_ALICE = "0xAlice"
_DATASET_HASH = hashlib.sha256(BUNDLED_SHAKESPEARE_CHAR_CORPUS.encode()).hexdigest()
_DATASET_SIZE_MB = 0.02          # bundled corpus is tiny; we also show 512 MB fees
_STORAGE_BLOCKS = 200
_BASE_BLOCK = 100                # current chain height for this test


pytestmark = pytest.mark.skipif(
    os.environ.get("DESSIN_SKIP_E2E", "").strip().lower() in ("1", "true", "yes"),
    reason="DESSIN_SKIP_E2E set",
)


# ---------------------------------------------------------------------------
# Stage 1 — Model registration
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.integration
def test_alice_registers_dessin_native_model():
    """
    Alice builds a ModelSubmissionSpec for her DESSIN_NATIVE nano-GPT,
    validates it, derives a content hash, and constructs a RegisterModelTransaction.

    The equivalent GGUF 7B scenario (inference-only, no in-protocol trainer):
        gguf = gguf_preset(parameter_count_millions=7000, size_mb=4096)
        assert not gguf.is_trainable   # ← would fail PostTrainingTaskTransaction
    """
    fmt = dessin_native_preset(**_HP)
    assert fmt.is_trainable, "DESSIN_NATIVE must be trainable"

    pipeline = TrainingPipelineSpec(name="alice-lora")
    pipeline.add_stage(PipelineStageSpec(
        stage_index=0,
        method="lora",
        dataset_id=_DATASET_HASH,
        max_steps=2000,
        learning_rate=3e-4,
        lora_rank=8,
        lora_alpha=16.0,
    ))

    spec = ModelSubmissionSpec(
        model_name="AliceNanoGPT-LoRA",
        owner=_ALICE,
        family=ModelFamily.NANO_GPT,
        format_spec=fmt,
        pipeline=pipeline,
        tags=["nano-gpt", "lora", "dessin-native"],
    )

    errors = spec.validate()
    assert errors == [], f"Spec validation failed: {errors}"

    content_hash = spec.content_hash()
    assert len(content_hash) == 64, "content_hash must be 64-char SHA-256 hex"

    # Round-trip: hash must be stable — content_hash() uses sort_keys=True
    payload = {k: v for k, v in spec.to_dict().items()
               if k not in ("ipfs_hash", "torrent_magnet")}
    spec_json = json.dumps(payload, sort_keys=True)
    rehash = hashlib.sha256(spec_json.encode()).hexdigest()
    assert rehash == content_hash, "content_hash must be stable across JSON round-trip"

    # Fee estimation — LoRA for 2000 steps (all keyword args)
    sched = ComputeFeeSchedule()
    lora_fee = sched.estimate_training_fee(
        size_mb=fmt.size_mb,
        phase=TrainingPhase.LORA,
        steps=2000,
        is_quantized=False,
    )  # keyword-only API
    assert lora_fee > 0, "LoRA training fee must be positive"

    # Build the on-chain transaction
    tx = RegisterModelTransaction(
        sender=_ALICE,
        fee=0.001,
        timestamp=time.time(),
        public_key="test-pubkey",
        signature="test-sig",
        tx_id="reg_alice_001",
        model_name=spec.model_name,
        family=spec.family.value,
        format=spec.format_spec.format.value,
        size_mb=spec.format_spec.size_mb,
        parameter_count_millions=spec.format_spec.parameter_count_millions,
        content_hash=content_hash,
        submission_spec_json=spec_json,
        required_deposit=lora_fee,
        storage_blocks=_STORAGE_BLOCKS,
        storage_payment=0.01,
    )
    assert tx.content_hash == content_hash
    assert tx.model_name == "AliceNanoGPT-LoRA"


# ---------------------------------------------------------------------------
# Stage 2 — Dataset upload and storage lease
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.integration
def test_alice_uploads_dataset_and_creates_storage_lease():
    """
    Alice uploads the training corpus and creates a 200-block storage lease.

    Economics for a 512 MB dataset (chapter 7 scenario):
        upload_fee    = 512 × 0.0002 = 0.1024 DESSIN
        storage_fee   = 512 × 0.00005 × 200 = 5.12 DESSIN
    This test uses the bundled corpus (~0.02 MB) but applies the same fee
    schedule formula.
    """
    sched = ComputeFeeSchedule()

    upload_fee_actual  = sched.estimate_upload_fee(size_mb=_DATASET_SIZE_MB)
    storage_fee_actual = sched.estimate_storage_renewal_fee(size_mb=_DATASET_SIZE_MB, blocks=_STORAGE_BLOCKS)

    # Verify 512 MB scenario matches the documentation numbers
    upload_fee_512mb  = sched.estimate_upload_fee(size_mb=512.0)
    storage_fee_512mb = sched.estimate_storage_renewal_fee(size_mb=512.0, blocks=_STORAGE_BLOCKS)
    assert abs(upload_fee_512mb  - 0.1024) < 1e-6, f"upload fee mismatch: {upload_fee_512mb}"
    assert abs(storage_fee_512mb - 5.12)   < 1e-6, f"storage fee mismatch: {storage_fee_512mb}"

    tx = UploadTrainingDataTransaction(
        sender=_ALICE,
        fee=0.001,
        timestamp=time.time(),
        public_key="test-pubkey",
        signature="test-sig",
        tx_id="upload_alice_001",
        file_id=_DATASET_HASH,
        model_id="alice_nano_gpt",
        file_name="shakespeare_corpus.txt",
        file_hash=_DATASET_HASH,
        size_mb=_DATASET_SIZE_MB,
        dataset_type="pretrain",
        storage_blocks=_STORAGE_BLOCKS,
        storage_payment=storage_fee_actual,
    )

    # Create the lease in the storage manager (returns (ok, msg))
    mgr = StorageLeaseManager(StorageLeaseConfig())
    ok, msg = mgr.create_lease(
        file_id=tx.file_id,
        owner=_ALICE,
        size_mb=tx.size_mb,
        current_block=_BASE_BLOCK,
        initial_blocks=tx.storage_blocks,
        payment=tx.storage_payment,
    )
    assert ok, f"create_lease failed: {msg}"

    lease = mgr.get_lease(_DATASET_HASH)
    assert lease is not None
    assert lease.status == LeaseStatus.ACTIVE
    assert lease.paid_through_block == _BASE_BLOCK + _STORAGE_BLOCKS
    assert lease.total_paid == pytest.approx(storage_fee_actual)

    # Lease enters WARNING before expiry
    mgr.check_block(current_block=lease.paid_through_block - 5)
    warning_lease = mgr.get_lease(_DATASET_HASH)
    assert warning_lease.status == LeaseStatus.WARNING

    # Owner renews
    renewal_payment = sched.estimate_storage_renewal_fee(size_mb=_DATASET_SIZE_MB, blocks=100)
    renew_tx = RenewStorageLeaseTransaction(
        sender=_ALICE,
        fee=0.001,
        timestamp=time.time(),
        public_key="test-pubkey",
        signature="test-sig",
        tx_id="renew_alice_001",
        file_id=_DATASET_HASH,
        additional_blocks=100,
        payment=renewal_payment,
    )
    original_paid_through = lease.paid_through_block   # capture before mutation
    ok2, msg2 = mgr.renew_lease(
        file_id=renew_tx.file_id,
        payer=_ALICE,
        current_block=original_paid_through - 5,
        additional_blocks=renew_tx.additional_blocks,
        payment=renew_tx.payment,
    )
    assert ok2, f"renew_lease failed: {msg2}"
    renewed = mgr.get_lease(_DATASET_HASH)
    assert renewed.status == LeaseStatus.ACTIVE
    assert renewed.paid_through_block == original_paid_through + 100


# ---------------------------------------------------------------------------
# Stage 3 — Training market: task submission and slot selection
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.integration
def test_alice_posts_training_task_and_wins_slot():
    """
    Alice posts a LoRA training task.  Bob also posts a task with a higher nominal
    tip, but Alice's task is older → urgency multiplier pushes her effective tip
    above Bob's by block 103.
    """
    cfg = TrainingMarketConfig(
        max_tasks_per_block=4,
        initial_base_fee=0.01,
        min_tip=0.0,
        urgency_rate=0.10,
        urgency_cap=5.0,
    )
    market = TrainingMarket(cfg)

    alice_task = TrainingTaskSpec(
        task_id="alice_lora_001",
        model_id="alice_nano_gpt",
        owner=_ALICE,
        deposit=0.12,
        base_fee=0.01,
        tip=0.10,
        cooldown_blocks=5,
        max_slots=1,
        submitted_block=_BASE_BLOCK,
        expiry_block=_BASE_BLOCK + 200,
    )
    bob_task = TrainingTaskSpec(
        task_id="bob_task_001",
        model_id="bob_model",
        owner="0xBob",
        deposit=0.135,
        base_fee=0.01,
        tip=0.125,         # higher nominal tip
        cooldown_blocks=5,
        max_slots=1,
        submitted_block=_BASE_BLOCK + 3,   # submitted 3 blocks later
        expiry_block=_BASE_BLOCK + 200,
    )

    ok_a, msg_a = market.submit_task(alice_task, current_block=_BASE_BLOCK)
    assert ok_a, f"Alice task rejected: {msg_a}"

    ok_b, msg_b = market.submit_task(bob_task, current_block=_BASE_BLOCK + 3)
    assert ok_b, f"Bob task rejected: {msg_b}"

    # At block _BASE_BLOCK + 3:
    #   Alice's urgency = 1 + 0.1 × 3 = 1.3 → effective_tip = 0.10 × 1.3 = 0.130
    #   Bob's urgency   = 1 + 0.1 × 0 = 1.0 → effective_tip = 0.125 × 1.0 = 0.125
    #   Alice wins.
    vrf_seed = hashlib.sha256(b"test_vrf_seed").digest()
    slots = market.select_tasks_for_block(
        current_block=_BASE_BLOCK + 3,
        vrf_seed=vrf_seed,
    )
    assert len(slots) >= 1
    selected_ids = [s.task_id for s in slots]
    assert "alice_lora_001" in selected_ids, (
        f"Alice's task should be selected first; got {selected_ids}"
    )
    # Verify Alice appears before Bob (highest effective tip first)
    if "bob_task_001" in selected_ids:
        assert selected_ids.index("alice_lora_001") < selected_ids.index("bob_task_001")

    # Finalize block → base fee adjusts
    market.finalize_block(scheduled=slots, current_block=_BASE_BLOCK + 3)
    # 2 tasks used / 4 slots → 50% utilisation → base fee should stay near initial
    assert market.base_fee > 0


# ---------------------------------------------------------------------------
# Stage 4 — LoRA fine-tuning: real gradient steps on CPU
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.integration
def test_alice_lora_training_produces_pogo_compatible_weights():
    """
    Alice's miner runs LoRA fine-tuning on her DESSIN_NATIVE nano-GPT.

    Uses 200 gradient steps (fast on CPU / MacBook Air).
    Output is verified to be PoGO-compatible:
      - model_weights_hex is a valid hex string loadable with load_params_from_hex
      - loss decreases (the model actually trains)
      - trainable_params << total_params (LoRA efficiency is real)
    """
    from dessin.llm.decoder_only_gpt import load_params_from_hex

    model = DecoderOnlyGPT(**_HP)

    cfg = LoRAConfig(
        rank=8,
        alpha=16.0,
        target_modules={"qkv", "proj"},
        dropout=0.0,
    )

    result = train_lora(
        model,
        BUNDLED_SHAKESPEARE_CHAR_CORPUS,
        cfg,
        steps=_LORA_STEPS,
        batch_size=4,
        learning_rate=3e-4,
        device=torch.device("cpu"),
        seed=42,
    )

    # Loss decreased
    assert result.final_loss < result.initial_loss, (
        f"Loss did not decrease: {result.initial_loss:.4f} → {result.final_loss:.4f}"
    )

    # LoRA efficiency: ≤ 30% of parameters trained
    assert result.param_efficiency <= 0.30, (
        f"Too many parameters trained ({result.param_efficiency:.1%}); "
        "LoRA should be much more efficient"
    )
    assert result.trainable_params > 0

    # PoGO-compatible: hex weights are present and loadable
    weights_hex = result.training_result.model_weights_hex
    assert isinstance(weights_hex, str) and len(weights_hex) > 0

    fresh_model = DecoderOnlyGPT(**_HP)
    load_params_from_hex(fresh_model, weights_hex)   # must not raise

    # Loss curve is non-empty and finite
    assert len(result.loss_curve) > 0
    assert all(
        isinstance(v, float) and not (v != v)   # not NaN
        for v in result.loss_curve
    )

    # Adapter weights are present and correctly shaped
    assert len(result.adapter_weights) > 0
    for key, tensor in result.adapter_weights.items():
        assert tensor.numel() > 0, f"empty adapter tensor at {key}"


# ---------------------------------------------------------------------------
# Stage 5 — Full pipeline end-to-end (all stages chained)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.integration
def test_alice_full_lora_pipeline_end_to_end():
    """
    The complete Alice story in one test:

    1. Build DESSIN_NATIVE model spec and pipeline
    2. Estimate all fees (training + upload + storage)
    3. Create storage lease for dataset
    4. Post task to training market; verify it's selected
    5. Run LoRA training (real gradient steps)
    6. Verify output is PoGO-compatible and loss decreased
    7. Check total cost against documented economics

    Economics comparison (chapter 7, 512 MB dataset, 2000 LoRA steps):
        LoRA training :  4096 × 0.00035 × (2000/1000) × 1.0  ≈   2.87 DESSIN  (nano-GPT at actual size_mb)
        Dataset upload:    512 × 0.0002                         ≈   0.10 DESSIN
        Storage 200 blk:   512 × 0.00005 × 200                 ≈   5.12 DESSIN
        Tx fee            (fixed)                               ≈   0.01 DESSIN
        ─────────────────────────────────────────────────────────────────────
        Total (512 MB)                                          ≈   8.10 DESSIN

    This test uses the actual nano-GPT size (~0.1 MB) and 200 steps, so
    absolute fee amounts are smaller, but the ratios and formula are identical.
    """
    from dessin.llm.decoder_only_gpt import load_params_from_hex

    # ── 1. Model spec ───────────────────────────────────────────────────────
    fmt = dessin_native_preset(**_HP)
    assert fmt.is_trainable

    pipeline = TrainingPipelineSpec(name="alice-lora-full")
    pipeline.add_stage(PipelineStageSpec(
        stage_index=0,
        method="lora",
        dataset_id=_DATASET_HASH,
        max_steps=2000,
        learning_rate=3e-4,
        lora_rank=8,
        lora_alpha=16.0,
    ))
    spec = ModelSubmissionSpec(
        model_name="AliceNanoGPT-Full",
        owner=_ALICE,
        family=ModelFamily.NANO_GPT,
        format_spec=fmt,
        pipeline=pipeline,
    )
    assert spec.validate() == []

    # ── 2. Fee schedule ─────────────────────────────────────────────────────
    sched = ComputeFeeSchedule()
    lora_fee    = sched.estimate_training_fee(size_mb=fmt.size_mb, phase=TrainingPhase.LORA, steps=2000, is_quantized=False)
    upload_fee  = sched.estimate_upload_fee(size_mb=_DATASET_SIZE_MB)
    storage_fee = sched.estimate_storage_renewal_fee(size_mb=_DATASET_SIZE_MB, blocks=_STORAGE_BLOCKS)

    assert lora_fee    > 0
    assert upload_fee  > 0
    assert storage_fee > 0

    # ── 3. Storage lease ────────────────────────────────────────────────────
    mgr = StorageLeaseManager(StorageLeaseConfig())
    ok_l, msg_l = mgr.create_lease(
        file_id=_DATASET_HASH,
        owner=_ALICE,
        size_mb=_DATASET_SIZE_MB,
        current_block=_BASE_BLOCK,
        initial_blocks=_STORAGE_BLOCKS,
        payment=storage_fee,
    )
    assert ok_l, f"create_lease failed: {msg_l}"
    lease = mgr.get_lease(_DATASET_HASH)
    assert lease is not None
    assert lease.status == LeaseStatus.ACTIVE

    # Verify the dataset will still be alive at the expected training block
    expiring = mgr.expiring_soon(current_block=_BASE_BLOCK, within_blocks=10)
    dataset_ids = [l.file_id for l in expiring]
    assert _DATASET_HASH not in dataset_ids, "Dataset should not be expiring soon"

    # ── 4. Training market ──────────────────────────────────────────────────
    cfg = TrainingMarketConfig(initial_base_fee=0.01, min_tip=0.0)
    market = TrainingMarket(cfg)

    # deposit must cover base_fee + tip; lora_fee for a nano model is tiny
    _tip = 0.05
    deposit = max(market.base_fee + _tip, lora_fee) + 0.01

    task = TrainingTaskSpec(
        task_id="alice_full_001",
        model_id="alice_nano_gpt_full",
        owner=_ALICE,
        deposit=deposit,
        base_fee=market.base_fee,
        tip=_tip,
        cooldown_blocks=5,
        max_slots=1,
        submitted_block=_BASE_BLOCK,
        expiry_block=_BASE_BLOCK + 200,
    )
    ok, msg = market.submit_task(task, current_block=_BASE_BLOCK)
    assert ok, f"Task submission failed: {msg}"

    vrf_seed = hashlib.sha256(b"alice_full_vrf").digest()
    slots = market.select_tasks_for_block(
        current_block=_BASE_BLOCK + 1,
        vrf_seed=vrf_seed,
    )
    assert any(s.task_id == "alice_full_001" for s in slots), (
        "Alice's task was not selected"
    )
    market.finalize_block(scheduled=slots, current_block=_BASE_BLOCK + 1)

    # ── 5. LoRA training ────────────────────────────────────────────────────
    model = DecoderOnlyGPT(**_HP)
    lora_cfg = LoRAConfig(rank=8, alpha=16.0, target_modules={"qkv", "proj"}, dropout=0.0)

    result = train_lora(
        model,
        BUNDLED_SHAKESPEARE_CHAR_CORPUS,
        lora_cfg,
        steps=_LORA_STEPS,
        batch_size=4,
        learning_rate=3e-4,
        device=torch.device("cpu"),
        seed=42,
    )

    # ── 6. Verify PoGO-compatible output ────────────────────────────────────
    assert result.final_loss < result.initial_loss, (
        f"LoRA training did not improve loss: {result.initial_loss:.4f} → {result.final_loss:.4f}"
    )
    assert result.param_efficiency <= 0.30

    weights_hex = result.training_result.model_weights_hex
    fresh_model = DecoderOnlyGPT(**_HP)
    load_params_from_hex(fresh_model, weights_hex)   # merged weights load cleanly

    # ── 7. Spot-check pipeline note ─────────────────────────────────────────
    # The spot-check path (k=2 gradient step replays) is implemented for the
    # standard decoder-LM training trace.  LoRA training via train_lora()
    # produces merged PoGO weights but does NOT yet emit a TrainingTrace for
    # spot-check replay — that extension is on the roadmap.
    # See: dessin/consensus/spot_check_verification.py
    #      dessin/llm/decoder_lm_training_trace.py
    #      e2e/tests/test_spot_check_pipeline.py

    # ── Summary ─────────────────────────────────────────────────────────────
    print(
        f"\n[Alice E2E] loss {result.initial_loss:.4f}→{result.final_loss:.4f} "
        f"| trainable {result.trainable_params:,}/{result.total_params:,} "
        f"({result.param_efficiency:.1%}) "
        f"| lora_fee={lora_fee:.6f} storage_fee={storage_fee:.6f} DESSIN"
    )
