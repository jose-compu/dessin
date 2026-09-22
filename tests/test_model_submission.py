"""
Tests for model format registry, submission spec, pipeline spec, and
RegisterModelTransaction / UploadTrainingDataTransaction /
RenewStorageLeaseTransaction.
"""

import json
import pytest

from dessin.models.model_format_registry import (
    ModelFormat,
    ModelFamily,
    ModelPrecision,
    ModelFormatSpec,
    HardwareRequirements,
    GGUFMetadata,
    PipelineStageSpec,
    TrainingPipelineSpec,
    ModelSubmissionSpec,
    dessin_native_preset,
    gguf_preset,
    safetensors_preset,
)
from dessin.models.model_submission_transactions import (
    RegisterModelTransaction,
    UploadTrainingDataTransaction,
    RenewStorageLeaseTransaction,
)
from dessin.economics.compute_fee_schedule import ComputeFeeSchedule, TrainingPhase


# ---------------------------------------------------------------------------
# ModelFormatSpec
# ---------------------------------------------------------------------------

class TestModelFormatSpec:
    def test_gguf_preset_round_trip(self):
        spec = gguf_preset(
            parameter_count_millions=7000,
            size_mb=4096,
            precision=ModelPrecision.INT4_K_M,
            context_length=4096,
            architecture="llama",
        )
        d = spec.to_dict()
        spec2 = ModelFormatSpec.from_dict(d)
        assert spec2.format == ModelFormat.GGUF
        assert spec2.precision == ModelPrecision.INT4_K_M
        assert spec2.size_mb == 4096
        assert spec2.is_quantized is True
        assert spec2.gguf_metadata is not None
        assert spec2.gguf_metadata.architecture == "llama"

    def test_dessin_native_preset_round_trip(self):
        spec = dessin_native_preset(
            n_layer=2, n_head=2, n_embd=64, block_size=32, vocab_size=65
        )
        d = spec.to_dict()
        spec2 = ModelFormatSpec.from_dict(d)
        assert spec2.format == ModelFormat.DESSIN_NATIVE
        assert spec2.is_trainable is True
        assert spec2.num_layers == 2
        assert spec2.hidden_size == 64
        assert spec2.context_length == 32

    def test_safetensors_preset_round_trip(self):
        spec = safetensors_preset(
            parameter_count_millions=1000,
            size_mb=2000,
            precision=ModelPrecision.FP16,
            num_layers=24,
            hidden_size=1024,
        )
        d = spec.to_dict()
        spec2 = ModelFormatSpec.from_dict(d)
        assert spec2.format == ModelFormat.SAFETENSORS
        assert spec2.num_layers == 24
        assert spec2.hidden_size == 1024
        assert spec2.is_quantized is False

    def test_size_gb_property(self):
        spec = gguf_preset(parameter_count_millions=7000, size_mb=4096)
        assert spec.size_gb == pytest.approx(4.0)

    def test_fp32_not_quantized(self):
        spec = safetensors_preset(
            parameter_count_millions=100, size_mb=400, precision=ModelPrecision.FP32
        )
        assert spec.is_quantized is False


# ---------------------------------------------------------------------------
# TrainingPipelineSpec
# ---------------------------------------------------------------------------

class TestPipelineSpec:
    def _make_pipeline(self) -> TrainingPipelineSpec:
        p = TrainingPipelineSpec(name="two-stage", description="SFT then DPO")
        p.add_stage(PipelineStageSpec(
            stage_index=0, method="lora", dataset_id="ds_001",
            max_steps=500, learning_rate=2e-4,
        ))
        p.add_stage(PipelineStageSpec(
            stage_index=1, method="dpo", dataset_id="ds_002",
            max_steps=200,
        ))
        return p

    def test_stage_count(self):
        p = self._make_pipeline()
        assert len(p.stages) == 2

    def test_round_trip(self):
        p = self._make_pipeline()
        d = p.to_dict()
        p2 = TrainingPipelineSpec.from_dict(d)
        assert len(p2.stages) == 2
        assert p2.stages[0].method == "lora"
        assert p2.stages[1].method == "dpo"
        assert p2.stages[0].dataset_id == "ds_001"

    def test_auto_stage_index(self):
        p = TrainingPipelineSpec()
        p.add_stage(PipelineStageSpec(stage_index=0, method="full", dataset_id="ds_x", max_steps=100))
        p.add_stage(PipelineStageSpec(stage_index=0, method="lora", dataset_id="ds_y", max_steps=50))
        assert p.stages[1].stage_index == 1


# ---------------------------------------------------------------------------
# ModelSubmissionSpec
# ---------------------------------------------------------------------------

class TestModelSubmissionSpec:
    def _make_spec(self) -> ModelSubmissionSpec:
        fmt = gguf_preset(parameter_count_millions=7000, size_mb=4096)
        pipeline = TrainingPipelineSpec(name="lora-only")
        pipeline.add_stage(PipelineStageSpec(
            stage_index=0, method="lora", dataset_id="ds_abc", max_steps=1000
        ))
        return ModelSubmissionSpec(
            model_name="LLaMA-7B-Q4",
            owner="0xAlice",
            family=ModelFamily.LLAMA,
            format_spec=fmt,
            pipeline=pipeline,
            license="apache-2.0",
            tags=["llm", "gguf"],
        )

    def test_round_trip(self):
        spec = self._make_spec()
        d = spec.to_dict()
        spec2 = ModelSubmissionSpec.from_dict(d)
        assert spec2.model_name == "LLaMA-7B-Q4"
        assert spec2.family == ModelFamily.LLAMA
        assert spec2.format_spec.format == ModelFormat.GGUF
        assert len(spec2.pipeline.stages) == 1

    def test_content_hash_stable(self):
        spec = self._make_spec()
        h1 = spec.content_hash()
        h2 = spec.content_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_content_hash_ignores_ipfs(self):
        spec = self._make_spec()
        h1 = spec.content_hash()
        spec.ipfs_hash = "QmSomething"
        h2 = spec.content_hash()
        assert h1 == h2

    def test_content_hash_changes_with_name(self):
        spec = self._make_spec()
        h1 = spec.content_hash()
        spec.model_name = "LLaMA-7B-Q4-modified"
        h2 = spec.content_hash()
        assert h1 != h2

    def test_validate_ok(self):
        spec = self._make_spec()
        errors = spec.validate()
        assert errors == []

    def test_validate_missing_name(self):
        spec = self._make_spec()
        spec.model_name = ""
        errors = spec.validate()
        assert any("model_name" in e for e in errors)

    def test_validate_missing_dataset_id(self):
        spec = self._make_spec()
        spec.pipeline.stages[0].dataset_id = ""
        errors = spec.validate()
        assert any("dataset_id" in e for e in errors)

    def test_validate_zero_size(self):
        spec = self._make_spec()
        spec.format_spec = gguf_preset(parameter_count_millions=7000, size_mb=0)
        errors = spec.validate()
        assert any("size_mb" in e for e in errors)


# ---------------------------------------------------------------------------
# ComputeFeeSchedule  (fee estimation)
# ---------------------------------------------------------------------------

class TestComputeFeeSchedule:
    def setup_method(self):
        self.sched = ComputeFeeSchedule()

    def test_fee_positive(self):
        fee = self.sched.estimate_training_fee(
            size_mb=4096, phase=TrainingPhase.LORA, steps=1000
        )
        assert fee > 0

    def test_lora_cheaper_than_full(self):
        full_fee = self.sched.estimate_training_fee(
            size_mb=4096, phase=TrainingPhase.FULL_FINETUNE, steps=1000
        )
        lora_fee = self.sched.estimate_training_fee(
            size_mb=4096, phase=TrainingPhase.LORA, steps=1000
        )
        assert lora_fee < full_fee

    def test_pretraining_most_expensive_standard(self):
        pretrain = self.sched.estimate_training_fee(
            size_mb=1000, phase=TrainingPhase.PRETRAINING, steps=1000
        )
        rlhf = self.sched.estimate_training_fee(
            size_mb=1000, phase=TrainingPhase.RLHF, steps=1000
        )
        prompt = self.sched.estimate_training_fee(
            size_mb=1000, phase=TrainingPhase.PROMPT_TUNING, steps=1000
        )
        assert pretrain > rlhf > prompt

    def test_quantized_discount_applied(self):
        fee_full = self.sched.estimate_training_fee(
            size_mb=2048, phase=TrainingPhase.FULL_FINETUNE, steps=1000, is_quantized=False
        )
        fee_quant = self.sched.estimate_training_fee(
            size_mb=2048, phase=TrainingPhase.FULL_FINETUNE, steps=1000, is_quantized=True
        )
        assert fee_quant < fee_full

    def test_steps_scale_linearly(self):
        f1 = self.sched.estimate_training_fee(
            size_mb=512, phase=TrainingPhase.SFT, steps=1000
        )
        f2 = self.sched.estimate_training_fee(
            size_mb=512, phase=TrainingPhase.SFT, steps=2000
        )
        assert f2 == pytest.approx(f1 * 2, rel=1e-6)

    def test_upload_fee_positive(self):
        assert self.sched.estimate_upload_fee(size_mb=512) > 0

    def test_renewal_fee_scales_with_blocks(self):
        r1 = self.sched.estimate_storage_renewal_fee(size_mb=512, blocks=100)
        r2 = self.sched.estimate_storage_renewal_fee(size_mb=512, blocks=200)
        assert r2 == pytest.approx(r1 * 2, rel=1e-6)

    def test_pipeline_fee_sums_stages(self):
        stages = [
            {"method": "lora", "max_steps": 1000},
            {"method": "dpo", "max_steps": 500},
        ]
        total = self.sched.estimate_pipeline_fee(size_mb=2048, stages=stages)
        lora = self.sched.estimate_training_fee(size_mb=2048, phase=TrainingPhase.LORA, steps=1000)
        dpo = self.sched.estimate_training_fee(size_mb=2048, phase=TrainingPhase.DPO, steps=500)
        assert total == pytest.approx(lora + dpo, rel=1e-6)

    def test_min_fee_floor(self):
        # Tiny model, few steps — should never go below min_fee
        fee = self.sched.estimate_training_fee(
            size_mb=0.001, phase=TrainingPhase.PROMPT_TUNING, steps=1
        )
        assert fee >= self.sched.min_fee

    def test_rate_card_keys(self):
        card = self.sched.rate_card()
        assert TrainingPhase.LORA.value in card
        assert TrainingPhase.PRETRAINING.value in card


# ---------------------------------------------------------------------------
# Transaction types
# ---------------------------------------------------------------------------

class TestModelSubmissionTransactions:
    def _make_register_tx(self, spec: ModelSubmissionSpec) -> RegisterModelTransaction:
        sched = ComputeFeeSchedule()
        deposit = sched.estimate_pipeline_fee(
            size_mb=spec.format_spec.size_mb,
            stages=[s.to_dict() for s in spec.pipeline.stages],
            is_quantized=spec.format_spec.is_quantized,
        )
        return RegisterModelTransaction(
            sender="0xAlice",
            fee=0.01,
            timestamp=1.0,
            public_key="pk_alice",
            signature="sig",
            tx_id="reg_001",
            model_name=spec.model_name,
            family=spec.family.value,
            format=spec.format_spec.format.value,
            size_mb=spec.format_spec.size_mb,
            parameter_count_millions=spec.format_spec.parameter_count_millions,
            content_hash=spec.content_hash(),
            submission_spec_json=json.dumps(spec.to_dict()),
            required_deposit=deposit,
            storage_blocks=100,
            storage_payment=0.5,
        )

    def test_register_model_tx_fields(self):
        fmt = gguf_preset(parameter_count_millions=7000, size_mb=4096)
        pipeline = TrainingPipelineSpec()
        pipeline.add_stage(PipelineStageSpec(stage_index=0, method="lora", dataset_id="ds_1", max_steps=1000))
        spec = ModelSubmissionSpec(model_name="Test", owner="0xAlice", family=ModelFamily.LLAMA, format_spec=fmt, pipeline=pipeline)
        tx = self._make_register_tx(spec)
        data = tx.get_transaction_data()
        assert data["type"] == "register_model"
        assert data["format"] == ModelFormat.GGUF.value
        assert data["size_mb"] == 4096

    def test_verify_content_hash(self):
        fmt = gguf_preset(parameter_count_millions=7000, size_mb=4096)
        pipeline = TrainingPipelineSpec()
        pipeline.add_stage(PipelineStageSpec(stage_index=0, method="lora", dataset_id="ds_1", max_steps=1000))
        spec = ModelSubmissionSpec(model_name="Test", owner="0xAlice", family=ModelFamily.LLAMA, format_spec=fmt, pipeline=pipeline)
        tx = self._make_register_tx(spec)
        assert tx.verify_content_hash() is True

    def test_tampered_spec_fails_hash(self):
        fmt = gguf_preset(parameter_count_millions=7000, size_mb=4096)
        pipeline = TrainingPipelineSpec()
        pipeline.add_stage(PipelineStageSpec(stage_index=0, method="lora", dataset_id="ds_1", max_steps=1000))
        spec = ModelSubmissionSpec(model_name="Test", owner="0xAlice", family=ModelFamily.LLAMA, format_spec=fmt, pipeline=pipeline)
        tx = self._make_register_tx(spec)
        # Tamper the JSON
        d = json.loads(tx.submission_spec_json)
        d["model_name"] = "TAMPERED"
        tx.submission_spec_json = json.dumps(d)
        assert tx.verify_content_hash() is False

    def test_upload_training_data_tx(self):
        tx = UploadTrainingDataTransaction(
            sender="0xBob",
            fee=0.01,
            timestamp=2.0,
            public_key="pk_bob",
            signature="sig2",
            tx_id="upload_001",
            file_id="file_abc123",
            model_id="model_001",
            file_name="train.jsonl",
            file_hash="deadbeef" * 8,
            size_mb=128.0,
            dataset_type="sft",
            storage_blocks=200,
            storage_payment=1.28,
        )
        data = tx.get_transaction_data()
        assert data["type"] == "upload_training_data"
        assert data["size_mb"] == 128.0
        assert data["storage_blocks"] == 200

    def test_renew_storage_lease_tx(self):
        tx = RenewStorageLeaseTransaction(
            sender="0xBob",
            fee=0.001,
            timestamp=3.0,
            public_key="pk_bob",
            signature="sig3",
            tx_id="renew_001",
            file_id="file_abc123",
            additional_blocks=100,
            payment=0.64,
        )
        data = tx.get_transaction_data()
        assert data["type"] == "renew_storage_lease"
        assert data["additional_blocks"] == 100
