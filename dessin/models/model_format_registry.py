"""
Model Format Registry
=====================

Standardised envelope for model formats supported by DeSSIN.

Supported formats
-----------------
DESSIN_NATIVE   – DecoderOnlyGPT hex-JSON weights.  **The only format with
                  a complete PoGO gradient-training pipeline** (dessin/llm/).
                  Trained with ``train_decoder_only_char_lm`` and friends.
GGUF            – llama.cpp quantised weights.  Inference-only via
                  ``llama-cpp-python`` (optional install).  Cannot be
                  trained inside PoGO blocks.
SAFETENSORS     – HuggingFace checkpoint.  Inference-only until ``peft``
                  is added as a project dependency.

Do not add new values to ``ModelFormat`` without a matching trainer module
under ``dessin/llm/`` and passing unit tests that perform real gradient steps.

Usage
-----
Build a :class:`ModelSubmissionSpec` and call
:meth:`ModelSubmissionSpec.validate` before submitting a
``RegisterModelTransaction``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelFormat(str, Enum):
    """
    On-disk container / serialisation format.

    Only ``DESSIN_NATIVE`` supports full PoGO gradient training inside the
    current codebase (``dessin/llm/`` trainers, verified by spot-check).
    ``GGUF`` is accepted for registration and local *inference* via
    ``llama-cpp-python``, but the PoGO consensus cannot train a GGUF model
    directly.  Do not add new values here without a matching trainer module.
    """

    DESSIN_NATIVE = "dessin_native"   # DecoderOnlyGPT hex-JSON — fully trainable
    GGUF = "gguf"                     # llama.cpp quants — inference only
    SAFETENSORS = "safetensors"       # HuggingFace — inference only (no peft dep yet)


class ModelPrecision(str, Enum):
    """
    Numeric precision / quantisation level of the stored weights.

    Used by the fee schedule (quantised models are cheaper to run).
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    INT4_K_M = "int4_k_m"   # GGUF k-quant medium
    INT4_K_S = "int4_k_s"   # GGUF k-quant small
    INT3_K_M = "int3_k_m"
    INT2_K = "int2_k"
    CUSTOM = "custom"


class ModelFamily(str, Enum):
    """High-level architecture lineage."""

    LLAMA = "llama"
    MISTRAL = "mistral"
    GPT2 = "gpt2"
    GEMMA = "gemma"
    PHI = "phi"
    QWEN = "qwen"
    FALCON = "falcon"
    MPT = "mpt"
    BLOOM = "bloom"
    NANO_GPT = "nano_gpt"       # DeSSIN micro-GPT (decoder_only_transformer)
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Hardware / capability requirements
# ---------------------------------------------------------------------------

@dataclass
class HardwareRequirements:
    """
    Minimum hardware needed to run / train this model.

    All memory values are in **megabytes** so they stay integers across size
    ranges (a 7-B model needs ~14 000 MB in fp16).
    """

    min_ram_mb: int = 0
    min_vram_mb: int = 0            # 0 = CPU-only model acceptable
    min_compute_capability: float = 0.0   # CUDA SM version, 0 = no GPU required
    supports_cpu_inference: bool = True
    supports_metal: bool = False    # Apple GPU (MLX)
    supports_cuda: bool = False
    recommended_batch_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardwareRequirements":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Per-format metadata
# ---------------------------------------------------------------------------

@dataclass
class GGUFMetadata:
    """Extra fields present in a GGUF file header."""

    architecture: str = ""          # e.g. "llama"
    quantization_version: int = 2   # GGUF quant version
    context_length: int = 4096
    rope_freq_base: float = 10000.0
    rope_scaling_type: str = "none"
    tensor_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "GGUFMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MLXMetadata:
    """Extra fields for Apple MLX format."""

    mlx_version: str = ""
    sharded: bool = False
    shard_count: int = 1
    config_file: str = "config.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MLXMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Core spec
# ---------------------------------------------------------------------------

MODEL_FORMAT_REGISTRY_VERSION = "1.0"


@dataclass
class ModelFormatSpec:
    """
    Format envelope that wraps any model weight file.

    This is *independent* of the neural architecture (e.g. a LLaMA-3 8-B
    model could be stored as GGUF, SafeTensors, or MLX — same architecture,
    different format specs).

    Fields
    ------
    format:
        Container format (see :class:`ModelFormat`).
    precision:
        Numeric dtype / quantisation level.
    parameter_count_millions:
        Total trainable parameters in millions (used for fee calculation).
    size_mb:
        Disk footprint of the complete model in megabytes.
    context_length:
        Maximum input context in tokens.
    vocab_size:
        Vocabulary size (0 = unknown / not applicable).
    hidden_size:
        Primary hidden dimension; 0 = unknown.
    num_layers:
        Transformer layer count; 0 = unknown.
    num_attention_heads:
        Attention head count; 0 = unknown.
    hardware:
        Minimum hardware to load/run this model.
    gguf_metadata / mlx_metadata:
        Optional format-specific fields.
    extra:
        Open dict for future fields without breaking schema.
    """

    format: ModelFormat
    precision: ModelPrecision
    parameter_count_millions: float     # e.g. 7000.0 for 7B
    size_mb: float                      # disk size of weights
    context_length: int = 4096
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    hardware: HardwareRequirements = field(default_factory=HardwareRequirements)
    gguf_metadata: Optional[GGUFMetadata] = None
    mlx_metadata: Optional[MLXMetadata] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def size_gb(self) -> float:
        return self.size_mb / 1024.0

    @property
    def is_quantized(self) -> bool:
        return self.precision in (
            ModelPrecision.INT8,
            ModelPrecision.INT4,
            ModelPrecision.INT4_K_M,
            ModelPrecision.INT4_K_S,
            ModelPrecision.INT3_K_M,
            ModelPrecision.INT2_K,
        )

    @property
    def is_trainable(self) -> bool:
        """
        True only for formats that have a real gradient-training pipeline in
        ``dessin/llm/``.  Attempting to schedule PoGO training for a
        non-trainable format will be rejected by the training market.
        """
        return self.format == ModelFormat.DESSIN_NATIVE

    def to_dict(self) -> Dict[str, Any]:
        # Explicit casts ensure int/float types are stable across JSON round-trips.
        d: Dict[str, Any] = {
            "schema_version": MODEL_FORMAT_REGISTRY_VERSION,
            "format": self.format.value,
            "precision": self.precision.value,
            "parameter_count_millions": float(self.parameter_count_millions),
            "size_mb": float(self.size_mb),
            "context_length": int(self.context_length),
            "vocab_size": int(self.vocab_size),
            "hidden_size": int(self.hidden_size),
            "num_layers": int(self.num_layers),
            "num_attention_heads": int(self.num_attention_heads),
            "hardware": self.hardware.to_dict(),
            "extra": self.extra,
        }
        if self.gguf_metadata:
            d["gguf_metadata"] = self.gguf_metadata.to_dict()
        if self.mlx_metadata:
            d["mlx_metadata"] = self.mlx_metadata.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelFormatSpec":
        hw = HardwareRequirements.from_dict(d.get("hardware") or {})
        gguf = GGUFMetadata.from_dict(d["gguf_metadata"]) if d.get("gguf_metadata") else None
        mlx = MLXMetadata.from_dict(d["mlx_metadata"]) if d.get("mlx_metadata") else None
        return cls(
            format=ModelFormat(d["format"]),
            precision=ModelPrecision(d["precision"]),
            parameter_count_millions=float(d["parameter_count_millions"]),
            size_mb=float(d["size_mb"]),
            context_length=int(d.get("context_length", 4096)),
            vocab_size=int(d.get("vocab_size", 0)),
            hidden_size=int(d.get("hidden_size", 0)),
            num_layers=int(d.get("num_layers", 0)),
            num_attention_heads=int(d.get("num_attention_heads", 0)),
            hardware=hw,
            gguf_metadata=gguf,
            mlx_metadata=mlx,
            extra=dict(d.get("extra") or {}),
        )


# ---------------------------------------------------------------------------
# Fine-tuning pipeline reference
# ---------------------------------------------------------------------------

@dataclass
class PipelineStageSpec:
    """
    One stage in a multi-step fine-tuning pipeline.

    ``method`` must match a :class:`~dessin.fine_tuning.fine_tuning_techniques.FineTuningMethod`
    value (stored as a plain string so this module has no circular import).
    """

    stage_index: int
    method: str                       # e.g. "lora", "full", "qlora"
    dataset_id: str                   # reference to a UploadTrainingDataTransaction file_id
    max_steps: int = 1000
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    lora_rank: int = 8                # ignored for non-LoRA methods
    lora_alpha: int = 16
    quantization: str = "none"        # none / int4 / int8
    extra_hyperparams: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Normalise numeric types so JSON round-trips produce identical strings.
        d["stage_index"] = int(d["stage_index"])
        d["max_steps"] = int(d["max_steps"])
        d["batch_size"] = int(d["batch_size"])
        d["gradient_accumulation_steps"] = int(d["gradient_accumulation_steps"])
        d["warmup_steps"] = int(d["warmup_steps"])
        d["lora_rank"] = int(d["lora_rank"])
        d["lora_alpha"] = int(d["lora_alpha"])
        d["learning_rate"] = float(d["learning_rate"])
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "PipelineStageSpec":
        hp = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**hp)


@dataclass
class TrainingPipelineSpec:
    """
    Ordered sequence of fine-tuning stages the owner wants applied to this model.

    Stages are executed one after another; the output checkpoint of stage N is
    the input of stage N+1.  Each stage references a dataset that must have a
    valid ``StorageLease`` at execution time.
    """

    stages: List[PipelineStageSpec] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def add_stage(self, stage: PipelineStageSpec) -> None:
        stage.stage_index = len(self.stages)
        self.stages.append(stage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingPipelineSpec":
        stages = [PipelineStageSpec.from_dict(s) for s in d.get("stages", [])]
        return cls(stages=stages, name=d.get("name", ""), description=d.get("description", ""))


# ---------------------------------------------------------------------------
# Full submission spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSubmissionSpec:
    """
    Everything needed to register a model on DeSSIN.

    Combines the format envelope (:class:`ModelFormatSpec`), the owner's
    requested training pipeline (:class:`TrainingPipelineSpec`), and
    administrative metadata.

    The :meth:`content_hash` is stored on-chain so nodes can verify they
    received the correct manifest without downloading the weights.
    """

    model_name: str
    owner: str
    family: ModelFamily
    format_spec: ModelFormatSpec
    pipeline: TrainingPipelineSpec = field(default_factory=TrainingPipelineSpec)
    license: str = "apache-2.0"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    source_repo: str = ""             # e.g. "https://huggingface.co/meta-llama/…"
    ipfs_hash: str = ""               # populated after upload
    torrent_magnet: str = ""          # populated after torrent seeding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "owner": self.owner,
            "family": self.family.value,
            "format_spec": self.format_spec.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "license": self.license,
            "tags": list(self.tags),
            "description": self.description,
            "source_repo": self.source_repo,
            "ipfs_hash": self.ipfs_hash,
            "torrent_magnet": self.torrent_magnet,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSubmissionSpec":
        return cls(
            model_name=d["model_name"],
            owner=d["owner"],
            family=ModelFamily(d.get("family", "custom")),
            format_spec=ModelFormatSpec.from_dict(d["format_spec"]),
            pipeline=TrainingPipelineSpec.from_dict(d.get("pipeline") or {}),
            license=d.get("license", "apache-2.0"),
            tags=list(d.get("tags") or []),
            description=d.get("description", ""),
            source_repo=d.get("source_repo", ""),
            ipfs_hash=d.get("ipfs_hash", ""),
            torrent_magnet=d.get("torrent_magnet", ""),
        )

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON representation (no ipfs/torrent fields)."""
        payload = {k: v for k, v in self.to_dict().items()
                   if k not in ("ipfs_hash", "torrent_magnet")}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def validate(self) -> List[str]:
        """
        Return a list of validation error strings.
        Empty list means the spec is valid.
        """
        errors: List[str] = []
        if not self.model_name.strip():
            errors.append("model_name is required")
        if not self.owner.strip():
            errors.append("owner address is required")
        if self.format_spec.size_mb <= 0:
            errors.append("format_spec.size_mb must be > 0")
        if self.format_spec.parameter_count_millions < 0:
            errors.append("parameter_count_millions must be >= 0")
        for i, stage in enumerate(self.pipeline.stages):
            if not stage.dataset_id.strip():
                errors.append(f"pipeline.stages[{i}].dataset_id is required")
            if stage.max_steps <= 0:
                errors.append(f"pipeline.stages[{i}].max_steps must be > 0")
        return errors


# ---------------------------------------------------------------------------
# Common presets
# ---------------------------------------------------------------------------

def dessin_native_preset(
    *,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    vocab_size: int,
    corpus_id: str = "shakespeare_sonnet1_excerpt_x80",
) -> ModelFormatSpec:
    """
    Spec for DeSSIN's own DecoderOnlyGPT char LM.

    This is the only format with a complete PoGO training pipeline
    (``dessin/llm/train_decoder_only_char_lm``).  Weights are stored as
    hex-encoded float64 JSON blobs and distributed via BitTorrent.
    """
    # Rough size estimate: each param is float64 (8 bytes)
    n_params = (
        vocab_size * n_embd           # token_emb
        + block_size * n_embd         # pos_emb
        + n_layer * (                 # transformer blocks
            3 * n_embd * n_embd       # qkv
            + n_embd * n_embd         # proj
            + 4 * n_embd * n_embd     # mlp up
            + 4 * n_embd * n_embd     # mlp down
        )
        + vocab_size * n_embd         # lm_head
    )
    size_mb = float(n_params * 8 / 1024 / 1024)  # float64
    return ModelFormatSpec(
        format=ModelFormat.DESSIN_NATIVE,
        precision=ModelPrecision.FP32,
        parameter_count_millions=float(n_params / 1_000_000),
        size_mb=size_mb,
        context_length=block_size,
        vocab_size=vocab_size,
        num_layers=n_layer,
        num_attention_heads=n_head,
        hidden_size=n_embd,
        hardware=HardwareRequirements(
            min_ram_mb=max(64, int(size_mb * 4)),
            supports_cpu_inference=True,
            supports_cuda=True,
        ),
        extra={"corpus_id": corpus_id},
    )


def gguf_preset(
    *,
    parameter_count_millions: float,
    size_mb: float,
    precision: ModelPrecision = ModelPrecision.INT4_K_M,
    context_length: int = 4096,
    architecture: str = "llama",
) -> ModelFormatSpec:
    """
    Spec for a GGUF model (llama.cpp format).

    **Inference only** — no PoGO gradient training is implemented for this
    format.  Register for hosting and querying only.
    """
    return ModelFormatSpec(
        format=ModelFormat.GGUF,
        precision=precision,
        parameter_count_millions=float(parameter_count_millions),
        size_mb=float(size_mb),
        context_length=context_length,
        hardware=HardwareRequirements(
            min_ram_mb=int(size_mb * 1.2),
            supports_cpu_inference=True,
            supports_cuda=True,
        ),
        gguf_metadata=GGUFMetadata(architecture=architecture, context_length=context_length),
    )


def safetensors_preset(
    *,
    parameter_count_millions: float,
    size_mb: float,
    precision: ModelPrecision = ModelPrecision.BF16,
    context_length: int = 2048,
    num_layers: int = 0,
    hidden_size: int = 0,
) -> ModelFormatSpec:
    """
    Spec for a HuggingFace SafeTensors checkpoint.

    **Inference only** — no PoGO gradient training is implemented for this
    format until ``peft`` is added as a project dependency.
    """
    return ModelFormatSpec(
        format=ModelFormat.SAFETENSORS,
        precision=precision,
        parameter_count_millions=float(parameter_count_millions),
        size_mb=float(size_mb),
        context_length=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        hardware=HardwareRequirements(
            min_vram_mb=int(size_mb * 1.1),
            supports_cpu_inference=True,
            supports_cuda=True,
        ),
    )
