"""LLM subpackage for DeSSIN language model training and inference."""

from .decoder_lm_training import (
    train_decoder_only_char_lm,
    BUNDLED_SHAKESPEARE_CHAR_CORPUS,
)
from .decoder_lm_training_trace import (
    train_decoder_only_char_lm_with_trace,
    TrainingTracePrivate,
    make_replay_fn,
)
from .decoder_lm_inference import (
    load_decoder_lm_from_torrent_json,
    greedy_generate_char_lm,
    format_prompt_for_char_lm_profile,
)
from .decoder_only_gpt import (
    DecoderOnlyGPT,
    CharTokenizer,
    load_params_from_hex,
)
from .micro_gpt_trainer import (
    train_micro_gpt,
)
from .gpt2_nano_trainer import (
    train_gpt2_nano,
)
from .simple_trainer import (
    SimpleTrainer,
    TrainingResult,
)
from .simple_gpt_inference import (
    generate_from_nanochat_model,
)
from .torrent_model_trainer import (
    TorrentModelTrainer,
    TorrentTrainingResult,
)
from .lora_trainer import (
    LoRAConfig,
    LoRALinear,
    LoRATrainingResult,
    apply_lora_to_model,
    strip_lora,
    lora_param_count,
    train_lora,
)

__all__ = [
    # Decoder LM training
    "train_decoder_only_char_lm",
    "BUNDLED_SHAKESPEARE_CHAR_CORPUS",
    # Training trace
    "train_decoder_only_char_lm_with_trace",
    "TrainingTracePrivate",
    "make_replay_fn",
    # Inference
    "load_decoder_lm_from_torrent_json",
    "greedy_generate_char_lm",
    "format_prompt_for_char_lm_profile",
    # Decoder GPT
    "DecoderOnlyGPT",
    "CharTokenizer",
    "load_params_from_hex",
    # Micro GPT
    "train_micro_gpt",
    # GPT2 Nano
    "train_gpt2_nano",
    # Simple trainer
    "SimpleTrainer",
    "TrainingResult",
    # Simple GPT inference
    "generate_from_nanochat_model",
    # Torrent trainer
    "TorrentModelTrainer",
    "TorrentTrainingResult",
    # LoRA fine-tuning
    "LoRAConfig",
    "LoRALinear",
    "LoRATrainingResult",
    "apply_lora_to_model",
    "strip_lora",
    "lora_param_count",
    "train_lora",
]
