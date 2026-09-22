"""GPT-2 nano preset shares the decoder stack with micro-GPT; keep CPU tests small."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dessin.llm.gpt2_nano_trainer import train_gpt2_nano


def test_gpt2_nano_reduces_loss_small_budget():
    r = train_gpt2_nano(
        data_seed=7,
        training_steps=40,
        batch_size=6,
        learning_rate=0.004,
        device="cpu",
        n_layer=2,
        n_head=2,
        n_embd=48,
        block_size=32,
    )
    assert r.model_architecture_json is not None
    arch = r.model_architecture_json
    assert arch.get("variant") == "gpt2_nano"
    assert arch.get("schema_version") == "1.0"
    assert arch["hyperparameters"]["n_embd"] == 48
    assert r.loss_after <= r.loss_before + 0.08
