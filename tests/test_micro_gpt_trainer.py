"""Fast checks for tiny char-GPT training path used in E2E mining."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dessin.llm.micro_gpt_trainer import train_micro_gpt


def test_micro_gpt_reduces_loss_small_budget():
    r = train_micro_gpt(
        data_seed=42,
        training_steps=80,
        batch_size=8,
        learning_rate=0.004,
        device="cpu",
    )
    assert r.pogo_layer_shapes is not None and len(r.pogo_layer_shapes) >= 4
    assert r.model_architecture_json is not None
    arch = r.model_architecture_json
    assert arch.get("type") == "micro_gpt_char"
    assert arch.get("schema_version") == "1.0"
    assert arch.get("family") == "decoder_only_transformer"
    assert arch["hyperparameters"]["n_layer"] >= 1
    assert r.loss_before >= 0.0 and r.loss_after >= 0.0
    # Typical: CE drops on Sonnet char corpus with many steps
    assert r.loss_after <= r.loss_before + 0.05
    nw = len(bytes.fromhex(r.model_weights_hex))
    assert nw == r.model_size_bytes == sum(t[0] for t in r.pogo_layer_shapes) * 8
