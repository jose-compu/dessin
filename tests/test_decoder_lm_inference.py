"""Round-trip: train micro-GPT → torrent JSON → load → greedy generate."""

import tempfile

import pytest

pytest.importorskip("torch")

from dessin.llm.decoder_lm_inference import greedy_generate_char_lm, load_decoder_lm_from_torrent_json
from dessin.llm.micro_gpt_trainer import train_micro_gpt
from dessin.llm.torrent_model_trainer import TorrentModelTrainer


def test_load_and_greedy_generate_from_publish_json():
    with tempfile.TemporaryDirectory() as tmp:
        r = train_micro_gpt(
            data_seed=3,
            training_steps=1,
            batch_size=2,
            learning_rate=0.001,
        )
        t = TorrentModelTrainer(tmp)
        out = t.publish_training_result("micro_gpt_char", r)
        assert out.model_file_path
        model, tok, spec = load_decoder_lm_from_torrent_json(out.model_file_path, device="cpu")
        text = greedy_generate_char_lm(
            model, tok, "From fairest", max_new_tokens=12, device="cpu", spec=spec
        )
        assert len(text) > len("From fairest")
        assert "From fairest" in text
