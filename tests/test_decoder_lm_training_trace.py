"""End-to-end tests for the trace-recording trainer + replay verifier.

Uses a tiny micro-GPT preset and a small step count so this is fast on CPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dessin.llm.decoder_lm_training_trace import (
    make_replay_fn,
    train_decoder_only_char_lm_with_trace,
)
from dessin.models.llm_model_spec import LLMVariant
from dessin.consensus.spot_check_verification import (
    derive_challenge_step_indices,
    verify_spot_replay,
    verify_statistical,
    verify_structural,
)


_HP = dict(n_layer=2, n_head=2, n_embd=32, block_size=16)


def _train_small(steps: int = 8):
    return train_decoder_only_char_lm_with_trace(
        LLMVariant.MICRO_GPT_CHAR,
        model_id="micro_gpt_char",
        data_seed=11,
        training_steps=steps,
        batch_size=4,
        learning_rate=0.05,
        env_prefix="DESSIN_MICRO_GPT_",
        device="cpu",
        **_HP,
    )


def test_traced_trainer_emits_consistent_trace():
    art = _train_small(steps=6)
    trace = art.trace
    assert trace.num_steps == 6
    assert len(trace.loss_curve) == 7
    assert len(trace.gradient_hash_chain) == 6
    assert len(trace.gradient_norms) == 6
    assert all(len(h) == 64 for h in trace.gradient_hash_chain)


def test_traced_trainer_replay_matches_recorded_witness():
    art = _train_small(steps=6)
    replay = make_replay_fn(
        variant=LLMVariant.MICRO_GPT_CHAR,
        n_layer=_HP["n_layer"], n_head=_HP["n_head"],
        n_embd=_HP["n_embd"], block_size=_HP["block_size"],
        data_seed=art.trace.data_seed,
        batch_size=art.trace.batch_size,
        learning_rate=art.trace.learning_rate,
        device="cpu",
    )

    ok, fails = verify_structural(
        art.trace,
        expected_num_steps=art.trace.num_steps,
        prev_block_state_hash=art.trace.state_0_hash,
        claimed_state_N_hash=art.trace.state_N_hash,
    )
    assert ok, fails

    ok2, fails2 = verify_statistical(art.trace)
    assert ok2, fails2

    seed = b"\x33" * 32
    challenges = derive_challenge_step_indices(seed, num_steps=art.trace.num_steps, k=2)
    ok3, fails3 = verify_spot_replay(
        art.trace,
        art.private,
        challenge_step_indices=challenges,
        replay_fn=replay,
    )
    assert ok3, fails3


def test_traced_trainer_detects_tampered_loss_curve_via_replay():
    art = _train_small(steps=4)
    replay = make_replay_fn(
        variant=LLMVariant.MICRO_GPT_CHAR,
        n_layer=_HP["n_layer"], n_head=_HP["n_head"],
        n_embd=_HP["n_embd"], block_size=_HP["block_size"],
        data_seed=art.trace.data_seed,
        batch_size=art.trace.batch_size,
        learning_rate=art.trace.learning_rate,
        device="cpu",
    )

    tampered_witness = type(art.private)(
        state_hex_before=tuple(
            (s[:-2] + ("aa" if i == 1 else s[-2:])) for i, s in enumerate(art.private.state_hex_before)
        ),
        grad_hex=art.private.grad_hex,
    )
    seed = b"\x55" * 32
    challenges = derive_challenge_step_indices(seed, num_steps=art.trace.num_steps, k=2)
    if 2 not in challenges:
        challenges = [2, *challenges][: max(2, len(challenges))]
    ok, fails = verify_spot_replay(
        art.trace,
        tampered_witness,
        challenge_step_indices=[2],
        replay_fn=replay,
    )
    assert not ok
    assert fails
