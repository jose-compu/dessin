"""
End-to-end pipeline test for the shared-VRF spot-check verification path.

Exercises (no mesh, no chaincraft) the full miner / verifier flow on a real
micro-GPT trace:

1. **Honest miner** trains with trace → 4 honest verifiers each independently
   run structural + statistical + replay-2-challenge-steps on the same shared
   post-publication seed. Approval ratio must clear the 0.66 threshold.

2. **Cheating miner — corrupted loss curve** → every honest verifier rejects on
   the cheap statistical layer (no replay needed). Approval ratio is 0.0.

3. **Cheating miner — tampered gradient on a challenged step** → every honest
   verifier rejects via spot replay. Approval ratio is 0.0.

4. **Strike accumulation** → 3 consecutive blocks whose loss curve never dips
   below ``L_0`` freezes the model.

Skipped under ``DESSIN_SKIP_E2E=1``. See ``docs/SPOT_CHECK_VERIFICATION.md``.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from dessin.llm.decoder_lm_training_trace import (
    make_replay_fn,
    train_decoder_only_char_lm_with_trace,
)
from dessin.models.llm_model_spec import LLMVariant
from dessin.consensus.spot_check_verification import (
    ChainStrikeState,
    StatisticalThresholds,
    TrainingTrace,
    accumulate_strikes,
    derive_challenge_step_indices,
    verify_spot_replay,
    verify_statistical,
    verify_structural,
)


_HP = dict(n_layer=2, n_head=2, n_embd=32, block_size=16)
_NUM_STEPS = 6
_K_CHALLENGES = 2
_NUM_VERIFIERS = 4
_THRESHOLD = 0.66


def _train_block(steps: int = _NUM_STEPS, *, data_seed: int = 11):
    return train_decoder_only_char_lm_with_trace(
        LLMVariant.MICRO_GPT_CHAR,
        model_id="micro_gpt_char",
        data_seed=data_seed,
        training_steps=steps,
        batch_size=4,
        learning_rate=0.05,
        env_prefix="DESSIN_MICRO_GPT_",
        device="cpu",
        **_HP,
    )


def _build_replay_fn(trace: TrainingTrace):
    return make_replay_fn(
        variant=LLMVariant.MICRO_GPT_CHAR,
        n_layer=_HP["n_layer"], n_head=_HP["n_head"],
        n_embd=_HP["n_embd"], block_size=_HP["block_size"],
        data_seed=trace.data_seed,
        batch_size=trace.batch_size,
        learning_rate=trace.learning_rate,
        device="cpu",
    )


def _verifier_vote(trace, private, challenges, *, expected_anchor: str) -> bool:
    """Single-verifier yes/no on a (trace, private witness, challenge step indices)."""
    s_ok, _ = verify_structural(
        trace,
        expected_num_steps=trace.num_steps,
        prev_block_state_hash=expected_anchor,
        claimed_state_N_hash=trace.state_N_hash,
    )
    if not s_ok:
        return False
    st_ok, _ = verify_statistical(trace, StatisticalThresholds())
    if not st_ok:
        return False
    replay = _build_replay_fn(trace)
    r_ok, _ = verify_spot_replay(
        trace, private, challenge_step_indices=challenges, replay_fn=replay,
    )
    return r_ok


pytestmark = pytest.mark.skipif(
    os.environ.get("DESSIN_SKIP_E2E", "").strip().lower() in ("1", "true", "yes"),
    reason="DESSIN_SKIP_E2E set",
)


@pytest.mark.e2e
@pytest.mark.integration
def test_spot_check_pipeline_honest_block_passes_threshold():
    art = _train_block()
    challenges = derive_challenge_step_indices(
        b"\xaa" * 32, num_steps=art.trace.num_steps, k=_K_CHALLENGES,
    )
    yes = sum(
        _verifier_vote(art.trace, art.private, challenges, expected_anchor=art.trace.state_0_hash)
        for _ in range(_NUM_VERIFIERS)
    )
    ratio = yes / _NUM_VERIFIERS
    assert ratio >= _THRESHOLD, f"approval ratio {ratio:.2f} below threshold {_THRESHOLD}"


@pytest.mark.e2e
@pytest.mark.integration
def test_spot_check_pipeline_cheater_corrupted_loss_curve_rejected_statistically():
    art = _train_block()
    bogus = (1.0, 1.0, 9999.0, 1.0, 1.0, 1.0, 1.0)
    cheat = TrainingTrace(**{**art.trace.to_dict(), "loss_curve": bogus})
    challenges = derive_challenge_step_indices(
        b"\xaa" * 32, num_steps=cheat.num_steps, k=_K_CHALLENGES,
    )
    yes = sum(
        _verifier_vote(cheat, art.private, challenges, expected_anchor=cheat.state_0_hash)
        for _ in range(_NUM_VERIFIERS)
    )
    assert yes == 0


@pytest.mark.e2e
@pytest.mark.integration
def test_spot_check_pipeline_cheater_tampered_step_rejected_by_replay():
    art = _train_block()
    seed = b"\xbb" * 32
    challenges = derive_challenge_step_indices(
        seed, num_steps=art.trace.num_steps, k=_K_CHALLENGES,
    )
    target = challenges[0]
    idx = target - 1
    tampered_pre_state = art.private.state_hex_before[idx]
    tampered = type(art.private)(
        state_hex_before=tuple(
            tampered_pre_state if i == idx else h
            for i, h in enumerate(art.private.state_hex_before)
        ),
        grad_hex=tuple(
            ("00" * (len(g) // 2)) if i == idx else g
            for i, g in enumerate(art.private.grad_hex)
        ),
    )
    yes = sum(
        _verifier_vote(art.trace, tampered, challenges, expected_anchor=art.trace.state_0_hash)
        for _ in range(_NUM_VERIFIERS)
    )
    assert yes == 0


@pytest.mark.e2e
@pytest.mark.integration
def test_spot_check_pipeline_three_unproductive_blocks_freeze_model():
    state = ChainStrikeState()
    template = _train_block().trace
    flat_losses = (5.0, 5.1, 5.2, 5.05, 5.3, 5.1, 5.4)
    bad = TrainingTrace(**{**template.to_dict(), "loss_curve": flat_losses})

    n1, f1 = accumulate_strikes(state, model_id="micro_gpt_char", trace=bad, max_strikes=3)
    n2, f2 = accumulate_strikes(state, model_id="micro_gpt_char", trace=bad, max_strikes=3)
    n3, f3 = accumulate_strikes(state, model_id="micro_gpt_char", trace=bad, max_strikes=3)
    assert (n1, n2, n3) == (1, 2, 3)
    assert (f1, f2, f3) == (False, False, True)
    assert state.is_frozen("micro_gpt_char")

    state.reset("micro_gpt_char")
    assert not state.is_frozen("micro_gpt_char")
