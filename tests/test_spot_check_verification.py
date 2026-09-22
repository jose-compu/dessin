"""Unit tests for the shared-VRF spot-check verification path (no torch needed)."""

from __future__ import annotations

import hashlib

import pytest

from dessin.consensus.spot_check_verification import (
    ChainStrikeState,
    ReplayOutcome,
    StatisticalThresholds,
    TrainingTrace,
    TrainingTracePrivate,
    accumulate_strikes,
    compute_grad_hash_chain,
    derive_challenge_step_indices,
    verify_spot_replay,
    verify_statistical,
    verify_structural,
)


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_synthetic_trace(num_steps: int = 6, model_id: str = "m") -> tuple[TrainingTrace, TrainingTracePrivate, list[bytes]]:
    state_hexes = [(b"S" + bytes([i])).hex() for i in range(num_steps + 1)]
    grad_bytes = [(b"G" + bytes([i + 1])) * 4 for i in range(num_steps)]
    grad_hex = [g.hex() for g in grad_bytes]
    chain = compute_grad_hash_chain(grad_bytes)
    norms = [1.0 + 0.05 * i for i in range(num_steps)]
    losses = [3.0 - 0.3 * i for i in range(num_steps + 1)]
    trace = TrainingTrace(
        model_id=model_id,
        num_steps=num_steps,
        state_0_hash=_h(bytes.fromhex(state_hexes[0])),
        state_N_hash=_h(bytes.fromhex(state_hexes[-1])),
        loss_curve=tuple(losses),
        gradient_hash_chain=tuple(chain),
        gradient_norms=tuple(norms),
        data_seed=42,
        learning_rate=0.01,
        batch_size=8,
    )
    private = TrainingTracePrivate(
        state_hex_before=tuple(state_hexes[:num_steps]),
        grad_hex=tuple(grad_hex),
    )
    post_state_hexes = state_hexes[1:]
    return trace, private, [bytes.fromhex(s) for s in post_state_hexes]


def test_derive_challenge_step_indices_deterministic_and_in_range():
    seed = b"\xaa" * 32
    a = derive_challenge_step_indices(seed, num_steps=100, k=2)
    b = derive_challenge_step_indices(seed, num_steps=100, k=2)
    assert a == b
    assert all(1 <= x <= 100 for x in a)


def test_derive_challenge_step_indices_different_seeds_diverge():
    a = derive_challenge_step_indices(b"\x01" * 32, num_steps=100, k=2)
    b = derive_challenge_step_indices(b"\x02" * 32, num_steps=100, k=2)
    assert a != b


def test_structural_passes_on_well_formed_trace():
    trace, _, _ = _make_synthetic_trace(num_steps=5)
    ok, fails = verify_structural(
        trace,
        expected_num_steps=5,
        prev_block_state_hash=trace.state_0_hash,
        claimed_state_N_hash=trace.state_N_hash,
    )
    assert ok, fails


def test_structural_fails_on_anchor_mismatch():
    trace, _, _ = _make_synthetic_trace(num_steps=5)
    ok, fails = verify_structural(
        trace,
        expected_num_steps=5,
        prev_block_state_hash="0" * 64,
        claimed_state_N_hash=trace.state_N_hash,
    )
    assert not ok
    assert any("does not anchor" in f for f in fails)


def test_structural_fails_on_step_count_mismatch():
    trace, _, _ = _make_synthetic_trace(num_steps=5)
    ok, fails = verify_structural(
        trace,
        expected_num_steps=6,
        prev_block_state_hash=trace.state_0_hash,
        claimed_state_N_hash=trace.state_N_hash,
    )
    assert not ok
    assert any("num_steps mismatch" in f for f in fails)


def test_statistical_passes_clean_curve():
    trace, _, _ = _make_synthetic_trace(num_steps=8)
    ok, fails = verify_statistical(trace)
    assert ok, fails


def test_statistical_flags_nonfinite_loss():
    trace, _, _ = _make_synthetic_trace(num_steps=5)
    bad = TrainingTrace(
        **{**trace.to_dict(), "loss_curve": (3.0, float("nan"), 2.0, 1.5, 1.2, 1.0)}  # type: ignore[arg-type]
    )
    ok, fails = verify_statistical(bad)
    assert not ok
    assert any("not finite" in f for f in fails)


def test_statistical_flags_huge_loss_jump():
    trace, _, _ = _make_synthetic_trace(num_steps=5)
    losses = (1.0, 1.0, 5000.0, 1.0, 1.0, 1.0)
    bad = TrainingTrace(**{**trace.to_dict(), "loss_curve": losses})  # type: ignore[arg-type]
    ok, fails = verify_statistical(bad, StatisticalThresholds(max_loss_jump_ratio=10.0))
    assert not ok
    assert any("loss jump" in f for f in fails)


def test_statistical_flags_grad_norm_out_of_band():
    trace, _, _ = _make_synthetic_trace(num_steps=5)
    norms = (1.0, 1.0, 1.0, 1.0, 1e12)
    bad = TrainingTrace(**{**trace.to_dict(), "gradient_norms": norms})  # type: ignore[arg-type]
    ok, fails = verify_statistical(bad, StatisticalThresholds(max_grad_norm=1e6))
    assert not ok
    assert any("above max" in f for f in fails)


def test_compute_grad_hash_chain_well_formed():
    grads = [b"a", b"bb", b"ccc"]
    chain = compute_grad_hash_chain(grads)
    assert len(chain) == 3
    assert all(len(h) == 64 for h in chain)
    assert len(set(chain)) == 3


def test_spot_replay_passes_when_replay_matches_witness():
    trace, private, post_states = _make_synthetic_trace(num_steps=4)

    def replay_fn(j: int, pre_state_hex: str) -> ReplayOutcome:
        idx = j - 1
        gbytes = bytes.fromhex(private.grad_hex[idx])
        post = post_states[idx]
        norm = trace.gradient_norms[idx]
        return ReplayOutcome(grad_bytes=gbytes, grad_norm=norm, state_hex_after=post.hex())

    ok, fails = verify_spot_replay(
        trace,
        private,
        challenge_step_indices=[1, 3],
        replay_fn=replay_fn,
    )
    assert ok, fails


def test_spot_replay_fails_when_post_state_tampered():
    trace, private, post_states = _make_synthetic_trace(num_steps=4)

    def replay_fn(j: int, pre_state_hex: str) -> ReplayOutcome:
        idx = j - 1
        gbytes = bytes.fromhex(private.grad_hex[idx])
        if j == 2:
            tampered = b"X" * len(post_states[idx])
            return ReplayOutcome(
                grad_bytes=gbytes,
                grad_norm=trace.gradient_norms[idx],
                state_hex_after=tampered.hex(),
            )
        return ReplayOutcome(
            grad_bytes=gbytes,
            grad_norm=trace.gradient_norms[idx],
            state_hex_after=post_states[idx].hex(),
        )

    ok, fails = verify_spot_replay(
        trace,
        private,
        challenge_step_indices=[2, 4],
        replay_fn=replay_fn,
    )
    assert not ok
    assert any("post-state hash" in f for f in fails)


def test_spot_replay_fails_when_gradient_bytes_tampered():
    trace, private, post_states = _make_synthetic_trace(num_steps=3)

    def replay_fn(j: int, pre_state_hex: str) -> ReplayOutcome:
        idx = j - 1
        if j == 1:
            return ReplayOutcome(
                grad_bytes=b"BOGUSBOGUSBOGUS!",
                grad_norm=trace.gradient_norms[idx],
                state_hex_after=post_states[idx].hex(),
            )
        gbytes = bytes.fromhex(private.grad_hex[idx])
        return ReplayOutcome(
            grad_bytes=gbytes,
            grad_norm=trace.gradient_norms[idx],
            state_hex_after=post_states[idx].hex(),
        )

    ok, fails = verify_spot_replay(
        trace,
        private,
        challenge_step_indices=[1],
        replay_fn=replay_fn,
    )
    assert not ok
    assert any("gradient hash" in f for f in fails)


def test_spot_replay_rejects_witness_with_wrong_length():
    trace, private, _ = _make_synthetic_trace(num_steps=3)
    short = TrainingTracePrivate(
        state_hex_before=tuple(private.state_hex_before[:1]),
        grad_hex=tuple(private.grad_hex[:1]),
    )
    ok, fails = verify_spot_replay(
        trace,
        short,
        challenge_step_indices=[1],
        replay_fn=lambda j, s: ReplayOutcome(b"", 0.0, ""),
    )
    assert not ok
    assert any("witness lengths mismatch" in f for f in fails)


def test_spot_replay_reports_out_of_range_index():
    trace, private, _ = _make_synthetic_trace(num_steps=3)
    ok, fails = verify_spot_replay(
        trace,
        private,
        challenge_step_indices=[0, 99],
        replay_fn=lambda j, s: ReplayOutcome(b"", 0.0, ""),
    )
    assert not ok
    assert any("out of range" in f for f in fails)


def test_strike_resets_when_curve_dips_below_l0():
    state = ChainStrikeState()
    trace, _, _ = _make_synthetic_trace(num_steps=4)
    n, frozen = accumulate_strikes(state, model_id="m", trace=trace, max_strikes=3)
    assert n == 0
    assert not frozen
    assert not state.is_frozen("m")


def test_strike_accumulates_and_freezes_on_three_unproductive_blocks():
    state = ChainStrikeState()
    flat_losses = (5.0, 5.1, 5.2, 5.05, 5.3)
    template, _, _ = _make_synthetic_trace(num_steps=4)
    bad_trace = TrainingTrace(**{**template.to_dict(), "loss_curve": flat_losses})  # type: ignore[arg-type]

    n1, f1 = accumulate_strikes(state, model_id="m", trace=bad_trace, max_strikes=3)
    n2, f2 = accumulate_strikes(state, model_id="m", trace=bad_trace, max_strikes=3)
    n3, f3 = accumulate_strikes(state, model_id="m", trace=bad_trace, max_strikes=3)
    assert (n1, n2, n3) == (1, 2, 3)
    assert (f1, f2) == (False, False)
    assert f3 is True
    assert state.is_frozen("m")


def test_strike_reset_on_refresh_clears_freeze():
    state = ChainStrikeState()
    flat_losses = (4.0, 4.1, 4.05, 4.2)
    template, _, _ = _make_synthetic_trace(num_steps=3)
    bad = TrainingTrace(**{**template.to_dict(), "loss_curve": flat_losses})  # type: ignore[arg-type]
    for _ in range(3):
        accumulate_strikes(state, model_id="m", trace=bad, max_strikes=3)
    assert state.is_frozen("m")
    state.reset("m")
    assert not state.is_frozen("m")
    assert state.strikes_by_model["m"] == 0
