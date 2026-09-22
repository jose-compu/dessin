#!/usr/bin/env python3
"""
Walkthrough of the shared-VRF spot-check verification pipeline.

Trains a tiny micro-GPT block with a recorded trace, derives 2 challenge step
indices from a synthetic post-publication seed, and runs 4 honest verifiers
+ 1 cheating-miner scenario. Prints the structural / statistical / replay
verdict for each verifier and the final approval ratio against the 0.66
threshold (see ``docs/SPOT_CHECK_VERIFICATION.md``).

Usage::

    python e2e/scripts/spot_check_demo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dessin.llm.decoder_lm_training_trace import (
    make_replay_fn,
    train_decoder_only_char_lm_with_trace,
)
from dessin.models.llm_model_spec import LLMVariant
from dessin.consensus.spot_check_verification import (
    StatisticalThresholds,
    TrainingTrace,
    derive_challenge_step_indices,
    verify_spot_replay,
    verify_statistical,
    verify_structural,
)


HP = dict(n_layer=2, n_head=2, n_embd=32, block_size=16)
NUM_STEPS = 8
NUM_VERIFIERS = 4
K_CHALLENGES = 2
THRESHOLD = 0.66


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def vote(label: str, trace, private, challenges, anchor: str) -> bool:
    s_ok, s_fail = verify_structural(
        trace,
        expected_num_steps=trace.num_steps,
        prev_block_state_hash=anchor,
        claimed_state_N_hash=trace.state_N_hash,
    )
    if not s_ok:
        print(f"  {label}: REJECT (structural) {s_fail}")
        return False
    st_ok, st_fail = verify_statistical(trace, StatisticalThresholds())
    if not st_ok:
        print(f"  {label}: REJECT (statistical) {st_fail}")
        return False
    replay = make_replay_fn(
        variant=LLMVariant.MICRO_GPT_CHAR,
        n_layer=HP["n_layer"], n_head=HP["n_head"],
        n_embd=HP["n_embd"], block_size=HP["block_size"],
        data_seed=trace.data_seed,
        batch_size=trace.batch_size,
        learning_rate=trace.learning_rate,
        device="cpu",
    )
    r_ok, r_fail = verify_spot_replay(
        trace, private, challenge_step_indices=challenges, replay_fn=replay,
    )
    if not r_ok:
        print(f"  {label}: REJECT (replay) {r_fail}")
        return False
    print(f"  {label}: APPROVE")
    return True


def run_scenario(name: str, trace, private, challenges, anchor: str) -> None:
    banner(name)
    yes = sum(
        vote(f"verifier_{i}", trace, private, challenges, anchor)
        for i in range(NUM_VERIFIERS)
    )
    ratio = yes / NUM_VERIFIERS
    decision = "ACCEPTED" if ratio >= THRESHOLD else "REJECTED"
    print(
        f"  → approval={yes}/{NUM_VERIFIERS} ratio={ratio:.2f} threshold={THRESHOLD:.2f} "
        f"decision={decision}"
    )


def main() -> int:
    banner("Train block with trace (micro-GPT, deterministic SGD)")
    art = train_decoder_only_char_lm_with_trace(
        LLMVariant.MICRO_GPT_CHAR,
        model_id="micro_gpt_char",
        data_seed=11,
        training_steps=NUM_STEPS,
        batch_size=4,
        learning_rate=0.05,
        env_prefix="DESSIN_MICRO_GPT_",
        device="cpu",
        **HP,
    )
    print(
        f"  num_steps={art.trace.num_steps} "
        f"loss_before={art.trace.loss_curve[0]:.4f} "
        f"loss_after={art.trace.loss_curve[-1]:.4f}"
    )
    print(f"  state_0_hash={art.trace.state_0_hash[:16]}…")
    print(f"  state_N_hash={art.trace.state_N_hash[:16]}…")

    seed = b"\xaa" * 32
    challenges = derive_challenge_step_indices(seed, num_steps=art.trace.num_steps, k=K_CHALLENGES)
    print(f"  shared-VRF post-publication seed → challenges j,k = {challenges}")

    run_scenario("Scenario A — honest miner", art.trace, art.private, challenges, art.trace.state_0_hash)

    cheat_curve = (1.0,) * (NUM_STEPS + 1)
    cheat_curve = (cheat_curve[0], 1.0, 9999.0) + cheat_curve[3:]
    cheat_trace = TrainingTrace(**{**art.trace.to_dict(), "loss_curve": cheat_curve})
    run_scenario(
        "Scenario B — cheating miner (corrupted loss curve)",
        cheat_trace, art.private, challenges, cheat_trace.state_0_hash,
    )

    target = challenges[0]
    idx = target - 1
    tampered = type(art.private)(
        state_hex_before=art.private.state_hex_before,
        grad_hex=tuple(
            ("00" * (len(g) // 2)) if i == idx else g
            for i, g in enumerate(art.private.grad_hex)
        ),
    )
    run_scenario(
        f"Scenario C — cheating miner (faked gradient on challenged step j={target})",
        art.trace, tampered, challenges, art.trace.state_0_hash,
    )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
