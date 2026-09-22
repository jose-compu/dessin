"""
Live four-node DeSSIN battery: real Chaincraft peers + PoGO consensus checks.

Skipped when DESSIN_SKIP_E2E=1. Duration and thresholds tune via env (see docs/E2E_NETWORK_TESTS.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

from dessin.runtime.node import DessinNode
from dessin.runtime.config import DessinConfig
from dessin.runtime.pretty_console import pretty_print

# This module is the micro-GPT / decoder-LM four-node battery only (never legacy MLP).
os.environ.setdefault("DESSIN_MICRO_GPT_TRAINING", "1")

# Soak defaults: ``bootstrap_period_blocks`` is 10 in the battery — target **20** new blocks
# (≈10 heights while still in bootstrap, then ≈10 after) at 1-minute mining slots.
E2E_BATTERY_DEFAULT_DURATION_SEC = 1500
E2E_BATTERY_DEFAULT_MIN_NEW_BLOCKS = 20

_e2e_clock_start: float | None = None
_e2e_budget_deadline: float | None = None


def _e2e_quiet() -> bool:
    return os.environ.get("DESSIN_E2E_QUIET", "").strip().lower() in ("1", "true", "yes")


def _e2e_timing_reset() -> None:
    global _e2e_clock_start, _e2e_budget_deadline
    _e2e_clock_start = None
    _e2e_budget_deadline = None


def _e2e_timing_start(
    *,
    warmup_sec: float,
    duration_sec: float,
    converge_sec: float,
    slack_sec: float = 240.0,
) -> None:
    """Budget ≈ warmup + soak + converge + slack (covers mesh/mining variance). Idempotent."""
    global _e2e_clock_start, _e2e_budget_deadline
    if _e2e_clock_start is not None:
        return
    now = time.monotonic()
    _e2e_clock_start = now
    _e2e_budget_deadline = now + float(warmup_sec) + float(duration_sec) + float(converge_sec) + float(slack_sec)


def _e2e_timing_extend_at_least(*, remaining_sec: float) -> None:
    """Ensure remaining estimate covers at least remaining_sec from now (e.g. converge wait)."""
    global _e2e_budget_deadline
    if _e2e_budget_deadline is None:
        return
    floor_end = time.monotonic() + float(remaining_sec)
    if floor_end > _e2e_budget_deadline:
        _e2e_budget_deadline = floor_end


def _e2e_timing_suffix() -> str:
    now = time.monotonic()
    parts: list[str] = []
    if _e2e_clock_start is not None:
        parts.append(f"⏱ elapsed={now - _e2e_clock_start:.1f}s")
    if _e2e_budget_deadline is not None:
        parts.append(f"⌛ remaining≈{max(0.0, _e2e_budget_deadline - now):.1f}s")
    return (" │ " + " │ ".join(parts)) if parts else ""


def _e2e_log(msg: str, *, kind: str = "info") -> None:
    if _e2e_quiet():
        return
    pretty_print(msg, kind=kind, tag="[e2e]", suffix=_e2e_timing_suffix())


def _e2e_mining_model_id() -> str:
    """Catalog / block ``model_id`` aligned with ``DESSIN_LLM_VARIANT`` (no legacy MLP id)."""
    v = os.environ.get("DESSIN_LLM_VARIANT", "micro_gpt_char").strip().lower()
    if v in ("gpt2_nano", "gpt2", "nanogpt", "gpt2-nano"):
        return "gpt2_nano"
    return "micro_gpt_char"


def _prepare_e2e_llm_env_variant(variant: str) -> None:
    """Enable decoder LM mining for e2e; ``variant`` is ``micro_gpt_char`` or ``gpt2_nano`` (see DESSIN_LLM_VARIANT)."""
    os.environ.setdefault("DESSIN_MICRO_GPT_TRAINING", "1")
    os.environ["DESSIN_LLM_VARIANT"] = variant
    os.environ.setdefault("DESSIN_MICRO_GPT_DEVICE", "cpu")
    if variant in ("gpt2_nano", "gpt2", "nanogpt", "gpt2-nano"):
        os.environ.setdefault("DESSIN_MICRO_GPT_STEPS", "48")
        os.environ.setdefault("DESSIN_MICRO_GPT_BATCH", "6")
        os.environ.setdefault("DESSIN_MICRO_GPT_LR", "0.003")
    else:
        os.environ.setdefault("DESSIN_MICRO_GPT_STEPS", "64")
        os.environ.setdefault("DESSIN_MICRO_GPT_BATCH", "10")
        os.environ.setdefault("DESSIN_MICRO_GPT_LR", "0.003")


def _find_free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            sockets.append(s)
            ports.append(s.getsockname()[1])
    finally:
        for s in sockets:
            s.close()
    return ports


def _mesh_connect(nodes: list[DessinNode]) -> None:
    """Full mesh outbound edges (UDP: local peer list only). Two passes with settle."""
    for _attempt in range(2):
        for i, node in enumerate(nodes):
            for j, other in enumerate(nodes):
                if i == j:
                    continue
                try:
                    port = int(other.chaincraft_node.port)
                    node.chaincraft_node.connect_to_peer("127.0.0.1", port)
                except Exception as exc:
                    if not _e2e_quiet():
                        _e2e_log(
                            f"mesh connect node{i}->node{j} (target_port={getattr(other.chaincraft_node, 'port', '?')}): {exc!r}",
                            kind="warn",
                        )
                time.sleep(0.06)
        time.sleep(0.22 * (_attempt + 1))


def _mesh_repair_starved(nodes: list[DessinNode], *, min_peers_each: int) -> None:
    """Repeat outbound connects from nodes still below the peer budget."""
    counts = _peer_counts(nodes)
    for idx, c in enumerate(counts):
        if c >= min_peers_each:
            continue
        for j, other in enumerate(nodes):
            if idx == j:
                continue
            try:
                port = int(other.chaincraft_node.port)
                nodes[idx].chaincraft_node.connect_to_peer("127.0.0.1", port)
            except Exception as exc:
                if not _e2e_quiet():
                    _e2e_log(
                        f"repair node{idx}->{j} (target_port={getattr(other.chaincraft_node, 'port', '?')}): {exc!r}",
                        kind="warn",
                    )
            time.sleep(0.05)


def _wait_mesh_peers(
    nodes: list[DessinNode],
    *,
    min_peers_each: int,
    rounds: int,
    pause_sec: float,
) -> list[int]:
    """Repeat outbound connects until every node reports >= min_peers_each or rounds exhausted."""
    counts = _peer_counts(nodes)
    for r in range(rounds):
        if counts and min(counts) >= min_peers_each:
            _e2e_log(
                f"mesh stable round={r}/{rounds} peers_each={counts} (target min_each={min_peers_each})",
                kind="ok",
            )
            break
        _e2e_log(f"mesh round={r}/{rounds} peers_each={counts} -> reconnect…", kind="mesh")
        _mesh_connect(nodes)
        _mesh_repair_starved(nodes, min_peers_each=min_peers_each)
        time.sleep(pause_sec)
        counts = _peer_counts(nodes)
    return counts


def _peer_counts(nodes: list[DessinNode]) -> list[int]:
    out = []
    for n in nodes:
        peers = getattr(n.chaincraft_node, "peers", None)
        out.append(len(peers) if peers is not None else 0)
    return out


def _chain_snapshots(nodes: list[DessinNode]) -> tuple[list[int], list[str]]:
    heights = []
    tips = []
    for n in nodes:
        info = n.get_chain_info()
        heights.append(int(info["chain_length"]))
        tips.append(str(info["latest_block"]["hash"]))
    return heights, tips


def _e2e_required_new_blocks_cap(
    duration_sec: float,
    min_new_blocks: int,
    nodes: list[DessinNode],
    *,
    headroom: float = 0.82,
) -> int:
    """Cap ``min_new_blocks`` by how many heights soak time can close at configured cadence.

    One shared chain advances ~one block per slowest mining/training slot among active miners;
    requiring e.g. ``baseline + 30`` with a short soak and 1-minute slots is impossible.
    """
    slot_min = 1.0
    for n in nodes:
        if float(n.config.consensus.block_time_minutes) <= 0.0:
            continue
        bt = float(n.config.consensus.block_time_minutes)
        tt = float(getattr(n.config.consensus, "training_block_time_minutes", bt))
        slot_min = max(slot_min, max(bt, tt))
    slot_sec = max(15.0, slot_min * 60.0)
    approx = int((float(duration_sec) * headroom) / slot_sec)
    feasible = max(4, approx)
    return max(1, min(int(min_new_blocks), feasible))


def _leader_node_label(nodes: list[DessinNode], miner_address: str) -> str:
    for i, n in enumerate(nodes):
        if n.address == miner_address:
            return f"node_{i}"
    return "unknown"


def _canonical_chain(nodes: list[DessinNode]) -> tuple[int, list]:
    """Longest local chain wins (tie-break lower node index)."""
    best_i = max(range(len(nodes)), key=lambda i: len(nodes[i].consensus.chain))
    return best_i, nodes[best_i].consensus.chain


def _e2e_spot_check_phase(nodes: list[DessinNode]) -> None:
    """
    Post-convergence: exercise the shared-VRF spot-check verification path
    (`dessin.consensus.spot_check_verification` + `dessin.llm.decoder_lm_training_trace`) on top of
    the running four-node mesh. Each running node casts one independent verifier
    vote (structural + statistical + 2-step replay on a shared VRF-derived seed).

    Asserts:
      * honest miner crosses ``spot_check_attestation_threshold`` (default 0.66),
      * a corrupted-loss-curve cheater is rejected by every voter (cheap statistical
        layer, no replay needed).

    Independent of the live PoGO chain (the trace pipeline is gated by
    ``enable_spot_check_verification`` and not yet bound to ``PogoBlock`` fields);
    this demonstrates the v2 verification path runs cleanly alongside live mining.

    Skip with ``DESSIN_E2E_SKIP_SPOT_CHECK=1``. Tune training cost with
    ``DESSIN_E2E_SPOT_CHECK_STEPS`` (default 6). See ``docs/SPOT_CHECK_VERIFICATION.md``.
    """
    if os.environ.get("DESSIN_E2E_SKIP_SPOT_CHECK", "").strip().lower() in ("1", "true", "yes"):
        _e2e_log("spot-check phase skipped (DESSIN_E2E_SKIP_SPOT_CHECK=1)", kind="info")
        return

    try:
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
    except Exception as exc:
        _e2e_log(f"spot-check phase: import failed ({exc!r}) — skipping", kind="warn")
        return

    cc = nodes[0].config.consensus
    threshold = float(cc.spot_check_attestation_threshold)
    k = int(cc.spot_check_count)
    train_steps = int(os.environ.get("DESSIN_E2E_SPOT_CHECK_STEPS", "6"))
    voters = list(range(len(nodes)))
    seed_honest = bytes.fromhex("a5" * 32)

    _e2e_log(
        f"spot-check phase: train_steps={train_steps} k={k} threshold={threshold:.2f} "
        f"voters={len(voters)} (1 vote per running node) — variant=micro_gpt_char (trace-trainer)",
        kind="setup",
    )

    art = train_decoder_only_char_lm_with_trace(
        LLMVariant.MICRO_GPT_CHAR,
        model_id=_e2e_mining_model_id(),
        data_seed=11,
        training_steps=train_steps,
        batch_size=4,
        learning_rate=0.05,
        env_prefix="DESSIN_MICRO_GPT_",
        device="cpu",
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
    )
    challenges = derive_challenge_step_indices(
        seed_honest, num_steps=art.trace.num_steps, k=k
    )
    _e2e_log(
        f"spot-check phase: challenge step indices={list(challenges)} "
        f"state0={art.trace.state_0_hash[:12]}… stateN={art.trace.state_N_hash[:12]}…",
        kind="info",
    )
    replay_fn = make_replay_fn(
        variant=LLMVariant.MICRO_GPT_CHAR,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
        data_seed=art.trace.data_seed,
        batch_size=art.trace.batch_size,
        learning_rate=art.trace.learning_rate,
        device="cpu",
    )

    def _vote(trace: TrainingTrace, private) -> bool:
        s_ok, s_fail = verify_structural(
            trace,
            expected_num_steps=trace.num_steps,
            prev_block_state_hash=trace.state_0_hash,
            claimed_state_N_hash=trace.state_N_hash,
        )
        if not s_ok:
            _e2e_log(f"spot-check vote: structural fail {s_fail!r}", kind="warn")
            return False
        st_ok, st_fail = verify_statistical(trace, StatisticalThresholds())
        if not st_ok:
            _e2e_log(f"spot-check vote: statistical fail {st_fail!r}", kind="warn")
            return False
        r_ok, r_fail = verify_spot_replay(
            trace,
            private,
            challenge_step_indices=challenges,
            replay_fn=replay_fn,
        )
        if not r_ok:
            _e2e_log(f"spot-check vote: replay fail {r_fail!r}", kind="warn")
        return r_ok

    yes_honest = sum(_vote(art.trace, art.private) for _ in voters)
    ratio_honest = yes_honest / len(voters)
    _e2e_log(
        f"spot-check honest: approvals={yes_honest}/{len(voters)} "
        f"ratio={ratio_honest:.2f} threshold={threshold:.2f}",
        kind="ok" if ratio_honest >= threshold else "bad",
    )
    assert ratio_honest >= threshold, (
        f"spot-check honest approval ratio {ratio_honest:.2f} below threshold {threshold:.2f}"
    )

    bogus_curve = list(art.trace.loss_curve)
    bogus_curve[max(1, len(bogus_curve) // 2)] = 9999.0
    cheat = TrainingTrace(**{**art.trace.to_dict(), "loss_curve": tuple(bogus_curve)})
    yes_cheat = sum(_vote(cheat, art.private) for _ in voters)
    _e2e_log(
        f"spot-check cheater (corrupted loss curve): approvals={yes_cheat}/{len(voters)} "
        "(expected 0; statistical layer should reject)",
        kind="ok" if yes_cheat == 0 else "bad",
    )
    assert yes_cheat == 0, (
        f"spot-check cheater approvals={yes_cheat}/{len(voters)} (expected 0)"
    )
    _e2e_log("spot-check phase: all assertions passed", kind="ok")


def _e2e_shakespeare_inference_from_battery(nodes: list[DessinNode], tmp_root: Path) -> None:
    """
    Greedy char-LM continuation using the **latest mined** checkpoint under this run's ``tmp_root``.

    Picks the JSON whose contents include the canonical tip block's ``torrent_hash`` when present,
    else the newest ``{model_id}_*.json`` by mtime (isolated temp caches → no stale global models).
    """
    if os.environ.get("DESSIN_E2E_SKIP_INFERENCE", "").strip().lower() in ("1", "true", "yes"):
        _e2e_log("inference skipped (DESSIN_E2E_SKIP_INFERENCE=1)", kind="info")
        return
    if _e2e_quiet():
        return
    try:
        from dessin.llm.decoder_lm_inference import greedy_generate_char_lm, load_decoder_lm_from_torrent_json
    except Exception as exc:
        _e2e_log(f"inference import failed: {exc!r}", kind="warn")
        return

    _, chain = _canonical_chain(nodes)
    mid = _e2e_mining_model_id()
    tip = None
    for b in reversed(chain):
        if getattr(b, "model_id", None) == mid and int(getattr(b, "model_size_bytes", 0) or 0) > 0:
            tip = b
            break
    if tip is None:
        _e2e_log("inference skipped: no LM block with model_size_bytes on canonical chain", kind="warn")
        return
    th = str(getattr(tip, "torrent_hash", None) or "")
    candidates = sorted(
        tmp_root.glob(f"cache_*/torrents/{mid}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    chosen: Path | None = None
    if th:
        for p in candidates:
            try:
                if th in p.read_text(encoding="utf-8"):
                    chosen = p
                    break
            except OSError:
                continue
    if chosen is None and candidates:
        chosen = candidates[0]
    if chosen is None:
        _e2e_log(f"inference skipped: no {mid}_*.json under {tmp_root}", kind="warn")
        return

    try:
        with chosen.open("r", encoding="utf-8") as f:
            blob = json.load(f)
        wh = (blob.get("model_weights") or {}).get("weights_hex", "")
        exp = (blob.get("model_weights") or {}).get("checksum", "")
        if wh and exp and hashlib.sha256(wh.encode()).hexdigest() != exp:
            _e2e_log("inference skipped: JSON checksum mismatch", kind="warn")
            return
    except Exception as exc:
        _e2e_log(f"inference skipped: cannot validate JSON ({exc!r})", kind="warn")
        return

    dev = os.environ.get("DESSIN_MICRO_GPT_DEVICE", "cpu")
    max_new = int(os.environ.get("DESSIN_E2E_INFER_MAX_NEW", "96"))
    try:
        model, tok, spec = load_decoder_lm_from_torrent_json(chosen, device=dev)
        prompt = "From fairest creatures"
        text = greedy_generate_char_lm(
            model, tok, prompt, max_new_tokens=max_new, device=dev, spec=spec
        )
    except Exception as exc:
        _e2e_log(f"inference failed: {exc!r}", kind="warn")
        return

    _e2e_log(
        f"inference ok file={chosen.name} block={tip.index} torrent={th[:16]}… "
        f"greedy_len={len(text)} prefix={text[:140]!r}",
        kind="ok",
    )


def _e2e_sync_canonical_model_owner(nodes: list[DessinNode]) -> None:
    """Align consensus ``model_owners`` so training fees debit the same address on every node (node 0)."""
    canonical = nodes[0].address
    mid = _e2e_mining_model_id()
    for n in nodes:
        n.consensus.register_model_owner(mid, canonical)


def _e2e_seed_initial_liquid_all_replicas(nodes: list[DessinNode], *, per_address: float) -> None:
    """Move liquid from genesis to each node address on **every** replica (identical pre-start state)."""
    if per_address <= 1e-12:
        return
    addrs = [n.address for n in nodes]
    total_per_replica = per_address * float(len(addrs))
    for n in nodes:
        es = n.consensus.economic_system
        g = es.genesis_address()
        avail = float(es.balances.get(g, 0.0))
        if avail < total_per_replica - 1e-6:
            raise RuntimeError(
                f"e2e genesis underfunded for DESSIN_E2E_INITIAL_LIQUID_BONUS: "
                f"need {total_per_replica:.2f} have {avail:.2f}"
            )
        for a in addrs:
            es.balances[g] = float(es.balances.get(g, 0.0)) - per_address
            es.balances[a] = float(es.balances.get(a, 0.0)) + per_address
    _e2e_log(
        f"💎🪙 e2e: +{per_address:.0f} DESSIN liquid per node from genesis on each replica "
        f"({len(addrs)} addrs × all nodes, before start)",
        kind="setup",
    )


def _e2e_model_owner_min_liquid(*, duration_sec: float, min_new_blocks: int) -> float:
    """Liquid DESSIN the model owner needs for ``training_payment_per_step`` remainder (no escrow in e2e)."""
    steps = int(os.environ.get("DESSIN_MICRO_GPT_STEPS", "64"))
    per_block = float(steps) * 0.11  # EconomicSystem.training_payment_per_step 0.1 + size slack
    # Soak can outpace min_new_blocks; assume several miners and ~25s effective inter-block during test.
    est_paid_blocks = float(min_new_blocks) + max(20.0, (float(duration_sec) / 22.0) * 4.5)
    return max(120.0, per_block * est_paid_blocks * 1.2)


def _e2e_prepare_model_owner_liquid(
    owner_node: DessinNode,
    *,
    min_liquid: float = 25.0,
    max_wait_sec: float = 120.0,
) -> None:
    """
    Training payments need liquid DESSIN on the model owner; bootstrap rewards are auto-staked.
    Wait for stake from mining, queue unstake, wait cooldown blocks, then credit matured balance.
    If stake never appears in time, mint from genesis (e2e reliability fallback).
    """
    cfg_cd = int(owner_node.config.consensus.unstake_cooldown_blocks)
    es = owner_node.consensus.economic_system
    addr = owner_node.address

    def tip() -> int:
        return owner_node.consensus.get_chain_length()

    def spendable() -> float:
        es.finalize_matured_unstakes(addr, tip())
        return es.preview_transferable_liquid(addr, tip())

    if spendable() >= min_liquid:
        return

    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait_sec:
        es.finalize_matured_unstakes(addr, tip())
        if es.get_staked_amount(addr) >= 2.0:
            break
        time.sleep(0.35)

    st = es.get_staked_amount(addr)
    if st >= 1e-6:
        unstake_amt = min(st, max(min_liquid * 2.0, 40.0))
        h0 = tip()
        owner_node.request_unstake(unstake_amt)
        unlock_at = h0 + cfg_cd
        t1 = time.monotonic()
        while time.monotonic() - t1 < max_wait_sec:
            if owner_node.consensus.get_latest_block().index >= unlock_at:
                break
            time.sleep(0.25)
        es.finalize_matured_unstakes(addr, owner_node.consensus.get_latest_block().index)

    if spendable() >= min_liquid:
        _e2e_log(
            f"model owner liquid ok spendable≈{spendable():.2f} DESSIN",
            kind="ok",
        )
        return

    g = es.genesis_address()
    need = min_liquid - spendable()
    avail = float(es.balances.get(g, 0.0))
    take = min(max(need, 10.0), avail)
    if take > 1e-9:
        es.balances[g] = avail - take
        es.balances[addr] = es.get_balance(addr) + take
        es.total_tokens_emitted += take
        _e2e_log(
            f"model owner liquid +{take:.2f} DESSIN from genesis (e2e fallback)",
            kind="warn",
        )


def _e2e_log_chain_progress(
    nodes: list[DessinNode],
    seen_hashes: set[str],
    *,
    phase: str,
) -> None:
    """Print each newly observed block: index, losses, leader (miner mapped to node_N)."""
    view_idx, chain = _canonical_chain(nodes)
    for block in chain:
        bh = getattr(block, "hash", "") or ""
        if not bh or bh in seen_hashes:
            continue
        seen_hashes.add(bh)
        leader = _leader_node_label(nodes, block.miner)
        delta = float(block.loss_before) - float(block.loss_after)
        miner_short = block.miner[:18] + "…" if len(block.miner) > 18 else block.miner
        _e2e_log(
            f"BLOCK idx={block.index} leader={leader} miner={miner_short} "
            f"loss_before={float(block.loss_before):.6f} loss_after={float(block.loss_after):.6f} "
            f"Δloss={delta:.6f} model={block.model_id} chain_view=node_{view_idx} phase={phase}",
            kind="block",
        )


def _wait_tip_agreement(
    nodes: list[DessinNode],
    *,
    timeout_sec: float,
    poll_sec: float,
    max_height_spread: int,
    block_seen_hashes: set[str] | None = None,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    logged_at = 0.0
    while time.monotonic() < deadline:
        if block_seen_hashes is not None:
            _e2e_log_chain_progress(nodes, block_seen_hashes, phase="wait_tip")
        heights, tips = _chain_snapshots(nodes)
        peers = _peer_counts(nodes)
        spread = max(heights) - min(heights) if heights else -1
        now = time.monotonic()
        if now - logged_at >= max(poll_sec, 8.0):
            tip_preview = [t[:12] + "…" if len(t) > 12 else t for t in tips]
            rem = deadline - now
            _e2e_log(
                f"wait_tip spread={spread} max_allowed={max_height_spread} "
                f"heights={heights} tip_prefixes={tip_preview} peers={peers} remaining_s={rem:.1f}",
                kind="wait",
            )
            logged_at = now
        if heights and spread <= max_height_spread:
            max_h = max(heights)
            tips_at_max = [t for t, h in zip(tips, heights) if h == max_h]
            if len(set(tips_at_max)) == 1:
                _e2e_log(
                    f"tip agreement ok max_h={max_h} tip={tips_at_max[0][:24]}… peers={peers}",
                    kind="ok",
                )
                return True
        time.sleep(poll_sec)
    heights, tips = _chain_snapshots(nodes)
    _e2e_log(
        f"wait_tip TIMEOUT heights={heights} peers={_peer_counts(nodes)} tips_prefix={[x[:16] for x in tips]}",
        kind="bad",
    )
    return False


def _run_four_node_battery(
    *,
    duration_sec: float,
    warmup_sec: float,
    min_new_blocks: int,
    min_peers_any_node: int,
    converge_timeout_sec: float,
    converge_poll_sec: float,
    max_height_spread: int,
) -> None:
    _e2e_timing_start(
        warmup_sec=warmup_sec,
        duration_sec=duration_sec,
        converge_sec=converge_timeout_sec,
    )

    ports = _find_free_ports(4)
    tmp_root = tempfile.mkdtemp(prefix="dessin_e2e_")
    tmp_root_path = Path(tmp_root)
    _e2e_log(f"workdir={tmp_root} ports={ports} (4 nodes)", kind="setup")

    os.environ.setdefault("DESSIN_UNSTAKE_COOLDOWN_BLOCKS", "2")

    configs = [
        {"mining": True},
        {"mining": True},
        {"mining": True},
        {"mining": False},
    ]
    nodes: list[DessinNode] = []

    try:
        for idx, port in enumerate(ports):
            cfg = DessinConfig.default()
            cfg.network.port = port
            cfg.network.log_node_label = f"node_{idx}"
            cfg.network.max_peers = 24
            cfg.model.model_cache_dir = str(tmp_root_path / f"cache_{idx}")
            # One-minute mining cadence; verification phases advance by chain height (not fixed wall-clock hours).
            cfg.consensus.training_block_time_minutes = 1.0
            cfg.consensus.verification_phase_completion_in_blocks = True
            cfg.consensus.phase1_window_blocks = int(
                os.environ.get("DESSIN_E2E_PHASE1_BLOCKS", "1")
            )
            cfg.consensus.phase2_window_blocks = int(
                os.environ.get("DESSIN_E2E_PHASE2_BLOCKS", "1")
            )
            cfg.consensus.finalization_window = int(
                os.environ.get("DESSIN_E2E_FINALIZATION_WINDOW_BLOCKS", "3")
            )
            cfg.consensus.bootstrap_period_blocks = int(
                os.environ.get("DESSIN_E2E_BOOTSTRAP_PERIOD_BLOCKS", "10")
            )
            # Spot-check verification gate (declarative for now): exercises config plumbing.
            # The trace pipeline is run as a post-converge phase below; live PogoBlocks still
            # use the quantization-consistency path until block-schema fields land.
            cfg.consensus.enable_spot_check_verification = True
            if configs[idx]["mining"]:
                cfg.consensus.block_time_minutes = 1.0
            else:
                cfg.consensus.block_time_minutes = 0.0

            node = DessinNode(cfg)
            if idx == 0:
                os.environ["DESSIN_CANONICAL_MODEL_OWNER"] = node.address
            nodes.append(node)
            _e2e_log(
                f"instantiated node {idx} port={port} address={node.address[:16]}… (start deferred)",
                kind="setup",
            )

        liq_bonus = float(os.environ.get("DESSIN_E2E_INITIAL_LIQUID_BONUS", "1000"))
        _e2e_seed_initial_liquid_all_replicas(nodes, per_address=liq_bonus)

        for idx, port in enumerate(ports):
            node = nodes[idx]
            assert node.start(), f"node {idx} failed to start on port {port}"
            role = "miner" if configs[idx]["mining"] else "listener"
            _e2e_log(
                f"started node {idx} ({role}) port={port} address={node.address[:16]}…",
                kind="setup",
            )
            time.sleep(0.35)

        genesis_hashes = [n.consensus.chain[0].hash for n in nodes]
        assert len(set(genesis_hashes)) == 1, (
            "all nodes must share identical canonical genesis tip; hashes="
            f"{genesis_hashes}"
        )

        _boot_n = int(nodes[0].config.consensus.bootstrap_period_blocks)
        _e2e_log(
            f"battery chain plan: target +{min_new_blocks} new blocks vs baseline; "
            f"bootstrap_period_blocks={_boot_n} (expect ≈{_boot_n} in-window + "
            f"≈{max(0, min_new_blocks - _boot_n)} post-bootstrap at 1-min slots)",
            kind="setup",
        )

        for idx, node in enumerate(nodes):
            bound = int(node.chaincraft_node.port)
            if bound != int(ports[idx]):
                _e2e_log(
                    f"node {idx}: requested port {ports[idx]} but Chaincraft bound {bound} "
                    "(UDP bind retry — mesh uses bound port)",
                    kind="warn",
                )

        # Bootstrap the configured LM model on **every** node so all miners share the
        # same catalog and never auto-bootstrap a divergent local model id (which
        # would fork the chain at the same height with different model_ids).
        try:
            mid = _e2e_mining_model_id()
            label = "Micro GPT (char)" if mid == "micro_gpt_char" else "GPT-2 nano (decoder LM)"
            for idx, node in enumerate(nodes):
                node._create_dummy_model(model_id=mid, name=label)
            _e2e_log(
                f"bootstrap: registered model_id={mid} on all {len(nodes)} nodes "
                f"(canonical owner=node_0 {nodes[0].address[:16]}…)",
                kind="setup",
            )
        except Exception as exc:
            _e2e_log(f"bootstrap: dummy model skipped ({exc!r})", kind="warn")

        settle = float(os.environ.get("DESSIN_E2E_MESH_SETTLE_SEC", "1.75"))
        time.sleep(settle)
        _e2e_log(
            f"mesh wait: warmup_budget rounds≈{max(12, int(warmup_sec / 2))} "
            f"pause≈{max(1.5, warmup_sec / 8):.1f}s min_peers_each={min_peers_any_node}",
            kind="mesh",
        )
        peers_after_warmup = _wait_mesh_peers(
            nodes,
            min_peers_each=min_peers_any_node,
            rounds=max(12, int(warmup_sec / 2)),
            pause_sec=max(1.5, warmup_sec / 8),
        )
        assert peers_after_warmup and min(peers_after_warmup) >= min_peers_any_node, (
            "expected full mesh (each node sees enough peers); peer counts="
            f"{peers_after_warmup}"
        )
        _e2e_log(f"mesh ok peers_each={peers_after_warmup}", kind="ok")

        # Sync the training proposer-lottery roster on every node so each peer
        # agrees on the same candidate set before two distinct miners appear in
        # chain history (``EnhancedPogoConsensus.expected_training_leader``).
        miner_addrs = [
            n.address
            for n in nodes
            if float(n.config.consensus.block_time_minutes) > 0.0
        ]
        for n in nodes:
            if hasattr(n.consensus, "note_participating_miner"):
                for a in miner_addrs:
                    n.consensus.note_participating_miner(a)
        _e2e_log(
            f"leader lottery roster synced: {len(miner_addrs)} miner addresses",
            kind="setup",
        )

        _e2e_sync_canonical_model_owner(nodes)

        # Model owner (node0) pays training remainder from liquid; miners' rewards are often staked.
        owner_min = _e2e_model_owner_min_liquid(
            duration_sec=duration_sec, min_new_blocks=min_new_blocks
        )
        _e2e_prepare_model_owner_liquid(nodes[0], min_liquid=owner_min)
        _e2e_log(
            f"model owner target liquid≥{owner_min:.1f} DESSIN (bootstrap={nodes[0].config.consensus.bootstrap_period_blocks} blocks)",
            kind="setup",
        )

        block_seen_hashes: set[str] = set()

        heights0, tips0 = _chain_snapshots(nodes)
        baseline_max = max(heights0)
        _e2e_log(
            f"baseline chain_lengths={heights0} max={baseline_max} tip_prefix={[t[:16] for t in tips0]}",
            kind="baseline",
        )
        _e2e_log_chain_progress(nodes, block_seen_hashes, phase="baseline")

        soak_deadline = time.monotonic() + duration_sec
        interval = max(10.0, min(40.0, duration_sec / 3))
        soak_iter = 0
        while True:
            remaining = soak_deadline - time.monotonic()
            if remaining <= 0:
                break
            sleep_for = min(interval, remaining)
            _e2e_timing_extend_at_least(remaining_sec=remaining + converge_timeout_sec)
            _e2e_log(
                f"soak sleep {sleep_for:.1f}s soak_phase_remaining≈{remaining:.1f}s "
                f"(+ {converge_timeout_sec:.0f}s converge budget after soak)",
                kind="soak",
            )
            time.sleep(sleep_for)
            soak_iter += 1
            _mesh_connect(nodes)
            _mesh_repair_starved(nodes, min_peers_each=min_peers_any_node)
            _e2e_log_chain_progress(nodes, block_seen_hashes, phase=f"soak_iter_{soak_iter}")
            h, t = _chain_snapshots(nodes)
            _e2e_log(
                f"soak iter={soak_iter} heights={h} spread={max(h) - min(h)} "
                f"peers_each={_peer_counts(nodes)} tip_prefix={[x[:12] + '…' for x in t]}",
                kind="soak",
            )

        _e2e_timing_extend_at_least(remaining_sec=converge_timeout_sec)
        _e2e_log(
            f"final converge wait timeout={converge_timeout_sec}s poll={converge_poll_sec}s "
            f"max_height_spread={max_height_spread}",
            kind="wait",
        )
        assert _wait_tip_agreement(
            nodes,
            timeout_sec=converge_timeout_sec,
            poll_sec=converge_poll_sec,
            max_height_spread=max_height_spread,
            block_seen_hashes=block_seen_hashes,
        ), (
            "nodes did not converge on the same tip within deadline; "
            f"heights,tips={_chain_snapshots(nodes)} peers={_peer_counts(nodes)}"
        )

        _e2e_log_chain_progress(nodes, block_seen_hashes, phase="post_converge")

        heights1, tips1 = _chain_snapshots(nodes)
        _e2e_log(
            f"post-check heights={heights1} peers_each={_peer_counts(nodes)} "
            f"growth_vs_baseline_max={max(heights1) - baseline_max}",
            kind="info",
        )
        assert max(heights1) - min(heights1) <= max_height_spread, (
            f"chain height spread too large: {heights1}"
        )

        max_h = max(heights1)
        tip_at_max = {t for t, h in zip(tips1, heights1) if h == max_h}
        assert len(tip_at_max) == 1, f"ambiguous tips at max height: {tips1!r} / {heights1!r}"
        consensus_tip = tip_at_max.pop()
        for t, h in zip(tips1, heights1):
            if h == max_h:
                assert t == consensus_tip

        growth = max(heights1) - baseline_max
        required_extra = _e2e_required_new_blocks_cap(
            duration_sec, min_new_blocks, nodes, headroom=0.82
        )
        assert growth >= required_extra, (
            "chain did not grow enough for soak budget — "
            f"growth={growth} need>={required_extra} "
            f"(min_new_blocks request={min_new_blocks} is capped by duration={duration_sec}s and slot cadence) "
            f"baseline_max={baseline_max} heights_after={heights1}"
        )
        _e2e_log(
            f"battery growth ok: +{growth} blocks (required>={required_extra}, min_new request={min_new_blocks})",
            kind="ok",
        )
        _e2e_shakespeare_inference_from_battery(nodes, tmp_root_path)
        _e2e_spot_check_phase(nodes)

    finally:
        os.environ.pop("DESSIN_CANONICAL_MODEL_OWNER", None)
        for node in nodes:
            try:
                node.stop()
            except Exception:
                pass
        _e2e_timing_reset()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(3600)
@pytest.mark.skipif(
    os.environ.get("DESSIN_SKIP_E2E", "").strip().lower() in ("1", "true", "yes"),
    reason="DESSIN_SKIP_E2E set",
)
def test_four_node_live_consensus_battery_micro_llm():
    """
    Four real nodes (three miners, one non-mining listener): mesh peers, mine, converge.

    Mining uses the **micro GPT char** preset (tiny LM on bundled Shakespeare; ``DESSIN_LLM_VARIANT=micro_gpt_char``).
    Consensus uses **1-minute** initial block/training slot, **block-based** verification completion
    (phase lengths from ``DESSIN_E2E_PHASE*_BLOCKS`` / ``DESSIN_E2E_FINALIZATION_WINDOW_BLOCKS``), not wall-clock verification hours.

    Environment (optional):
      DESSIN_E2E_DURATION_SEC — soak after warmup (default 1500 ≈25 min for 20×1-min slots)
      DESSIN_E2E_WARMUP_SEC — settle mesh before baseline (default 25)
      DESSIN_E2E_MIN_NEW_BLOCKS — requested minimum new blocks vs baseline (default 20; capped by soak × slot cadence)
      DESSIN_E2E_MIN_PEERS — minimum outbound peers on every node after warmup (default 2)
      DESSIN_E2E_CONVERGE_TIMEOUT_SEC — final tip agreement wait (default 120)
      DESSIN_E2E_MAX_HEIGHT_SPREAD — allowed height skew across nodes (default 2)
      DESSIN_E2E_QUIET — set to 1 to suppress [e2e] lines (including per-block loss / leader logs)
      NO_COLOR / DESSIN_E2E_NO_FANCY — disable ANSI + emoji styling (plain text only)
      DESSIN_MICRO_GPT_* — see docs/E2E_NETWORK_TESTS.md (steps, batch, lr, …)
      DESSIN_E2E_PHASE1_BLOCKS / DESSIN_E2E_PHASE2_BLOCKS / DESSIN_E2E_FINALIZATION_WINDOW_BLOCKS — verification spans in blocks (defaults 1 / 1 / 3)
      DESSIN_E2E_INITIAL_LIQUID_BONUS — liquid from genesis per node address on every replica before start (default 1000)
      DESSIN_E2E_SKIP_SPOT_CHECK — set to 1 to skip the post-converge spot-check verification phase
      DESSIN_E2E_SPOT_CHECK_STEPS — gradient steps for the trace-recorded block (default 6; see docs/SPOT_CHECK_VERIFICATION.md)

    Run with ``pytest -svv`` so stdout is visible; see docs/E2E_NETWORK_TESTS.md.
    """
    _e2e_timing_reset()
    _prepare_e2e_llm_env_variant("micro_gpt_char")

    duration = float(os.environ.get("DESSIN_E2E_DURATION_SEC", str(E2E_BATTERY_DEFAULT_DURATION_SEC)))
    warmup = float(os.environ.get("DESSIN_E2E_WARMUP_SEC", "25"))
    min_new = int(os.environ.get("DESSIN_E2E_MIN_NEW_BLOCKS", str(E2E_BATTERY_DEFAULT_MIN_NEW_BLOCKS)))
    min_peers = int(os.environ.get("DESSIN_E2E_MIN_PEERS", "2"))
    converge_timeout = float(os.environ.get("DESSIN_E2E_CONVERGE_TIMEOUT_SEC", "120"))
    spread = int(os.environ.get("DESSIN_E2E_MAX_HEIGHT_SPREAD", "2"))

    assert duration >= 60.0, "DESSIN_E2E_DURATION_SEC must be >= 60 for a meaningful soak"

    _e2e_timing_start(
        warmup_sec=warmup,
        duration_sec=duration,
        converge_sec=converge_timeout,
    )
    _e2e_log(
        f"config duration={duration}s warmup={warmup}s min_new_blocks={min_new} "
        f"min_peers_each={min_peers} converge_timeout={converge_timeout}s spread<={spread}",
        kind="setup",
    )

    _run_four_node_battery(
        duration_sec=duration,
        warmup_sec=warmup,
        min_new_blocks=min_new,
        min_peers_any_node=min_peers,
        converge_timeout_sec=converge_timeout,
        converge_poll_sec=4.0,
        max_height_spread=spread,
    )


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(3600)
@pytest.mark.skipif(
    os.environ.get("DESSIN_SKIP_E2E", "").strip().lower() in ("1", "true", "yes"),
    reason="DESSIN_SKIP_E2E set",
)
@pytest.mark.skipif(
    os.environ.get("DESSIN_E2E_INCLUDE_GPT2_NANO", "").strip().lower()
    not in ("1", "true", "yes"),
    reason="Heavy path: export DESSIN_E2E_INCLUDE_GPT2_NANO=1 to run GPT-2-class (gpt2_nano) four-node battery",
)
def test_four_node_live_consensus_battery_gpt2_nano():
    """
    Same mesh/mining battery as ``test_four_node_live_consensus_battery_micro_llm`` but forces
    ``DESSIN_LLM_VARIANT=gpt2_nano`` (decoder LM with GPT-2-class defaults; see ``dessin.llm.gpt2_nano_trainer``).

    Optional overrides: ``DESSIN_GPT2_*`` / ``DESSIN_MICRO_GPT_*`` (steps still read ``DESSIN_MICRO_GPT_STEPS`` in trainer).
    """
    _e2e_timing_reset()
    _prepare_e2e_llm_env_variant("gpt2_nano")

    duration = float(os.environ.get("DESSIN_E2E_DURATION_SEC", str(E2E_BATTERY_DEFAULT_DURATION_SEC)))
    warmup = float(os.environ.get("DESSIN_E2E_WARMUP_SEC", "25"))
    min_new = int(os.environ.get("DESSIN_E2E_MIN_NEW_BLOCKS", str(E2E_BATTERY_DEFAULT_MIN_NEW_BLOCKS)))
    min_peers = int(os.environ.get("DESSIN_E2E_MIN_PEERS", "2"))
    converge_timeout = float(os.environ.get("DESSIN_E2E_CONVERGE_TIMEOUT_SEC", "120"))
    spread = int(os.environ.get("DESSIN_E2E_MAX_HEIGHT_SPREAD", "2"))

    assert duration >= 60.0, "DESSIN_E2E_DURATION_SEC must be >= 60 for a meaningful soak"

    _e2e_timing_start(
        warmup_sec=warmup,
        duration_sec=duration,
        converge_sec=converge_timeout,
    )
    _e2e_log(
        f"[gpt2_nano] config duration={duration}s warmup={warmup}s min_new_blocks={min_new} "
        f"min_peers_each={min_peers} converge_timeout={converge_timeout}s spread<={spread}",
        kind="setup",
    )

    _run_four_node_battery(
        duration_sec=duration,
        warmup_sec=warmup,
        min_new_blocks=min_new,
        min_peers_any_node=min_peers,
        converge_timeout_sec=converge_timeout,
        converge_poll_sec=4.0,
        max_height_spread=spread,
    )
