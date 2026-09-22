"""
Unit tests for limited PoGO block propagation as a Chaincraft-style gossip pipeline.

Mirrors the sequential SharedObject pattern in Chaincraft's blockchain example tests
(validate all registered objects, apply add_message, optionally pass StateMemento downstream):
https://github.com/jose-compu/chaincraft/blob/main/tests/test_blockchain_example.py

These tests do not open UDP sockets; they simulate one node publishing chain
extensions and another (or the same) consensus applying POGO_BLOCK gossip payloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("chaincraft.shared_message")

from chaincraft.shared_message import SharedMessage
from chaincraft.state_memento import StateMemento

from dessin import consensus as consensus_mod
from dessin.runtime.config import ConsensusConfig, ModelConfig
from dessin.consensus import PogoConsensus
from dessin.models.model_manager import ModelManager

from tests._pogo_test_helpers import (
    append_synthetic_block,
    make_pair,
    make_synthetic_block_for_tip,
    pogo_block_payload,
    register_minimal_model,
)


def _apply_pipeline(
    receiver: PogoConsensus,
    payload: dict,
    *,
    frontier_state: StateMemento | None = None,
) -> StateMemento | None:
    """Chaincraft-example style: validate then apply, optional frontier from prior object."""
    msg = SharedMessage(data=payload)
    assert receiver.is_valid(msg), payload
    return receiver.add_message(msg, frontier_state=frontier_state)


@pytest.fixture
def tmp_models(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


def test_propagate_three_full_blocks_between_two_consensus_nodes(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
):
    """Sender extends chain locally; receiver applies three gossip envelopes (no UDP)."""
    sender, receiver = make_pair(
        tmp_models, monkeypatch, miner_a="0xsender", miner_b="0xreceiver"
    )

    for i in range(3):
        b = append_synthetic_block(sender, vrf_byte=0x10 + i)
        st = _apply_pipeline(receiver, pogo_block_payload(b))
        assert isinstance(st, StateMemento)
        assert st.canonical_digest == receiver.get_latest_block_hash()

    assert len(sender.chain) == 4
    assert len(receiver.chain) == 4
    for i in range(4):
        assert sender.chain[i].hash == receiver.chain[i].hash


def test_frontier_state_threaded_across_block_applies(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
):
    """Sequential add_message calls pass prior StateMemento (SPECS multi-object pattern)."""
    sender, receiver = make_pair(
        tmp_models, monkeypatch, miner_a="0xsender", miner_b="0xreceiver"
    )

    frontier: StateMemento | None = None
    for i in range(3):
        b = append_synthetic_block(sender, vrf_byte=0x20 + i)
        frontier = _apply_pipeline(
            receiver, pogo_block_payload(b), frontier_state=frontier
        )
        assert isinstance(frontier, StateMemento)

    assert len(sender.chain) == 4
    assert len(receiver.chain) == 4


def test_double_apply_same_block_idempotent(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
):
    """Duplicate POGO_BLOCK payloads validate (gossip idempotency) but do not extend the chain twice."""
    sender, recv = make_pair(tmp_models, monkeypatch, miner_a="0xa", miner_b="0xrecv")
    b = append_synthetic_block(sender, vrf_byte=0x33)
    payload = pogo_block_payload(b)

    _apply_pipeline(recv, payload)
    assert len(recv.chain) == 2

    msg2 = SharedMessage(data=payload)
    assert recv.is_valid(msg2)
    recv.add_message(msg2)
    assert len(recv.chain) == 2


def test_pipeline_single_node_two_step_validate_like_blockchain_example(
    tmp_models: Path, monkeypatch: pytest.MonkeyPatch
):
    """Mirror multi-object validation: two is_valid checks, one add_message (no pre-append)."""
    monkeypatch.setattr(consensus_mod.time, "time", lambda: 1_700_000_000.0)
    cfg = ConsensusConfig()
    mc = ModelConfig()
    mc.model_cache_dir = str(tmp_models)
    mgr = ModelManager(mc)
    register_minimal_model(mgr)
    node = PogoConsensus(cfg, mgr, "0xnode", enable_bittorrent=False)

    b = make_synthetic_block_for_tip(node, vrf_byte=0x44)
    msg = SharedMessage(data=pogo_block_payload(b))
    assert node.is_valid(msg)
    assert node.is_valid(msg)
    out = node.add_message(msg)
    assert isinstance(out, StateMemento)
    assert len(node.chain) == 2
    assert out.canonical_digest == node.get_latest_block_hash()
