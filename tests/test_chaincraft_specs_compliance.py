"""
Regression tests for Chaincraft SPECS v2 expectations.

See: https://github.com/jose-compu/chaincraft/blob/main/SPECS.md
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Optional, Union, get_args, get_origin, get_type_hints

import pytest

pytest.importorskip("chaincraft.state_memento")

from chaincraft.node import ChaincraftNode
from chaincraft.shared_message import SharedMessage
from chaincraft.shared_object import SharedObject
from chaincraft.state_memento import StateMemento

from dessin.runtime.config import ConsensusConfig, ModelConfig
from dessin.consensus import PogoConsensus
from dessin.consensus.enhanced_pogo_consensus import EnhancedPogoConsensus
from dessin.models.model_manager import ModelManager
from dessin.runtime.node import DessinNode


def _is_optional_state_memento(annotation: object) -> bool:
    if annotation is None:
        return False
    origin = get_origin(annotation)
    args = tuple(get_args(annotation))
    if origin is Union:
        types = list(args)
        return StateMemento in types and type(None) in types
    return annotation is StateMemento


@pytest.fixture
def consensus_specs():
    tmp = Path(tempfile.mkdtemp())
    import shutil

    try:
        mc = ModelConfig()
        mc.model_cache_dir = str(tmp)
        mgr = ModelManager(mc)
        mgr.register_model(
            model_id="specs_model",
            name="Specs Model",
            size_gb=0.1,
            format="pytorch",
            quantization="32bit",
            parameters=1000,
            ipfs_hash="QmSpecs",
            owner="0x" + "1" * 40,
            upload_block=0,
            storage_expires=1000,
            model_hash="mh",
        )
        cfg = ConsensusConfig()
        yield PogoConsensus(cfg, mgr, "0xminer_spec", enable_bittorrent=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_specs_sharedobject_subclass(consensus_specs: PogoConsensus):
    """SPECS: protocol logic is a SharedObject subclass."""
    assert isinstance(consensus_specs, SharedObject)


def test_specs_add_message_accepts_frontier_state_and_returns_optional_memento(
    consensus_specs: PogoConsensus,
):
    """SPECS gossip pipeline: add_message(..., frontier_state?) -> Optional[StateMemento]."""
    import sys

    if sys.version_info >= (3, 9):
        hints = get_type_hints(PogoConsensus.add_message, include_extras=False)
    else:
        hints = get_type_hints(PogoConsensus.add_message)
    assert hints.get("frontier_state") == Optional[StateMemento]
    sig = inspect.signature(PogoConsensus.add_message)
    assert "frontier_state" in sig.parameters

    ann = hints.get("return")
    assert _is_optional_state_memento(
        ann
    ), f"expected Optional[StateMemento] return annotation, got {ann!r}"


def test_specs_emit_and_get_state_digests(consensus_specs: PogoConsensus):
    """SPECS: emit_state_memento / get_state_digests expose frontier for pipeline."""
    tip = consensus_specs.get_latest_block_hash()
    memento = consensus_specs.emit_state_memento()
    assert isinstance(memento, StateMemento)
    assert memento.canonical_digest == tip
    digs = consensus_specs.get_state_digests()
    assert isinstance(digs, list)
    assert tip in digs


def test_specs_add_message_returns_memento_compact_pogo_block(consensus_specs: PogoConsensus):
    """Compact POGO_BLOCK notification must validate and emit memento (gossip via create_shared_message)."""
    msg = SharedMessage(
        data={
            "message_type": "POGO_BLOCK",
            "height": 999,
            "block_hash": "ab" * 32,
            "miner": consensus_specs.miner_address,
            "model_id": "specs_model",
            "torrent_hash": None,
        }
    )
    assert consensus_specs.is_valid(msg)
    out = consensus_specs.add_message(msg, frontier_state=None)
    assert isinstance(out, StateMemento)
    assert out.canonical_digest == consensus_specs.get_latest_block_hash()


def test_specs_add_message_accepts_upstream_memento(consensus_specs: PogoConsensus):
    upstream = consensus_specs.emit_state_memento()
    msg = SharedMessage(
        data={
            "message_type": "POGO_TRANSACTION",
            "transaction_data": {"tx_type": "noop"},
        }
    )
    assert consensus_specs.is_valid(msg)
    out = consensus_specs.add_message(msg, frontier_state=upstream)
    assert isinstance(out, StateMemento)


def test_specs_merkelized_surface(consensus_specs: PogoConsensus):
    """SPECS Merkelized Objects: digest API + gossip_object returns SharedMessages."""
    assert consensus_specs.is_merkelized() is True
    tip = consensus_specs.get_latest_block_hash()
    assert consensus_specs.get_latest_digest() == tip
    assert consensus_specs.has_digest(tip) is True
    assert consensus_specs.is_valid_digest(tip) is True

    gossiped = consensus_specs.gossip_object(tip)
    assert isinstance(gossiped, list)
    for item in gossiped:
        assert isinstance(item, SharedMessage)

    since = consensus_specs.get_messages_since_digest(tip)
    assert isinstance(since, list)


def test_specs_create_shared_message_gossip_path_integration(consensus_specs: PogoConsensus):
    """SPECS local create: validate + store pipeline via create_shared_message (no broadcast)."""
    node = ChaincraftNode(max_peers=5, port=0)
    node.add_shared_object(consensus_specs)
    node.start()
    try:
        payload = {
            "message_type": "POGO_BLOCK",
            "height": 42,
            "block_hash": "cd" * 32,
            "miner": consensus_specs.miner_address,
            "model_id": "specs_model",
            "torrent_hash": "th",
        }
        mh, wrapped = node.create_shared_message(payload)
        assert isinstance(mh, str) and len(mh) > 0
        assert isinstance(wrapped, SharedMessage)
    finally:
        if hasattr(node, "close"):
            node.close()
        elif hasattr(node, "stop"):
            node.stop()


def test_specs_dessin_node_mine_block_uses_create_shared_message_not_broadcast():
    """SPECS: prefer node.create_shared_message(data) over broadcast() for gossip protocol messages."""
    src = inspect.getsource(DessinNode.mine_block)
    assert "create_shared_message" in src
    assert ".broadcast(" not in src


def test_specs_enhanced_pogo_same_add_message_contract():
    """EnhancedPogoConsensus must keep parent SPECS add_message wiring."""
    assert EnhancedPogoConsensus.add_message is PogoConsensus.add_message
