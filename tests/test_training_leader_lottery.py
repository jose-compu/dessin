"""Training proposer lottery (``EnhancedPogoConsensus.expected_training_leader``)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from dessin.runtime.config import ConsensusConfig
from dessin.consensus import PogoBlock
from dessin.consensus.enhanced_pogo_consensus import EnhancedPogoConsensus
from dessin.models.model_manager import ModelManager


def _block(idx: int, miner: str, prev: str, h: str) -> PogoBlock:
    return PogoBlock(
        index=idx,
        timestamp=1.0,
        previous_hash=prev,
        miner=miner,
        model_id="m",
        training_data_hash="0" * 64,
        loss_before=1.0,
        loss_after=0.9,
        hash_full_model_32="0" * 64,
        hash_quant_4="0" * 64,
        vrf_proof=b"x",
        training_steps=1,
        learning_rate=0.01,
        batch_size=1,
        hash=h,
    )


def test_expected_training_leader_none_until_two_candidates():
    cfg = ConsensusConfig(bootstrap_period_blocks=0)
    mm = Mock(spec=ModelManager)
    c = EnhancedPogoConsensus(cfg, mm, miner_address="0xaaa")
    c.chain = [
        PogoBlock(
            index=0,
            timestamp=0.0,
            previous_hash="0" * 64,
            miner="genesis",
            model_id="genesis",
            training_data_hash="0" * 64,
            loss_before=0.0,
            loss_after=0.0,
            hash_full_model_32="0" * 64,
            hash_quant_4="0" * 64,
            vrf_proof=b"genesis",
            training_steps=0,
            learning_rate=0.0,
            batch_size=0,
            hash="genesis_hash",
        ),
        _block(1, "0xaaa", "genesis_hash", "hash_block_1"),
    ]
    assert c.expected_training_leader() is None


def test_expected_training_leader_deterministic_for_same_chain():
    cfg = ConsensusConfig(bootstrap_period_blocks=0)
    mm = Mock(spec=ModelManager)
    c = EnhancedPogoConsensus(cfg, mm, miner_address="0xccc")
    c.note_participating_miner("0xaaa")
    c.note_participating_miner("0xbbb")
    c.chain = [
        PogoBlock(
            index=0,
            timestamp=0.0,
            previous_hash="0" * 64,
            miner="genesis",
            model_id="genesis",
            training_data_hash="0" * 64,
            loss_before=0.0,
            loss_after=0.0,
            hash_full_model_32="0" * 64,
            hash_quant_4="0" * 64,
            vrf_proof=b"genesis",
            training_steps=0,
            learning_rate=0.0,
            batch_size=0,
            hash="H0",
        ),
        _block(1, "0xaaa", "H0", "H1"),
        _block(2, "0xbbb", "H1", "H2"),
    ]
    a = c.expected_training_leader()
    b = c.expected_training_leader()
    assert a is not None and a == b
    assert a in ("0xaaa", "0xbbb")


def test_expected_training_leader_valid_after_chain_extension():
    cfg = ConsensusConfig(bootstrap_period_blocks=0)
    mm = Mock(spec=ModelManager)
    c = EnhancedPogoConsensus(cfg, mm, miner_address="0xccc")
    for addr in ("0xaaa", "0xbbb", "0xccc", "0xddd", "0xeee"):
        c.note_participating_miner(addr)
    c.chain = [
        PogoBlock(
            index=0,
            timestamp=0.0,
            previous_hash="0" * 64,
            miner="genesis",
            model_id="genesis",
            training_data_hash="0" * 64,
            loss_before=0.0,
            loss_after=0.0,
            hash_full_model_32="0" * 64,
            hash_quant_4="0" * 64,
            vrf_proof=b"genesis",
            training_steps=0,
            learning_rate=0.0,
            batch_size=0,
            hash="H0",
        ),
        _block(1, "0xaaa", "H0", "H1"),
        _block(2, "0xbbb", "H1", "H2"),
    ]
    leader_before = c.expected_training_leader()
    c.chain.append(_block(3, "0xccc", "H2", "H3"))
    leader_after = c.expected_training_leader()
    pool = sorted(c.training_leader_candidates())
    assert leader_before is not None and leader_before in pool
    assert leader_after is not None and leader_after in pool
