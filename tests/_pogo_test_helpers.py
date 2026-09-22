"""Shared helpers for PoGO propagation tests (in-process + UDP integration)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dessin import consensus as consensus_mod
from dessin.runtime.config import ConsensusConfig, ModelConfig
from dessin.consensus import PogoConsensus, PogoBlock
from dessin.models.model_manager import ModelManager


def register_minimal_model(manager: ModelManager, model_id: str = "pipeline_model") -> None:
    manager.register_model(
        model_id=model_id,
        name="Pipeline Model",
        size_gb=0.001,
        format="pytorch",
        quantization="32bit",
        parameters=1000,
        ipfs_hash="QmPipeline",
        owner="0x" + "ab" * 20,
        upload_block=0,
        storage_expires=1000,
        model_hash="pipeline_hash",
    )


def make_pair(
    tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    miner_a: str,
    miner_b: str,
    fixed_ts: float = 1_700_000_000.0,
) -> tuple[PogoConsensus, PogoConsensus]:
    monkeypatch.setattr(consensus_mod.time, "time", lambda: fixed_ts)
    cfg = ConsensusConfig()
    mc = ModelConfig()
    mc.model_cache_dir = str(tmp)
    mgr_a = ModelManager(mc)
    mgr_b = ModelManager(mc)
    register_minimal_model(mgr_a)
    register_minimal_model(mgr_b)
    a = PogoConsensus(cfg, mgr_a, miner_a, enable_bittorrent=False)
    b = PogoConsensus(cfg, mgr_b, miner_b, enable_bittorrent=False)
    assert a.get_latest_block().hash == b.get_latest_block().hash
    return a, b


def make_synthetic_block_for_tip(
    consensus: PogoConsensus,
    *,
    model_id: str = "pipeline_model",
    vrf_byte: int = 7,
) -> PogoBlock:
    prev = consensus.get_latest_block()
    idx = prev.index + 1
    return PogoBlock(
        index=idx,
        timestamp=prev.timestamp + idx,
        previous_hash=prev.hash,
        miner=consensus.miner_address,
        model_id=model_id,
        training_data_hash=f"{idx:02x}" * 32,
        loss_before=1.0,
        loss_after=max(0.1, 1.0 - 0.01 * idx),
        hash_full_model_32="af" * 32,
        hash_quant_4="bf" * 32,
        vrf_proof=bytes([vrf_byte & 0xFF]) * 32,
        training_steps=idx,
        learning_rate=0.01,
        batch_size=8,
        finalization_block=idx + 10,
        torrent_hash=f"th{idx}",
        model_size_bytes=1024 * idx,
    )


def append_synthetic_block(consensus: PogoConsensus, **kwargs) -> PogoBlock:
    block = make_synthetic_block_for_tip(consensus, **kwargs)
    consensus.chain.append(block)
    return block


def pogo_block_payload(block: PogoBlock) -> dict:
    return {
        "message_type": "POGO_BLOCK",
        "height": block.index,
        "block_data": block.to_dict(),
    }
