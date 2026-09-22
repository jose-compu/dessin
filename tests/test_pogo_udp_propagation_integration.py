"""
UDP integration: gossip POGO_BLOCK through Chaincraft (localhost, >3 nodes).

Uses a ring topology so each node's broadcast relays to the next peer until all
listeners have appended the block (Chaincraft validates + floods).

Ports: omit ``port`` so Chaincraft allocates ``random.randint(5000, 9000)`` for
``self.port`` before bind. Do not pass ``port=0`` (bound port is ephemeral but
``self.port`` can remain 0, breaking ``connect_to_peer``).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("chaincraft.node")

from chaincraft.shared_message import SharedMessage
from chaincraft.node import ChaincraftNode

from dessin import consensus as consensus_mod
from dessin.runtime.config import ConsensusConfig, ModelConfig
from dessin.consensus import PogoConsensus
from dessin.models.model_manager import ModelManager

from tests._pogo_test_helpers import (
    make_synthetic_block_for_tip,
    pogo_block_payload,
    register_minimal_model,
)


def _wait_all_chains(
    stacks: list[tuple[ChaincraftNode, PogoConsensus]],
    *,
    expected_len: int,
    common_tip: str | None = None,
    timeout_s: float = 14.0,
    step_s: float = 0.06,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_lengths: list[int] = []
    while time.monotonic() < deadline:
        last_lengths = [len(c.chain) for _, c in stacks]
        if last_lengths != [expected_len] * len(stacks):
            time.sleep(step_s)
            continue
        if common_tip is None:
            return
        tips = [c.get_latest_block_hash() for _, c in stacks]
        if len(set(tips)) == 1 and tips[0] == common_tip:
            return
        time.sleep(step_s)
    pytest.fail(f"UDP gossip timeout lengths={last_lengths} expected_len={expected_len}")


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_udp_four_node_ring_three_pogo_blocks_converge(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
):
    monkeypatch.setattr(consensus_mod.time, "time", lambda: 1_734_441_777.0)
    cfg = ConsensusConfig()
    stacks: list[tuple[ChaincraftNode, PogoConsensus]] = []

    try:
        for i in range(4):
            d = Path(tmp_path_factory.mktemp(f"udp_node_{i}"))
            mc = ModelConfig()
            mc.model_cache_dir = str(d)
            mgr = ModelManager(mc)
            register_minimal_model(mgr)
            consensus = PogoConsensus(
                cfg, mgr, f"0xudp_miner{i}", enable_bittorrent=False
            )
            node = ChaincraftNode(
                max_peers=20,
                local_discovery=False,
                debug=False,
            )
            node.add_shared_object(consensus)
            node.start()
            stacks.append((node, consensus))
            time.sleep(0.04)

        for i in range(4):
            a = stacks[i][0]
            b = stacks[(i + 1) % 4][0]
            a.connect_to_peer(b.host, int(b.port))
        time.sleep(0.1)

        publisher_con = stacks[0][1]

        for round_i in range(3):
            block = make_synthetic_block_for_tip(
                publisher_con, vrf_byte=0x60 + round_i
            )
            payload = pogo_block_payload(block)
            assert publisher_con.is_valid(SharedMessage(data=payload)), round_i

            stacks[0][0].create_shared_message(payload)

            want_len = len(publisher_con.chain)
            tip = publisher_con.get_latest_block_hash()

            _wait_all_chains(
                stacks,
                expected_len=want_len,
                common_tip=tip,
                timeout_s=16.0,
            )

        for _, consensus in stacks:
            assert len(consensus.chain) == 4

        tip_final = stacks[0][1].get_latest_block_hash()
        for _, consensus in stacks[1:]:
            assert consensus.get_latest_block_hash() == tip_final
    finally:
        for cn, _ in stacks:
            try:
                cn.close()
            except Exception:
                pass
