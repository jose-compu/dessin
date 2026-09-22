# DeSSIN — Advanced Technical Documentation

This index organises the reference-level documents for contributors and protocol implementors. Each doc assumes familiarity with the concepts introduced in the [guided book](README.md) (Chapters 1–8).

---

## 1. Consensus & Verification

| Document | What it specifies |
|---|---|
| [POGO_PROTOCOL_SPECIFICATION.md](POGO_PROTOCOL_SPECIFICATION.md) | Full formal spec: block fields, gradient hash chain, Merkle tree, two-phase verification, quantization, torrents, economics |
| [SPOT_CHECK_VERIFICATION.md](SPOT_CHECK_VERIFICATION.md) | VRF-based spot-check path: mining commitments, verifier steps, strike ledger |
| [EFFICIENT_VERIFICATION.md](EFFICIENT_VERIFICATION.md) | Verification strategy: spot-check ($k=2$ gradient replays) is the active security primitive; quantization-consistency shortcut is designed but not yet wired in production |
| [ADAPTIVE_BLOCK_TIME.md](ADAPTIVE_BLOCK_TIME.md) | Utilization-based adaptive block time: stress/slack thresholds (+15% / −5%), asymmetric adjustment |
| [BLOCK_TIME_SCALING.md](BLOCK_TIME_SCALING.md) | Logarithmic size-based block-time scaling formula; `calculate_model_size_factor` |

---

## 2. Training & Fine-Tuning

| Document | What it specifies |
|---|---|
| [FINE_TUNING_PIPELINE.md](FINE_TUNING_PIPELINE.md) | Multi-stage pipeline manager: `FineTuningPipelineManager`, progression conditions, transactions |
| [FINE_TUNING_TECHNIQUES.md](FINE_TUNING_TECHNIQUES.md) | Technique catalog: LoRA, QLoRA, full SFT, DPO, RLHF; presets, optimizers, schedulers |
| [MODEL_TRAINING_ROTATION.md](MODEL_TRAINING_ROTATION.md) | Fair rotation scheduler: `ModelTrainingScheduler`, priorities, blockchain integration |

---

## 3. Models & Data

| Document | What it specifies |
|---|---|
| [MODEL_LIFECYCLE.md](MODEL_LIFECYCLE.md) | Pause / resume / deprecate / withdraw flows; escrow economics; lifecycle transactions |
| [MODEL_SIZES.md](MODEL_SIZES.md) | GPT depth table (d12–d32): parameter count formula, training and storage estimates |
| [FILE_UPLOAD_SYSTEM.md](FILE_UPLOAD_SYSTEM.md) | `ModelFileManager`, upload phases, bundles, torrents, `UploadFileTransaction` |
| [NANOCHAT_INTEGRATION.md](NANOCHAT_INTEGRATION.md) | External nanochat setup, model creation, web UI, block-time bands |

---

## 4. Economics & Fees

| Document | What it specifies |
|---|---|
| [DYNAMIC_ECONOMICS.md](DYNAMIC_ECONOMICS.md) | Leader parameter tweaks, storage top-ups, owner controls, nanochat creation |
| [NODE_OPERATOR_API.md](NODE_OPERATOR_API.md) | Per-node pricing API, bounds, bulk updates, leader pricing targets |

---

## 5. Infrastructure & Networking

| Document | What it specifies |
|---|---|
| [BITTORRENT_SPECIFICATION.md](BITTORRENT_SPECIFICATION.md) | Formal BitTorrent model: piece math, magnet/torrent fields, peer state, PoGO integration |
| [E2E_NETWORK_TESTS.md](E2E_NETWORK_TESTS.md) | Four-node live E2E battery: Chaincraft mesh, env knobs, spot-check, how to run |

---

## Correct Import Paths (quick reference)

All internal modules follow the `dessin.<package>.<module>` convention:

| Symbol | Correct import |
|---|---|
| `DessinNode` | `from dessin.runtime.node import DessinNode` |
| `DessinConfig`, `ConsensusConfig`, `TrainingMarketConfig` | `from dessin.runtime.config import ...` |
| `QuantizationVerifier` | `from dessin.protocol.quantization_verifier import QuantizationVerifier` |
| `SpotCheckVerifier` | `from dessin.consensus.spot_check_verification import SpotCheckVerifier` |
| `DynamicBlockTimeManager` | `from dessin.consensus.dynamic_block_time import DynamicBlockTimeManager` |
| `NodeOperatorAPI` | `from dessin.networking.node_operator_api import NodeOperatorAPI` |
| `FineTuningPipelineManager` | `from dessin.fine_tuning.fine_tuning_pipeline import FineTuningPipelineManager` |
| `FineTuningTechniques`, `LoRAConfig` | `from dessin.fine_tuning import ...` |
| `ComputeFeeSchedule` | `from dessin.economics.compute_fee_schedule import ComputeFeeSchedule` |
| `DynamicParameters` | `from dessin.economics.dynamic_parameters import DynamicParameters` |
| `StorageLeaseManager` | `from dessin.economics.storage_lease import StorageLeaseManager` |
| `NanochatIntegration` | `from dessin.nanochat.nanochat_integration import NanochatIntegration` |
| `ModelFileManager` | `from dessin.models.model_file_manager import ModelFileManager` |
| `ModelTrainingScheduler` | `from dessin.models.model_training_scheduler import ModelTrainingScheduler` |

---

> **Note on test counts** — any document that references a specific test count (e.g. "418 passing") should be treated as a snapshot, not a live count. Always run `pytest tests/ -q` to get the current number.
