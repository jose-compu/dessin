# DeSSIN: Decentralized Secure Super Intelligence Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)

DeSSIN is a blockchain implementation based on **PoGO (Proof of Gradient Optimization)** consensus that enables decentralized training and verification of machine learning models. It supports GGUF model formats, real ML inference, conditional transfers based on model outputs, and comprehensive fine-tuning capabilities.

## Key Features

### 🧠 AI-Native Blockchain
- **PoGO Consensus**: Proof of Gradient Optimization with verifiable ML training
- **GGUF Model Support**: Native support for GGUF quantized models from HuggingFace
- **Model Queries**: Pay-per-token model inference on-chain
- **Conditional Transfers**: Smart contracts based on AI model outputs

### 🔒 Security & Verification
- **Multiple Merkle Proofs**: Enhanced security with 2-8 Merkle proofs per block
- **VRF Randomness**: Chaincraft-based VRF for training data selection
- **Quantization Verification**: 4-bit quantized model verification
- **Attestation System**: Distributed verification with positive/negative attestations

### 🚀 Advanced ML Features
- **HuggingFace Integration**: Download models from trusted publishers
- **Fine-tuning Support**: Full fine-tuning with blockchain verification
- **Model Economics**: Token-based pricing for compute and storage
- **Gradual Training**: Hour-long block times for meaningful ML training

### 🌐 Network & P2P
- **Chaincraft P2P**: Depends on Chaincraft **0.5.1** from [PyPI](https://pypi.org/project/chaincraft/) (`chaincraft==0.5.1` in `requirements.txt`; aligns with [SPECS v2](https://github.com/jose-compu/chaincraft/blob/main/SPECS.md) gossip path and optional `frontier_state` / `emit_state_memento`)
- **Enhanced BitTorrent**: Multi-port architecture (6881-6884) with complete torrent metadata in blocks
- **Local Discovery**: Automatic peer discovery for testing

## 📚 Documentation

Comprehensive technical specifications are available in the [`docs/`](docs/) directory:

### Core Specifications
- **[PoGO Protocol Specification](docs/POGO_PROTOCOL_SPECIFICATION.md)** - Complete mathematical specification of PoGO consensus
- **[BitTorrent Integration Specification](docs/BITTORRENT_SPECIFICATION.md)** - Mathematical framework for decentralized model distribution

### Feature Documentation
- **[Nanochat Integration](docs/NANOCHAT_INTEGRATION.md)** - Real nanochat LLM integration guide
- **[Dynamic Economics](docs/DYNAMIC_ECONOMICS.md)** - Economic system with dynamic pricing
- **[Node Operator API](docs/NODE_OPERATOR_API.md)** - Per-node pricing configuration
- **[Adaptive Block Time](docs/ADAPTIVE_BLOCK_TIME.md)** - Simple ±10% adjustment based on validation performance
- **[Model sizes](docs/MODEL_SIZES.md)** - GPT/nanochat parameter notes and sizing

### Overview
- **[Documentation Index](docs/README.md)** - Complete documentation overview

## Quick Start

### Prerequisites

```bash
# Python 3.10+ required (see pyproject.toml)
python3 --version

git clone https://github.com/<org>/dessin.git
cd dessin
```

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### Running a Single Node

```bash
# Start a DeSSIN node with mining
python3 dessin_cli.py start --port 8000 --mining
```

### Running a Test Network

```bash
# Start a 3-node test network
python3 dessin_cli.py test-network
```

This will start 3 nodes that automatically:
- Create test models
- Mine blocks every few seconds  
- Perform model training verification
- Show network statistics

### Web UI (local demo)

From the repository root with the venv activated (`pip install -e .`):

```bash
python e2e/scripts/start_web_ui.py
# or full node + UI:
python e2e/scripts/run_network_with_web_ui.py
```

Shell helper: `e2e/scripts/start_chat_network.sh`

### Running a 4-Node Local Network

For more comprehensive testing and development, you can run a 4-node interconnected network manually. This setup allows you to observe peer-to-peer communication, consensus mechanisms, and network behavior in real-time.

#### Prerequisites

Make sure you have 4 terminal windows/tabs available and the environment activated:

```bash
# In each terminal, activate the project venv
cd dessin
source .venv/bin/activate
```

#### Step-by-Step Setup

**Terminal 1 - Bootstrap Node (Node 1):**
```bash
# Start the first node as bootstrap
python3 e2e/scripts/start_node1.py
```

**Terminal 2 - Node 2:**
```bash
# Start second node (miner)
python3 e2e/scripts/start_node2.py
```

**Terminal 3 - Node 3:**
```bash
# Start third node (miner)
python3 e2e/scripts/start_node3.py
```

**Terminal 4 - Node 4:**
```bash
# Start fourth node (validator only)
python3 e2e/scripts/start_node4.py
```

**Alternative - Single Script:**
```bash
# Run all 4 nodes from a single terminal
python3 e2e/scripts/run_4node_network.py
```

**BitTorrent Model Sharing:**
```bash
# Run 4-node network with BitTorrent model distribution
python3 e2e/scripts/run_4node_torrent_network.py
```

#### Network Behavior

Once all nodes are running, you should observe:

1. **Peer Connections**: Each node connects to all other nodes for full mesh networking
2. **Block Mining**: Nodes mine blocks every 15 seconds for fast testing
3. **Model Training**: Each mining node trains real neural networks and creates model files
4. **BitTorrent Distribution**: Model files are shared via BitTorrent with torrent hashes stored in blocks
5. **Block Propagation**: When one node mines a block, it propagates to all peers with torrent metadata
6. **Consensus**: All nodes validate and accept valid blocks with PoGO protocol compliance
7. **Network Stats**: Each node shows connected peers and network statistics
6. **Full Connectivity**: All nodes should show 3 peers (connected to all other nodes)

#### Monitoring the Network

Each terminal will display:
- Node address and port
- Connected peers list
- Block mining progress
- Transaction processing
- Network statistics

#### Testing Network Resilience

You can test network behavior by:

1. **Stopping a node**: Kill one terminal and observe how the network continues
2. **Restarting a node**: Restart the stopped node and see it sync with the network
3. **Network partitions**: Stop multiple nodes to simulate network splits
4. **Transaction flow**: Create transactions on one node and observe propagation

#### Advanced Configuration

For more control over the network setup:

```bash
# Custom block time (15 seconds for testing)
python3 dessin_cli.py start --port 8001 --mining --bootstrap --block-time 15

# Disable mining on some nodes (validator-only nodes)
python3 dessin_cli.py start --port 8002 --peers 127.0.0.1:8001

# Set custom network parameters
python3 dessin_cli.py start --port 8003 --mining --peers 127.0.0.1:8001 --max-peers 20
```

#### BitTorrent Model Distribution

The enhanced 4-node network includes BitTorrent integration for decentralized model sharing:

**Model File Creation:**
- Each mining node creates actual model files (.json) containing:
  - Trained neural network weights (hex-encoded)
  - Training metadata (loss improvement, parameters)
  - Model architecture information
  - Integrity checksums

**Torrent Generation:**
- Model files are automatically converted to torrents
- Torrent hashes are included in blockchain blocks
- Magnet links enable peer-to-peer model sharing

**Network Distribution:**
- Miner nodes act as seeders for their trained models
- Validator nodes can download models from multiple peers
- File integrity is verified using checksums

**Monitoring:**
```bash
# Check model files created by each node
ls model_cache/node_*/torrents/*.json

# View model file content
head -20 model_cache/node_8001_torrents/*.json
```

#### Troubleshooting

If nodes fail to connect:

1. **Check ports**: Ensure ports 8001-8004 are available
2. **Firewall**: Make sure local firewall allows connections
3. **Bootstrap first**: Always start the bootstrap node first
4. **Wait for sync**: Give nodes time to discover and connect to each other

#### Network Cleanup

To stop the network:
1. Press `Ctrl+C` in each terminal to stop nodes gracefully
2. Or kill all processes: `pkill -f dessin_cli.py`

## Architecture Overview

### Core Components

1. **DessinNode**: Main blockchain node with P2P networking
2. **PogoConsensus**: PoGO consensus implementation with VRF and attestations
3. **ModelManager**: GGUF model loading, Merkle trees, and quantization
4. **TransactionSystem**: Multiple transaction types for AI operations
5. **FineTuningManager**: Full fine-tuning support with blockchain verification

### Transaction Types

- **Transfer**: Standard token transfers
- **ModelQuery**: Pay-per-token model inference
- **ConditionalTransfer**: Transfers based on AI model outputs  
- **ModelUpload**: Upload new models to the network
- **ModelFork**: Fork existing models for customization
- **Attestation**: Verifier attestations for consensus
- **FineTuning**: Start fine-tuning jobs
- **DatasetUpload**: Upload training datasets

### Consensus Flow

1. **Block Creation**: Miner selected via VRF performs model training
2. **Training Verification**: Actual gradient descent with loss improvement
3. **Quantization Check**: 4-bit model shows equivalent improvement
4. **Merkle Commitment**: Multiple random leaf proofs for security
5. **Attestation Phase**: Verifiers submit positive/negative attestations
6. **Finalization**: Block finalized if 2/3+ positive attestations

## Usage Examples

### Download and Register a Model

```python
from dessin import DessinNode
from dessin.hf_integration import HuggingFaceIntegration

# Start node
node = DessinNode()
node.start()

# Download a small model from HuggingFace
hf = HuggingFaceIntegration(node.model_manager)
model_id = hf.download_and_register_model(
    repo_id="bartowski/gemma-2-2b-it-GGUF",
    owner_address=node.address
)

print(f"Registered model: {model_id}")
```

### Query a Model

```python
# Query the model (pays per token)
response = node.query_model(
    model_id=model_id,
    query="What is the capital of France?",
    max_tokens=50
)

print(f"Response: {response}")
print(f"Balance after query: {node.get_balance()}")
```

### Create a Conditional Transfer

```python
# Transfer tokens only if sentiment is positive
success = node.create_conditional_transfer(
    recipient="0x1234567890123456789012345678901234567890",
    amount=10.0,
    model_id=sentiment_model_id,
    condition_query="I love sunny days!",
    condition_expected="positive",
    fallback_recipient="0x0987654321098765432109876543210987654321"
)
```

### Fine-tune a Model

```python
from dessin.fine_tuning import FineTuningManager

# Create fine-tuning manager
ft_manager = FineTuningManager(node.model_manager)

# Register a dataset
dataset_id = ft_manager.register_dataset(
    dataset_name="Custom Chat Dataset",
    dataset_path="./data/chat_dataset.jsonl",
    dataset_format="jsonl",
    owner=node.address
)

# Create fine-tuning job
job_id = ft_manager.create_fine_tuning_job(
    base_model_id=model_id,
    dataset_id=dataset_id,
    owner=node.address,
    num_epochs=3,
    max_cost_dessin=50.0
)

# Start fine-tuning
ft_manager.start_fine_tuning(job_id)
```

## Testing

### Run Unit Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test files
python3 -m pytest tests/test_transactions.py -v
python3 -m pytest tests/test_consensus.py -v
python3 -m pytest tests/test_model_manager.py -v
```

### Integration Tests

```bash
# Run integration tests (includes multi-node testing)
python3 -m pytest tests/test_integration.py -v
```

### Live four-node E2E (longer run)

Verbose output requires disabling pytest capture (`-s`). Example with env tuning and a full log file:

```bash
cd /Users/joseignacio/Documents/GitHub/dessin && \
DESSIN_E2E_DURATION_SEC=35 DESSIN_E2E_WARMUP_SEC=18 DESSIN_E2E_MIN_NEW_BLOCKS=1 \
DESSIN_E2E_MIN_PEERS=2 DESSIN_E2E_CONVERGE_TIMEOUT_SEC=120 \
python -m pytest e2e/tests/test_four_node_live_network.py \
  -svv --tb=short --color=yes --log-cli-level=INFO 2>&1 | tee /tmp/dessin_e2e_pytest.log
```

Defaults without env overrides follow [docs/E2E_NETWORK_TESTS.md](docs/E2E_NETWORK_TESTS.md) (soak duration and `DESSIN_E2E_MIN_NEW_BLOCKS`). The command above is a shorter smoke run.

See [docs/E2E_NETWORK_TESTS.md](docs/E2E_NETWORK_TESTS.md) for variables (`DESSIN_E2E_QUIET`, skip flags, etc.).

### Manual Testing

```bash
# List available HuggingFace models
python3 -c "
from dessin.hf_integration import HuggingFaceIntegration
from dessin.model_manager import ModelManager
from dessin.config import ModelConfig
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    config = ModelConfig()
    config.model_cache_dir = tmp
    manager = ModelManager(config)
    hf = HuggingFaceIntegration(manager)
    hf.list_available_models('tiny')
"
```

## Configuration

### Environment Variables

```bash
# Network configuration
export DESSIN_MAX_PEERS=10
export DESSIN_PORT=8000
export DESSIN_HOST=127.0.0.1

# Consensus configuration  
export DESSIN_BLOCK_TIME_MINUTES=0.6   # ~36s cadence for testing (legacy: DESSIN_BLOCK_TIME_HOURS×60)
export DESSIN_FINALIZATION_WINDOW=5

# Model configuration
export DESSIN_MODEL_CACHE=/path/to/model/cache
export DESSIN_MAX_MODEL_SIZE_GB=50
```

### Configuration Classes

```python
from dessin.config import DessinConfig

# Create custom configuration
config = DessinConfig.default()
config.consensus.block_time_minutes = 0.06  # ~3.6s cadence for local mining
config.consensus.max_merkle_proofs = 8     # Maximum security
config.network.max_peers = 20               # Large network
config.model.max_model_size_gb = 100        # Support larger models

node = DessinNode(config)
```

## Network Economics

### Token System
- **Symbol**: DESSIN
- **Decimals**: 18
- **Initial Supply**: 1,000,000 DESSIN
- **Mining Reward**: 10 DESSIN per block

### Pricing Model
- **Model Queries**: ~0.001 DESSIN per token
- **Storage Rental**: ~0.001 DESSIN per GB per block
- **Fine-tuning**: Based on model size and training steps
- **Transaction Fees**: 0.001-0.01 DESSIN

### Economic Incentives
- **Miners**: Earn rewards for valid training blocks
- **Verifiers**: Earn fees for attestations
- **Model Owners**: Earn from query fees
- **Users**: Pay for compute and storage

## Technical Specifications

### Supported Models
- **Format**: GGUF (primary), extensible to others
- **Quantization**: 4-bit, 8-bit, 16-bit
- **Size**: Up to 50GB (configurable)
- **Publishers**: Microsoft, Google, Meta, TheBloke, etc.

### Performance
- **Block Time**: 1 hour (configurable, can be seconds for testing)
- **Finalization**: 20 blocks (configurable)
- **TPS**: Optimized for ML operations, not high-frequency trading
- **Verification**: ~10-100x cheaper than training

### Security Features
- **Multiple Merkle Proofs**: 2-8 proofs per block (configurable)
- **VRF Randomness**: Prevents manipulation of training data selection
- **Quantization Verification**: Ensures consistency between models
- **Attestation Threshold**: 2/3 stake required for finalization
- **Slashing**: Penalties for invalid training claims

## Development

### Project Structure

```
dessin/
├── dessin/                       # Core package
│   ├── __init__.py              # Main exports
│   ├── node.py … progress_tracker.py  # Thin shims (stable import paths)
│   ├── pogo_protocol.py … quantization_verifier.py  # Shims → protocol/
│   │
│   ├── runtime/                 # Node runtime (config, node, CLI, console, progress)
│   │   ├── config.py
│   │   ├── node.py
│   │   ├── cli.py
│   │   ├── pretty_console.py
│   │   └── progress_tracker.py
│   │
│   ├── protocol/                # PoGO commitments, Merkle trees, quantization helpers
│   │   ├── pogo_protocol.py
│   │   └── quantization_verifier.py
│   │
│   ├── consensus/               # Consensus and verification
│   │   ├── consensus.py         # PoGO consensus implementation
│   │   ├── enhanced_pogo_consensus.py  # Enhanced consensus with attestations
│   │   ├── two_phase_verification.py   # Two-phase verification system
│   │   ├── spot_check_verification.py  # Spot-check replay verification
│   │   ├── slashing_mechanism.py       # Slashing logic
│   │   ├── dynamic_block_time.py       # Dynamic block time adjustment
│   │   └── transactions.py             # Transaction types
│   │
│   ├── economics/               # Tokenomics and payments
│   │   ├── economic_system.py   # Economic system
│   │   ├── storage_topup.py     # Storage rental management
│   │   └── dynamic_parameters.py # Dynamic parameter governance
│   │
│   ├── networking/              # Networking and health monitoring
│   │   ├── enhanced_networking.py
│   │   ├── chaincraft_peer_debug.py
│   │   ├── health_monitor.py
│   │   └── node_operator_api.py
│   │
│   ├── models/                  # Model management
│   │   ├── model_manager.py
│   │   ├── model_file_manager.py
│   │   ├── model_lifecycle_transactions.py
│   │   ├── model_training_scheduler.py
│   │   ├── model_tokenization_ops.py
│   │   ├── file_upload_transactions.py
│   │   └── llm_model_spec.py
│   │
│   ├── llm/                     # LLM training and inference
│   │   ├── decoder_lm_training.py
│   │   ├── decoder_lm_training_trace.py
│   │   ├── decoder_lm_inference.py
│   │   ├── decoder_only_gpt.py
│   │   ├── micro_gpt_trainer.py
│   │   ├── gpt2_nano_trainer.py
│   │   ├── simple_trainer.py
│   │   ├── simple_gpt_inference.py
│   │   └── torrent_model_trainer.py
│   │
│   ├── distribution/            # Model distribution (BitTorrent/HF)
│   │   ├── bittorrent_distributor.py
│   │   ├── enhanced_torrent_distributor.py
│   │   ├── real_torrent_distributor.py
│   │   ├── huggingface_distributor.py
│   │   ├── hybrid_distributor.py
│   │   ├── hf_integration.py
│   │   ├── hf_mock.py
│   │   └── hf_updater.py
│   │
│   ├── nanochat/              # Nanochat integration
│   │   ├── nanochat_integration.py
│   │   ├── nanochat_web.py
│   │   ├── nanochat_wrapper.py
│   │   └── local_model_runner.py
│   │
│   └── fine_tuning/           # Fine-tuning support
│       ├── fine_tuning.py
│       ├── fine_tuning_pipeline.py
│       ├── fine_tuning_techniques.py
│       └── fine_tuning_transactions.py
├── tests/                 # Test suite
│   ├── test_transactions.py
│   ├── test_consensus.py
│   ├── test_model_manager.py
│   └── test_integration.py
├── chaincraft/            # Optional vendored clone for development (prefer PyPI chaincraft==0.5.1)
├── e2e/
│   ├── scripts/           # Local demos, multi-node runners, web UI flows
│   └── data/demo_torrents/ # Torrent demo artifacts (gitignored outputs)
├── dessin_cli.py          # CLI helper (run from repo root)
├── scripts/               # Benchmarks, setup helpers (run from repo root)
└── README.md              # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Extending the System

- **New Transaction Types**: Extend `BaseTransaction` class
- **Custom Consensus**: Implement custom verification logic
- **Additional Model Formats**: Extend `ModelManager`
- **New P2P Features**: Use Chaincraft primitives

## Comparison to Other Systems

### vs. Traditional Blockchains
- **Purpose-built for AI**: Native ML operations vs. general computation
- **Useful Work**: Actual model training vs. arbitrary hash computation
- **Longer Block Times**: Meaningful training vs. fast finality

### vs. Bittensor
- **Cryptographic Verification**: Merkle proofs vs. peer scoring
- **Training Focus**: Model updates vs. inference markets
- **Decentralized Training**: Actual gradient descent vs. output evaluation

### vs. Centralized AI
- **Decentralized**: No single point of control
- **Verifiable**: Cryptographic proof of training
- **Incentivized**: Token rewards for contributions
- **Censorship Resistant**: Global model training network

## Roadmap

### Phase 1: Core Implementation ✅
- PoGO consensus with VRF and Merkle proofs
- GGUF model support and HuggingFace integration
- Basic transaction types and P2P networking
- Comprehensive testing suite

### Phase 2: Advanced Features (Current)
- Enhanced consensus with multiple Merkle proofs
- Fine-tuning support with blockchain verification
- Advanced model economics and pricing
- Performance optimizations

### Phase 3: Production Ready (Future)
- Real GGUF model inference with llama.cpp
- Advanced quantization methods
- Cross-chain model sharing
- Governance and DAO features

### Phase 4: Ecosystem (Future)
- Model marketplace and discovery
- Advanced ML workloads (training, inference, fine-tuning)
- Integration with other AI/ML tools
- Research collaborations

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** or later. See the [LICENSE](LICENSE) file for details.

### What This Means

The AGPL-3.0 license ensures that:
- **Free Software**: Anyone can use, modify, and distribute this software
- **Source Code Access**: Users who interact with the software over a network must have access to the source code
- **Contributions**: Contributions to this project will be licensed under the same AGPL-3.0 license
- **Derivative Works**: Modified versions must also be released under AGPL-3.0

For more information, see the [GNU AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.html).

## Contributing

We welcome contributions! By contributing to this project, you agree that your contributions will be licensed under the AGPL-3.0 license.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## Acknowledgments

- **PoGO Paper**: Based on the Proof of Gradient Optimization research
- **Chaincraft**: [Chaincraft](https://pypi.org/project/chaincraft/) on PyPI (SPECS v2 gossip + merkelized sync); source [jose-compu/chaincraft](https://github.com/jose-compu/chaincraft)
- **HuggingFace**: Integration with HuggingFace model ecosystem
- **GGUF Format**: Support for efficient quantized models

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.

---

**DeSSIN: Making AI training decentralized, verifiable, and incentivized.** 🧠⛓️
