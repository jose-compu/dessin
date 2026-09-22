# Changelog

All notable changes to the DeSSIN project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- **PoGO Consensus Implementation**: Full Proof of Gradient Optimization consensus mechanism
  - VRF-based leader selection using Chaincraft crypto primitives
  - Verifiable model training with gradient descent
  - Multiple Merkle proof verification (2-8 proofs per block)
  - Quantization consistency checks (FLOAT32 vs INT4)
  
- **AI-Native Blockchain Features**:
  - GGUF model support with HuggingFace integration
  - Model quantization (4-bit, 8-bit, 16-bit, 32-bit)
  - Merkle tree commitments for model layers
  - Fine-tuning support with blockchain verification
  
- **Transaction Types**:
  - Transfer: Standard token transfers
  - ModelQuery: Pay-per-token model inference
  - ConditionalTransfer: AI-based conditional payments
  - ModelUpload: Upload models to network
  - ModelFork: Fork existing models
  - Attestation: Verifier attestations for consensus
  - FineTuning: Start fine-tuning jobs
  - DatasetUpload: Upload training datasets
  
- **Network & P2P**:
  - Chaincraft 0.4.1 P2P networking integration
  - Enhanced BitTorrent model distribution
  - Multi-port architecture (6881-6884)
  - Automatic peer discovery for testing
  
- **Economic System**:
  - Token-based model pricing
  - Storage rental system
  - Mining rewards (10 DESSIN per block)
  - Verifier attestation fees
  
- **Testing Infrastructure**:
  - 336 unit tests covering all major components
  - Integration tests for multi-node networks
  - End-to-end tests for model distribution
  - Test utilities for consensus verification
  
- **Documentation**:
  - Complete PoGO Protocol Specification
  - BitTorrent Integration Specification
  - Comprehensive README with examples
  - API documentation for key classes

### Security
- AGPL-3.0 license to ensure network modifications remain open source
- Multiple Merkle proof verification for enhanced security
- VRF randomness to prevent training data manipulation
- Slashing mechanism for invalid training claims
- Attestation threshold (2/3 stake) for block finalization

### Performance
- Configurable block times (1 hour default, adjustable to seconds for testing)
- Efficient quantized model verification (~10-100x cheaper than training)
- Parallel BitTorrent distribution for large models
- Optimized Merkle tree operations

### Known Limitations
- **Experimental Release**: This is an alpha release for research and testing
- Mock implementations for some HuggingFace features
- Limited to local/test networks (not production-ready)
- GGUF inference requires additional setup
- No cross-chain integrations yet

### Dependencies
- Python 3.10+
- Chaincraft 0.4.1+ (P2P networking)
- PyTorch, Transformers, HuggingFace Hub
- libtorrent for BitTorrent distribution
- See requirements.txt for full list

### Breaking Changes
None (initial release)

---

## [Unreleased]

### Planned for 0.2.0
- Real GGUF model inference with llama.cpp
- Enhanced quantization methods
- Cross-chain model sharing
- Governance features
- Performance optimizations
- Production-ready networking

### Future Considerations
- Model marketplace and discovery
- Advanced ML workloads
- Integration with other AI/ML tools
- Research collaborations
- DAO governance structure

---

For more information, see the [README](README.md) and [documentation](docs/).

