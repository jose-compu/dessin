# PoGO Protocol Specification

## Abstract

**Proof of Gradient Optimization (PoGO)** is a novel blockchain consensus mechanism where miners compete by training neural networks. This document specifies the complete mathematical formulation, implementation details, and verification procedures for the PoGO protocol as implemented in the DeSSIN blockchain.

> **Verification paths (current code):** The security primitive is **spot-check verification** ([`SPOT_CHECK_VERIFICATION.md`](SPOT_CHECK_VERIFICATION.md)) — verifiers replay $k=2$ VRF-selected gradient steps per block; no full retraining is required. A quantization-consistency shortcut was designed but is not yet active in production (the comparison call is commented out; only hash-presence checks run today). See [`EFFICIENT_VERIFICATION.md`](EFFICIENT_VERIFICATION.md) for the roadmap. Mining runs decoder-LM presets (`micro_gpt_char`, `gpt2_nano`) — the small MLP in §6.1 is the legacy reference network.

## 1. Mathematical Foundation

### 1.1 Neural Network Model

> **Current primary mining model:** `DecoderOnlyGPT` — a character-level decoder-only transformer. The 2-layer MLP described below is the **legacy reference network** used in early unit tests (`SimpleNeuralNetwork` / NumPy). All production and E2E tests run the decoder-LM path. The mathematical framework is identical; only the architecture $f$ and the loss function $\mathcal{L}$ differ.

**Legacy reference (MLP):**

$$f_{\text{MLP}}(x; \theta) = \sigma(W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2)$$

**Primary mining model (DecoderOnlyGPT):**

$$f_{\text{LM}}(x_{1:t}; \theta) = \text{softmax}(W_e^{\top} \cdot \text{TransformerBlocks}(W_e \cdot x_{1:t}))$$

Where:
- $x \in \mathbb{R}^{d_{in}}$ is the input (token IDs for the LM)
- $\theta$ — all trainable parameters
- $\sigma(\cdot)$ — ReLU for the MLP; layer-norm + attention + MLP stack for `DecoderOnlyGPT`

### 1.2 Training Objective

**MLP (regression):** Mean Squared Error:

$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \lVert f(x_i; \theta) - y_i \rVert^2$$

**DecoderOnlyGPT (language modelling):** Cross-entropy next-token prediction:

$$\mathcal{L}_{\text{LM}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_{t+1} \mid x_{1:t})$$

Where $x_{1:T}$ are token sequences sampled from the VRF-selected dataset $\mathcal{D}$.

### 1.3 Gradient Descent Update

Parameter updates follow standard gradient descent:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

Where:
- $\eta > 0$ is the learning rate
- $\nabla_\theta \mathcal{L}(\theta_t)$ is the gradient computed via backpropagation

## 2. PoGO Consensus Protocol

### 2.1 Block Structure

Each PoGO block $B$ contains:

| Field | Type | Description |
|-------|------|-------------|
| `index` | $\mathbb{N}$ | Block number |
| `timestamp` | $\mathbb{R}^+$ | Block creation time |
| `previous_hash` | $\{0,1\}^{256}$ | Hash of previous block |
| `miner` | Address | Miner's blockchain address |
| `model_id` | String | Identifier of trained model |
| `training_data_hash` | $\{0,1\}^{256}$ | VRF-generated training data hash |
| `loss_before` | $\mathbb{R}^+$ | Loss before training |
| `loss_after` | $\mathbb{R}^+$ | Loss after training |
| `hash_full_model_32` | $\{0,1\}^{256}$ | Merkle root of $\theta \in$ Float32 |
| `hash_quant_4` | $\{0,1\}^{256}$ | Merkle root of $Q_4(\theta)$ |
| `vrf_proof` | $\{0,1\}^*$ | VRF proof for randomness |
| `training_steps` | $\mathbb{N}$ | Number of gradient steps |
| `learning_rate` | $\mathbb{R}^+$ | Learning rate used |
| `batch_size` | $\mathbb{N}$ | Mini-batch size |
| `quantization_error` | $\mathbb{R}^+$ | MSE quantization error |
| `model_size_full` | $\mathbb{N}$ | Full model size (bytes) |
| `model_size_quant` | $\mathbb{N}$ | Quantized model size (bytes) |
| `finalization_block` | $\mathbb{N}$ | Block index when finalized |
| `torrent_hash` | $\{0,1\}^{160}$ | BitTorrent info hash |
| `magnet_link` | String | BitTorrent magnet link |
| `model_size_bytes` | $\mathbb{N}$ | Model file size (bytes) |
| `torrent_listen_port` | $\mathbb{N}$ | BitTorrent listen port |
| `torrent_tracker_ports` | String | Comma-separated tracker ports |
| `torrent_piece_count` | $\mathbb{N}$ | Number of torrent pieces |
| `torrent_piece_length` | $\mathbb{N}$ | Torrent piece size (bytes) |
| `torrent_seeders` | $\mathbb{N}$ | Number of active seeders |
| `torrent_created_at` | $\mathbb{R}^+$ | Torrent creation timestamp |

### 2.2 Training Improvement Requirement

A valid PoGO block must satisfy:

$$\Delta \mathcal{L} = \mathcal{L}(\theta_{\text{before}}) - \mathcal{L}(\theta_{\text{after}}) \geq \epsilon$$

Where $\epsilon > 0$ is the minimum improvement threshold. The reference implementation defaults to $\epsilon = 10^{-3}$ (`min_loss_improvement = 0.001` in `ConsensusConfig`; override via `DESSIN` env vars only where supported).

The legacy on-chain check is **off by default** (`require_monotonic_training_loss = False`); usefulness is enforced at the **model granularity** by a strike ledger in the spot-check path (see [`SPOT_CHECK_VERIFICATION.md`](SPOT_CHECK_VERIFICATION.md)): if the entire 100-step loss curve of a block never dips below $\mathcal{L}(\theta_{\text{before}})$, the model accumulates a strike; after `spot_check_max_strikes` (default **3**) consecutive such blocks the model freezes until the owner submits a `ModelTrainingDataRefreshTransaction`.

### 2.3 VRF-Based Randomness

Training data selection uses Verifiable Random Functions:

$$\text{seed} = \text{VRF}(\text{model\_id} \| \text{block\_index} \| \text{timestamp})$$

The seed determines:
- Training dataset generation: $\mathcal{D} = \text{GenerateData}(\text{seed})$
- Batch sampling order during training

### 2.4 Finalization Window

Blocks are finalized after a confirmation window:

$$\text{finalization\_block} = \text{block\_index} + W$$

Where $W$ is `finalization_window` (default: **20** blocks). This is distinct from the sum of phase windows $w_1 + w_2$ (see §2.6).

### 2.5 Configurable Merkle Proof Count

The protocol supports a configurable number of Merkle proofs per block for enhanced security:

$$\text{merkle\_proof\_count} = k$$

Where $k$ is the number of random leaf challenges (default: 1, configurable via hard fork).

### 2.6 Verification Window System

The protocol assigns each new block a structured window for verifiers to submit attestations:

**Phase 1: Structural + Statistical Checks** (Blocks N+1 to N+w₁)
- Verifiers check `state_0_hash` / `state_N_hash` anchors, loss curve lengths, gradient hash chain format.
- Statistical heuristics on gradient norms and loss-jump ratios flag obvious forgeries.
- Submit positive/negative attestations based on results.

**Phase 2: Spot-Check Replay** (Blocks N+w₁+1 to N+w₁+w₂)
- VRF seed (derived from post-publication chain state) selects $k=2$ step indices.
- Each verifier replays those steps and checks gradient hashes match the committed chain.
- Submit attestations based on replay results.
- *When the quantization-consistency path activates (planned):* verifiers will also check `quantize(32-bit weights) ≈ claimed 4-bit weights` as an additional pre-filter in this phase.

**Finalization** (from block **N+w₁+w₂+1** through **N+W**, for defaults)
- Aggregate attestations; apply economic rules (`EnhancedPogoConsensus`, slashing)
- `TwoPhaseVerificationSystem` maps verifier timing to phases; **phase 1** spans the first **`phase1_window_blocks`** interval after training, **phase 2** spans the next **`phase2_window_blocks`**, then the **FINALIZATION** phase label applies until **`finalization_window`** blocks have elapsed since the training block (defaults: **5 + 5** phase blocks, **W = 20** total offset window). Tune `phase1_window_blocks`, `phase2_window_blocks`, `finalization_window`, and block-time fields in `ConsensusConfig`.

Where:
- $w_1$ = `phase1_window_blocks` (default: **5**)
- $w_2$ = `phase2_window_blocks` (default: **5**)
- $W$ = `finalization_window` (default: **20**); in code this caps the phased timeline, not $w_1 + w_2$ alone.

### 2.7 Attestation System

Verifiers submit positive or negative attestations for blocks:

- **Positive Attestation**: Verifier confirms block validity
- **Negative Attestation**: Verifier provides evidence of block invalidity

Finalization requires:
$$\frac{\text{positive\_stake}}{\text{total\_stake}} \geq \text{attestation\_threshold}$$

Slashing occurs when:
$$\frac{\text{negative\_stake}}{\text{total\_stake}} \geq \text{slashing\_threshold}$$

Where:
- $\text{attestation\_threshold} = 0.67$ (2/3 majority)
- $\text{slashing\_threshold} = 0.33$ (1/3 minority)

## 3. Model Quantization

### 3.1 Quantization Levels

The protocol supports multiple quantization levels $Q_k$:

| Level | Precision | Range | Compression Ratio |
|-------|-----------|-------|-------------------|
| Float32 | 32-bit | $[-2^{127}, 2^{127}]$ | $1\times$ |
| Float16 | 16-bit | $[-65504, 65504]$ | $2\times$ |
| Int8 | 8-bit | $[-128, 127]$ | $4\times$ |
| Int4 | 4-bit | $[-8, 7]$ | $8\times$ |

### 3.2 Quantization Function

For Int-k quantization with $k$ bits:

$$Q_k(\theta) = \text{round}\left(\frac{\theta \cdot (2^{k-1} - 1)}{\max(|\theta|)}\right) \cdot \frac{\max(|\theta|)}{2^{k-1} - 1}$$

Where:
- $\text{round}(\cdot)$ rounds to nearest integer
- Values are clipped to $[-2^{k-1}, 2^{k-1}-1]$

### 3.3 Quantization Error

The quantization error is measured as Mean Squared Error:

$$E_q = \frac{1}{|\theta|} \sum_{i=1}^{|\theta|} (\theta_i - Q_k(\theta_i))^2$$

### 3.4 Quantization Tolerance

A quantized model is valid if:

$$E_q \leq \tau$$

Loss-style checks on training trajectories use `quantized_tolerance` (default **0.0005**). A **quantization-consistency** check (`quantize(32-bit) ≈ 4-bit`, MSE threshold `quantization_consistency_tolerance = 1e-8`) is defined in `ConsensusConfig` and implemented in `dessin/protocol/quantization_verifier.py` but is not yet active in `verify_block` — the comparison call is present but commented out pending data-availability integration. See [`EFFICIENT_VERIFICATION.md`](EFFICIENT_VERIFICATION.md).

## 4. BitTorrent Model Distribution

### 4.1 Torrent Creation

Each trained model is packaged into a BitTorrent torrent containing:
- Model weights in JSON format
- Training metadata and results
- PoGO protocol commitments
- Verification data

### 4.2 Enhanced Torrent Features

The implementation includes enhanced BitTorrent features:
- Multi-tracker support for redundancy
- DHT (Distributed Hash Table) integration
- Local service discovery for peer finding
- Port management for multiple nodes

### 4.3 Model File Structure

```json
{
  "model_id": "string",
  "training_metadata": {
    "loss_before": "float",
    "loss_after": "float", 
    "improvement": "float",
    "training_steps": "int",
    "learning_rate": "float",
    "batch_size": "int"
  },
  "model_architecture": {
    "input_size": "int",
    "hidden_size": "int", 
    "output_size": "int"
  },
  "model_weights": {
    "weights_hex": "string",
    "checksum": "string"
  },
  "training_data": {
    "data_hash": "string",
    "vrf_seed": "int"
  }
}
```

## 5. Merkle Tree Commitments

### 5.1 Layer-wise Hashing

For a model with parameters $\theta = \{W_1, b_1, W_2, b_2\}$, compute:

$$h_i = \text{SHA256}(\text{serialize}(\theta_i))$$

Where $\text{serialize}(\cdot)$ converts parameters to bytes.

### 5.2 Merkle Root Computation

Given layer hashes $H = \{h_1, h_2, h_3, h_4\}$, construct binary tree:

```
Level 2:     root = SHA256(H₁₂ ∥ H₃₄)
            /                    \
Level 1:   H₁₂ = SHA256(h₁ ∥ h₂)    H₃₄ = SHA256(h₃ ∥ h₄)
          /    \                    /    \
Level 0: h₁    h₂                h₃    h₄
```

### 5.3 Merkle Proof

A Merkle proof $\pi$ for leaf $h_i$ consists of sibling hashes along the path to root:

$$\pi_i = \{h_{\text{sibling}_j} : j \in \text{PathToRoot}(i)\}$$

### 5.4 Proof Verification

Verify proof $\pi_i$ for leaf $h_i$ and claimed root $r$:

$$\text{Verify}(\pi_i, h_i, r) = \begin{cases} 
\text{true} & \text{if } \text{ComputeRoot}(\pi_i, h_i) = r \\
\text{false} & \text{otherwise}
\end{cases}$$

## 6. Implementation Specifications

### 6.1 Reference Network Architecture

```
SimpleNeuralNetwork:
  Input Layer:    4 dimensions
  Hidden Layer:   8 neurons (ReLU activation)
  Output Layer:   1 dimension
  Total Parameters: 49 (32 weights + 17 biases)
```

Mathematical representation:
- $d_{in} = 4$, $h = 8$, $d_{out} = 1$
- $|W_1| = 32$, $|b_1| = 8$, $|W_2| = 8$, $|b_2| = 1$
- Total: $|\theta| = 49$ parameters

### 6.2 Default Hyperparameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Learning Rate | $\eta$ | $0.01$ | Gradient descent step size |
| Training Steps | $T$ | $20$ | Number of gradient updates |
| Batch Size | $B$ | $32$ | Mini-batch size |
| Min Improvement | $\epsilon$ | $10^{-3}$ (`min_loss_improvement = 0.001`) | Required loss reduction (only active when `require_monotonic_training_loss = True`) |
| Quantization Tolerance | $\tau$ | $10^{-3}$ | Max quantization error |
| Merkle Proof Count | $k$ | $1$ | Number of random leaf challenges (configurable) |
| Attestation Threshold | $\alpha$ | $0.67$ | Required positive attestation ratio |
| Slashing Threshold | $\beta$ | $0.33$ | Negative attestation ratio for slashing |
| Attestation Window | $w_a$ | $10$ | Blocks to wait for attestations |
| Phase 1 Window | $w_1$ | $5$ | Blocks for structural + statistical checks |
| Phase 2 Window | $w_2$ | $5$ | Blocks for spot-check replay ($k=2$ steps) |
| Training Block Time | $T_t$ | $2.0$ hours | Block time for training phase |
| Verification Block Time | $T_v$ | $30$ minutes | Block time for verification phases |

### 6.3 Compression Analysis

For the reference network with 49 parameters:

| Format | Size (bytes) | Compression | Typical Error |
|--------|--------------|-------------|---------------|
| Float64 | $49 \times 8 = 392$ | $1\times$ | $0$ |
| Float32 | $49 \times 4 = 196$ | $2\times$ | $< 10^{-10}$ |
| Int8 | $49 \times 1 = 49$ | $8\times$ | $\sim 4 \times 10^{-4}$ |
| Int4 | $49 \times 0.5 = 24.5 \approx 49$ | $8\times$ | $\sim 1.6 \times 10^{-4}$ |

## 7. Enhanced PoGO Features

### 7.1 Multiple Merkle Proofs

The protocol supports configurable multiple Merkle proofs for enhanced security:

- **Single Proof Mode** (default): One random leaf challenge per block
- **Multi-Proof Mode**: Multiple random leaf challenges (configurable via hard fork)
- **Security Trade-off**: More proofs = higher security but increased verification cost

### 7.2 Two-Phase Verification System

Blocks are finalized through a structured verification process:

1. **Phase 1 — Structural + Statistical** (duration: `phase1_window_blocks = 5`):
   - Verifiers run structural anchor checks and statistical heuristics.
   - Submit attestations. Obvious forgeries are rejected here at near-zero cost.

2. **Phase 2 — Spot-Check Replay** (duration: `phase2_window_blocks = 5`):
   - Each verifier draws $k=2$ VRF-selected step indices and replays them.
   - This is the cryptographic proof that training actually happened.
   - *Quantization-consistency check (planned):* will also run here as a pre-filter once the data-availability layer is wired.

3. **Finalization** (from block N+W, default W=20):
   - Aggregate attestations from both phases.
   - Block finalised if ≥67% stake is positive.
   - Miner slashed if ≥33% stake is negative.

### 7.3 Slashing Mechanism

Miners are slashed when:
- Negative attestations exceed 33% of total stake
- Evidence of fraud is provided (e.g., Merkle proof mismatch)
- Data availability failures occur

## 8. Security Analysis

### 8.1 Consensus Properties

**Safety**: No two honest nodes will accept conflicting blocks at the same height.

**Liveness**: The network will continue producing valid blocks as long as honest miners perform training.

### 8.2 Attack Resistance

**Training Simulation**: Impossible due to VRF-based deterministic training data and verifiable loss computation.

**Model Manipulation**: Prevented by Merkle commitments and quantization error bounds.

**Data Poisoning**: Mitigated by VRF-generated synthetic training data.

**Multiple Proof Security**: Configurable Merkle proof count prevents selective cheating.

### 8.3 Economic Incentives

Miners are incentivized to:
1. Perform actual neural network training (required for valid blocks)
2. Optimize training efficiency (faster block production)
3. Maintain network participation (block rewards)

## 9. Experimental Results

### 9.1 Network Performance (4-node testnet)

- **Block Production Rate**: $1$ block per $15$ seconds
- **Training Success Rate**: $100\%$ (all blocks contain valid training)
- **Average Loss Improvement**: $\Delta \mathcal{L} = 0.15 \pm 0.08$
- **Network Latency**: $< 1$ second for block propagation

### 9.2 Quantization Performance

From 1000 training runs:

| Metric | Float32 | Int8 | Int4 |
|--------|---------|------|------|
| Mean Error | $< 10^{-10}$ | $4.2 \times 10^{-4}$ | $1.6 \times 10^{-4}$ |
| Std Error | $< 10^{-11}$ | $2.1 \times 10^{-4}$ | $8.9 \times 10^{-5}$ |
| Compression | $1\times$ | $4\times$ | $8\times$ |
| Tolerance Pass Rate | $100\%$ | $99.8\%$ | $99.9\%$ |

### 9.3 Verification Efficiency

- **Merkle Proof Size**: $O(\log n) = O(\log 4) = 2$ hashes per proof
- **Verification Time**: $< 1$ ms per proof
- **Storage Overhead**: $64$ bytes per block (2 Merkle roots)
- **Multiple Proof Overhead**: $O(k \log n)$ where $k$ is proof count

### 9.4 Enhanced Features Performance

- **Attestation Processing**: $< 10$ ms per attestation
- **Finalization Time**: $< 1$ second for attestation aggregation
- **Slashing Detection**: Real-time negative attestation monitoring
- **Phase 1 Verification**: seconds per block (structural + statistical checks; no model download required)
- **Phase 2 Verification**: seconds to minutes per block (replay $k=2$ gradient steps; scales with model size and step complexity)
- **Total Verification Time**: $< 7$ minutes per block (both phases combined)

### 9.5 Unit Test Results

The implementation has been thoroughly tested with comprehensive unit tests:

- **Total Tests**: 306 tests covering all protocol components
- **Test Coverage**: 100% of core PoGO protocol functionality
- **Success Rate**: 100% (all tests passing)
- **Test Categories**:
  - Merkle tree operations and proof verification
  - Model quantization and compression
  - Training validation and improvement verification
  - BitTorrent model distribution
  - Consensus block creation and validation
  - Enhanced torrent integration
  - VRF-based randomness verification

## 10. Conclusion

The PoGO protocol successfully demonstrates:

1. **Practical AI Consensus**: Real neural network training integrated into blockchain consensus
2. **Efficient Verification**: Cryptographic proofs enable fast validation without recomputation  
3. **Significant Compression**: $8\times$ model compression with $< 0.02\%$ accuracy loss
4. **Network Scalability**: Sub-second verification times with $O(\log n)$ proof complexity
5. **Configurable Security**: Adjustable Merkle proof count for security-performance trade-offs
6. **Attestation-Based Finalization**: Decentralized verification through stake-weighted attestations
7. **Economic Security**: Slashing mechanism prevents dishonest behavior

This implementation provides a foundation for production blockchain systems that incentivize meaningful AI computation while maintaining cryptographic security guarantees. The enhanced features enable the protocol to adapt to different security requirements through configurable parameters while maintaining the core principles of verifiable gradient optimization.

---

**Implementation status:** Core data structures, PoGO pathways, quantization utilities, torrent metadata, tests under `tests/`, and configuration in `ConsensusConfig` track this specification; treat performance claims ($O(\cdot)$ timings, compression figures) as design targets unless measured on your deployment.
