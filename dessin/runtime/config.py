"""
Configuration settings for the DeSSIN blockchain network.
"""

from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class NetworkConfig:
    """Network configuration for DeSSIN node"""
    max_peers: int = 10
    port: Optional[int] = None
    host: str = "127.0.0.1"
    local_discovery: bool = True
    use_compression: bool = True
    # Optional short id for console lines when several nodes share one process (e.g. ``node_0``).
    # When unset, ``pretty_console`` falls back to ``p{port}`` when ``port`` is set.
    log_node_label: Optional[str] = None
    
@dataclass
class ConsensusConfig:
    """PoGO consensus configuration"""
    finalization_window: int = 20  # w blocks for finalization
    min_loss_improvement: float = 0.001  # epsilon when require_monotonic_training_loss is enabled
    require_monotonic_training_loss: bool = False  # True: mining salts + verify_block require Δloss ≥ min_loss_improvement (legacy)
    quantized_tolerance: float = 0.0005  # epsilon_quant for 4-bit models
    attestation_threshold: float = 0.67  # 2/3 threshold for positive attestations
    block_time_minutes: float = 120.0  # mining cadence (default 2h); as low as 0.25 (15s) when min_block_time_seconds allows; DESSIN_BLOCK_TIME_MINUTES or legacy HOURS×60
    merkle_leaf_size_mb: int = 10  # size of each Merkle tree leaf in MB
    max_merkle_proofs: int = 8  # hard cap on proofs per block (see enhanced_pogo_consensus)
    # Independent random leaf proofs bundled per training block; ↑ reduces chance a bad model skips audit (linear verify cost, max 8)
    merkle_proof_count: int = 5
    attestation_window_blocks: int = 10  # blocks to wait for attestations before finalization
    slashing_threshold: float = 0.33  # threshold of negative attestations to trigger slashing
    
    # Two-phase verification timing
    phase1_window_blocks: int = 5  # blocks for Phase 1 (quantized model verification)
    phase2_window_blocks: int = 5  # blocks for Phase 2 (random leaf challenge)
    # When True, phase completion/progress use chain height vs phase windows (see two_phase_verification).
    verification_phase_completion_in_blocks: bool = False
    verification_block_time_minutes: int = 30  # wall-clock phase duration when verification_phase_completion_in_blocks is False (minutes)
    training_block_time_minutes: float = 120.0  # training-phase cadence; DESSIN_TRAINING_BLOCK_TIME_MINUTES or legacy *_HOURS×60
    
    # Dynamic block time adjustment
    enable_dynamic_block_time: bool = True  # enable dynamic block time adjustment
    block_time_adjustment_seconds: int = 10  # seconds to adjust block time by
    min_block_time_seconds: int = 15  # floor for dynamic/clamp (15s = 0.25 min mining cadence for tests)
    max_block_time_seconds: int = 3600  # maximum block time in seconds (1 hour)
    # Asymmetric dynamics: gentle shorten on slack (−5%/step); aggressive lengthen under stress (+15%/step)
    block_time_ramp_up_capacity_pct: float = 0.05   # shorten interval this fraction when verification is fast (slack path)
    block_time_ramp_down_capacity_pct: float = 0.15 # lengthen interval this fraction when stressed / slow validation (stress path)
    block_time_slack_step_multiplier: float = 1.45  # scales additive proposal steps when shortening interval
    block_time_stress_step_multiplier: float = 0.55  # scales additive proposal steps when lengthening interval
    verification_timeout_threshold: float = 0.5  # 50% of allocated time triggers adjustment
    
    # === EFFICIENT VERIFICATION (10-100x speedup!) ===
    # Instead of re-executing training, verify quantization consistency
    enable_efficient_verification: bool = True  # Use quantization consistency instead of re-execution
    quantization_consistency_tolerance: float = 1e-8  # MSE threshold for consistency check
    merkle_challenge_count: int = 5  # Number of random layer challenges (5 = good balance of speed/security)
    enable_loss_sanity_check: bool = True  # Optional loss range validation
    loss_sanity_tolerance: float = 0.2  # 20% deviation allowed in sanity check

    # === SHARED-VRF SPOT-CHECK VERIFICATION (no full retraining) ===
    # Miners publish state_0/state_N anchors + full loss curve + gradient hash chain.
    # Verifiers replay only ``spot_check_count`` step indices drawn from a post-publication
    # VRF seed; structural + statistical heuristics catch obvious fakes for free.
    enable_spot_check_verification: bool = False  # gate; off by default until trace pipeline lands in blocks
    spot_check_steps_per_block: int = 100         # exact gradient steps each training block records
    spot_check_count: int = 2                     # k = number of step replays per verifier per block
    spot_check_attestation_threshold: float = 0.66  # fraction of stake/voters that must replay-and-agree
    spot_check_max_strikes: int = 3               # model freezes after this many unproductive blocks
    # Statistical thresholds (forwarded into ``StatisticalThresholds``):
    spot_check_min_grad_norm: float = 1e-9
    spot_check_max_grad_norm: float = 1e6
    spot_check_max_loss_jump_ratio: float = 50.0
    spot_check_min_grad_loss_corr: float = -0.9

    # Salt retries per mining attempt (when require_monotonic_training_loss: sequential seeds until Δloss ≥ ε).
    # When monotonic is off, first successful finite-loss run wins. 32 is a practical default for the former;
    # for fast iteration set DESSIN_TRAINING_SAMPLE_MAX_ATTEMPTS=4–8.
    training_sample_max_attempts: int = 32
    training_failure_strikes_to_freeze: int = 3  # consecutive "no improvement" rounds freeze the model
    # Owner refresh tx: register new training data + unfreeze (fee + optional escrow rules)
    training_refresh_min_fee: float = 0.001
    training_refresh_min_escrow_when_frozen: float = 0.0  # require top-up when model was frozen

    # Bootstrap: free registration, virtual stake (non-transferable until end), max active miners, emission schedule
    bootstrap_period_blocks: int = 50  # 0 = disabled; default window length in blocks; override via DESSIN_BOOTSTRAP_PERIOD_BLOCKS
    bootstrap_virtual_stake: float = 10_000.0  # locked stake assigned at registration (not liquid during bootstrap)
    bootstrap_max_active_nodes: int = 500  # top-N by (virtual + reward stake) may mine; others slashed
    bootstrap_materialization_grant: float = 100_000.0  # minted to stake when bootstrap ends if not slashed (with virtual + carried reward stake)
    annual_emission_rate_bootstrap: float = 0.24  # prorated per block → ~24%/year of initial supply baseline
    annual_emission_rate_post_bootstrap: float = 0.12
    post_bootstrap_min_holdings_fraction: float = 0.001  # 0.1% of total emitted supply to be eligible miner
    avg_block_time_seconds: float = 7200.0  # default 2h; used with annual rates (override via DESSIN_AVG_BLOCK_TIME_SECONDS)
    # After bootstrap, protocol stake is exited via unstake; funds become liquid only after this many blocks.
    # Maturity is applied lazily when the account pays fees / transfers / other debits (see EconomicSystem).
    unstake_cooldown_blocks: int = 2
    
@dataclass
class TrainingMarketConfig:
    """
    EIP-1559-style fee market for scheduling model training tasks per block.

    Each block leader packs up to ``max_tasks_per_block`` training tasks from
    the pending queue, sorted by *effective tip* (``tip × urgency_multiplier``).
    A VRF seed breaks ties so the leader cannot cherry-pick after seeing the
    draw.  The base-fee auto-adjusts ±``base_fee_change_rate`` per block so
    that the mempool targets ``target_utilization`` of the available slots.

    Cooldowns prevent one model from monopolising every block.  An urgency
    multiplier rewards tasks that have been waiting the longest, ensuring fair
    scheduling even for low-tip tasks.
    """
    # Slot budget per block
    max_tasks_per_block: int = 4
    # Fee dynamics (EIP-1559-style)
    initial_base_fee: float = 0.01         # DESSIN; burned from deposit
    base_fee_change_rate: float = 0.125    # ±12.5% per block (same as Ethereum)
    target_utilization: float = 0.5        # target fraction of slots filled
    min_base_fee: float = 0.001            # floor; can never go below this
    max_base_fee: float = 1000.0           # ceiling
    # Urgency: grows linearly with blocks in queue so old tasks get promoted
    urgency_rate: float = 0.10             # multiplier increase per waiting block
    urgency_cap: float = 5.0              # cap so tips don't grow without bound
    # Cooldown: min gap between two training blocks for the same model
    default_cooldown_blocks: int = 5
    max_cooldown_blocks: int = 100
    # Task lifecycle
    task_expiry_blocks: int = 200          # task cancelled if not scheduled by then
    # Tip floor (0 = pure base-fee mode; owner just pays gas)
    min_tip: float = 0.0


@dataclass  
class ModelConfig:
    """Model management configuration"""
    max_model_size_gb: int = 50  # maximum model size to handle
    model_cache_dir: str = "./model_cache"
    supported_formats: List[str] = None
    quantization_bits: int = 4  # for quantized models
    max_training_steps: int = 100  # max steps per block
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["gguf"]
            
@dataclass
class TokenConfig:
    """Token economics configuration"""
    token_symbol: str = "DESSIN"
    decimals: int = 18
    initial_supply: int = 1000000  # 1M tokens
    mining_reward: float = 10.0
    base_compute_price: float = 0.01  # DESSIN per compute unit
    base_storage_price: float = 0.001  # DESSIN per GB per block
    
@dataclass
class DessinConfig:
    """Main configuration container for DeSSIN node"""
    network: NetworkConfig
    consensus: ConsensusConfig
    model: ModelConfig
    token: TokenConfig
    training_market: TrainingMarketConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.training_market is None:
            self.training_market = TrainingMarketConfig()

    @classmethod
    def default(cls) -> "DessinConfig":
        """Create default configuration"""
        return cls(
            network=NetworkConfig(),
            consensus=ConsensusConfig(),
            model=ModelConfig(),
            token=TokenConfig(),
            training_market=TrainingMarketConfig(),
        )
        
    @classmethod
    def from_env(cls) -> "DessinConfig":
        """Create configuration from environment variables"""
        config = cls.default()
        
        # Network config from env
        config.network.max_peers = int(os.getenv("DESSIN_MAX_PEERS", config.network.max_peers))
        config.network.port = int(os.getenv("DESSIN_PORT")) if os.getenv("DESSIN_PORT") else None
        config.network.host = os.getenv("DESSIN_HOST", config.network.host)
        
        # Consensus config from env (minutes primary; DESSIN_*_BLOCK_TIME_HOURS still accepted ×60)
        block_minutes_env = os.getenv("DESSIN_BLOCK_TIME_MINUTES")
        block_hours_env = os.getenv("DESSIN_BLOCK_TIME_HOURS")
        if block_minutes_env:
            config.consensus.block_time_minutes = float(block_minutes_env)
        elif block_hours_env:
            config.consensus.block_time_minutes = float(block_hours_env) * 60.0
        config.consensus.finalization_window = int(os.getenv("DESSIN_FINALIZATION_WINDOW", config.consensus.finalization_window))
        config.consensus.merkle_proof_count = int(os.getenv("DESSIN_MERKLE_PROOF_COUNT", config.consensus.merkle_proof_count))
        config.consensus.require_monotonic_training_loss = (
            os.getenv("DESSIN_REQUIRE_MONOTONIC_TRAINING_LOSS", "").strip().lower() in ("1", "true", "yes")
        )
        config.consensus.attestation_window_blocks = int(os.getenv("DESSIN_ATTESTATION_WINDOW", config.consensus.attestation_window_blocks))
        config.consensus.slashing_threshold = float(os.getenv("DESSIN_SLASHING_THRESHOLD", config.consensus.slashing_threshold))
        
        # Two-phase verification config from env
        config.consensus.phase1_window_blocks = int(os.getenv("DESSIN_PHASE1_WINDOW", config.consensus.phase1_window_blocks))
        config.consensus.phase2_window_blocks = int(os.getenv("DESSIN_PHASE2_WINDOW", config.consensus.phase2_window_blocks))
        _vphase_blocks = os.getenv("DESSIN_VERIFICATION_PHASE_COMPLETION_IN_BLOCKS")
        if _vphase_blocks is not None and str(_vphase_blocks).strip() != "":
            config.consensus.verification_phase_completion_in_blocks = str(_vphase_blocks).strip().lower() in (
                "1",
                "true",
                "yes",
            )
        config.consensus.verification_block_time_minutes = int(os.getenv("DESSIN_VERIFICATION_BLOCK_TIME_MINUTES", config.consensus.verification_block_time_minutes))
        train_minutes_env = os.getenv("DESSIN_TRAINING_BLOCK_TIME_MINUTES")
        train_hours_env = os.getenv("DESSIN_TRAINING_BLOCK_TIME_HOURS")
        if train_minutes_env:
            config.consensus.training_block_time_minutes = float(train_minutes_env)
        elif train_hours_env:
            config.consensus.training_block_time_minutes = float(train_hours_env) * 60.0
        
        # Dynamic block time config from env
        config.consensus.enable_dynamic_block_time = os.getenv("DESSIN_ENABLE_DYNAMIC_BLOCK_TIME", "true").lower() == "true"
        config.consensus.block_time_adjustment_seconds = int(os.getenv("DESSIN_BLOCK_TIME_ADJUSTMENT_SECONDS", config.consensus.block_time_adjustment_seconds))
        config.consensus.min_block_time_seconds = int(os.getenv("DESSIN_MIN_BLOCK_TIME_SECONDS", config.consensus.min_block_time_seconds))
        config.consensus.max_block_time_seconds = int(os.getenv("DESSIN_MAX_BLOCK_TIME_SECONDS", config.consensus.max_block_time_seconds))
        config.consensus.verification_timeout_threshold = float(os.getenv("DESSIN_VERIFICATION_TIMEOUT_THRESHOLD", config.consensus.verification_timeout_threshold))
        config.consensus.block_time_ramp_up_capacity_pct = float(
            os.getenv(
                "DESSIN_BLOCK_TIME_RAMP_UP_PCT",
                str(config.consensus.block_time_ramp_up_capacity_pct),
            )
        )
        config.consensus.block_time_ramp_down_capacity_pct = float(
            os.getenv(
                "DESSIN_BLOCK_TIME_RAMP_DOWN_PCT",
                str(config.consensus.block_time_ramp_down_capacity_pct),
            )
        )
        config.consensus.block_time_slack_step_multiplier = float(
            os.getenv(
                "DESSIN_BLOCK_TIME_SLACK_STEP_MULT",
                str(config.consensus.block_time_slack_step_multiplier),
            )
        )
        config.consensus.block_time_stress_step_multiplier = float(
            os.getenv(
                "DESSIN_BLOCK_TIME_STRESS_STEP_MULT",
                str(config.consensus.block_time_stress_step_multiplier),
            )
        )
        config.consensus.training_sample_max_attempts = int(
            os.getenv("DESSIN_TRAINING_SAMPLE_MAX_ATTEMPTS", str(config.consensus.training_sample_max_attempts))
        )
        config.consensus.training_failure_strikes_to_freeze = int(
            os.getenv("DESSIN_TRAINING_FAILURE_STRIKES_TO_FREEZE", str(config.consensus.training_failure_strikes_to_freeze))
        )
        config.consensus.training_refresh_min_fee = float(
            os.getenv("DESSIN_TRAINING_REFRESH_MIN_FEE", str(config.consensus.training_refresh_min_fee))
        )
        config.consensus.training_refresh_min_escrow_when_frozen = float(
            os.getenv(
                "DESSIN_TRAINING_REFRESH_MIN_ESCROW_WHEN_FROZEN",
                str(config.consensus.training_refresh_min_escrow_when_frozen),
            )
        )
        config.consensus.bootstrap_period_blocks = int(
            os.getenv("DESSIN_BOOTSTRAP_PERIOD_BLOCKS", str(config.consensus.bootstrap_period_blocks))
        )
        config.consensus.bootstrap_virtual_stake = float(
            os.getenv("DESSIN_BOOTSTRAP_VIRTUAL_STAKE", str(config.consensus.bootstrap_virtual_stake))
        )
        config.consensus.bootstrap_max_active_nodes = int(
            os.getenv("DESSIN_BOOTSTRAP_MAX_ACTIVE_NODES", str(config.consensus.bootstrap_max_active_nodes))
        )
        config.consensus.bootstrap_materialization_grant = float(
            os.getenv(
                "DESSIN_BOOTSTRAP_MATERIALIZATION_GRANT",
                str(config.consensus.bootstrap_materialization_grant),
            )
        )
        config.consensus.annual_emission_rate_bootstrap = float(
            os.getenv(
                "DESSIN_ANNUAL_EMISSION_BOOTSTRAP",
                str(config.consensus.annual_emission_rate_bootstrap),
            )
        )
        config.consensus.annual_emission_rate_post_bootstrap = float(
            os.getenv(
                "DESSIN_ANNUAL_EMISSION_POST_BOOTSTRAP",
                str(config.consensus.annual_emission_rate_post_bootstrap),
            )
        )
        config.consensus.post_bootstrap_min_holdings_fraction = float(
            os.getenv(
                "DESSIN_POST_BOOTSTRAP_MIN_FRACTION",
                str(config.consensus.post_bootstrap_min_holdings_fraction),
            )
        )
        config.consensus.avg_block_time_seconds = float(
            os.getenv(
                "DESSIN_AVG_BLOCK_TIME_SECONDS",
                str(config.consensus.avg_block_time_seconds),
            )
        )
        config.consensus.unstake_cooldown_blocks = int(
            os.getenv(
                "DESSIN_UNSTAKE_COOLDOWN_BLOCKS",
                str(config.consensus.unstake_cooldown_blocks),
            )
        )

        config.consensus.enable_spot_check_verification = (
            os.getenv("DESSIN_ENABLE_SPOT_CHECK_VERIFICATION", "").strip().lower()
            in ("1", "true", "yes")
        )
        config.consensus.spot_check_steps_per_block = int(
            os.getenv(
                "DESSIN_SPOT_CHECK_STEPS_PER_BLOCK",
                str(config.consensus.spot_check_steps_per_block),
            )
        )
        config.consensus.spot_check_count = int(
            os.getenv(
                "DESSIN_SPOT_CHECK_COUNT", str(config.consensus.spot_check_count)
            )
        )
        config.consensus.spot_check_attestation_threshold = float(
            os.getenv(
                "DESSIN_SPOT_CHECK_ATTESTATION_THRESHOLD",
                str(config.consensus.spot_check_attestation_threshold),
            )
        )
        config.consensus.spot_check_max_strikes = int(
            os.getenv(
                "DESSIN_SPOT_CHECK_MAX_STRIKES",
                str(config.consensus.spot_check_max_strikes),
            )
        )
        
        # Model config from env
        config.model.model_cache_dir = os.getenv("DESSIN_MODEL_CACHE", config.model.model_cache_dir)
        config.model.max_model_size_gb = int(os.getenv("DESSIN_MAX_MODEL_SIZE_GB", config.model.max_model_size_gb))

        # Training market config from env
        _tm = config.training_market
        if os.getenv("DESSIN_MARKET_MAX_TASKS_PER_BLOCK"):
            _tm.max_tasks_per_block = int(os.getenv("DESSIN_MARKET_MAX_TASKS_PER_BLOCK"))
        if os.getenv("DESSIN_MARKET_INITIAL_BASE_FEE"):
            _tm.initial_base_fee = float(os.getenv("DESSIN_MARKET_INITIAL_BASE_FEE"))
        if os.getenv("DESSIN_MARKET_URGENCY_RATE"):
            _tm.urgency_rate = float(os.getenv("DESSIN_MARKET_URGENCY_RATE"))
        if os.getenv("DESSIN_MARKET_URGENCY_CAP"):
            _tm.urgency_cap = float(os.getenv("DESSIN_MARKET_URGENCY_CAP"))
        if os.getenv("DESSIN_MARKET_COOLDOWN_BLOCKS"):
            _tm.default_cooldown_blocks = int(os.getenv("DESSIN_MARKET_COOLDOWN_BLOCKS"))

        return config
