"""Algorithm configuration. This remains model-based SuperMimic, not PPO."""
from dataclasses import dataclass, field
from supermimic.utils.paths import REPO_ROOT


@dataclass
class WorldModelCfg:
    wm_window: int = 12
    wm_prediction: str | None = None  # new: state_delta; resume: saved choice
    wm_batches: int = 10
    wm_batch: int = 2048  # per GPU


@dataclass
class ActorCfg:
    horizon: int = 48
    pi_batch: int = 4096  # per GPU
    gamma: float = 0.995
    lam: float = 0.95
    detach_a_prev: bool = False


@dataclass
class CriticCfg:
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_tau: float = 0.05
    no_critic: bool = False
    critic_layer_norm: bool | None = None  # new: True; old checkpoint: False


@dataclass
class ReplayCfg:
    buffer_steps: int = 1000
    utd: float = 0.4
    warmup_steps: int = 64
    wm_only_iters: int = 500


@dataclass
class OptimizerCfg:
    # Adam parameters/architecture are unchanged; this refactor adds no Muon switch.
    lr: float = 2e-4
    lr_decay_iters: int = 250000
    amp: str = "bf16"


@dataclass
class SuperMimicRunnerCfg:
    iterations: int = 320000
    seed: int = 0
    ckpt_every: int = 2000
    log_every: int = 50
    log_dir: str = str(REPO_ROOT / "logs" / "run")
    resume: str | None = None
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    actor: ActorCfg = field(default_factory=ActorCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)
