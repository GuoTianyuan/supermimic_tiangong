"""Adapt readable task/agent declarations to the stable runner CLI/checkpoint API."""
from dataclasses import asdict, fields, is_dataclass, replace
import os
from supermimic.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, RandomizationRangesCfg, randomization_ranges
from supermimic.tasks.tracking.agents.supermimic_cfg import SuperMimicRunnerCfg
from supermimic.tasks.tracking.config import from_task_cfg, reward_cfg, checkpoint_reward_cfg, TRACKING_TERMS, PENALTY_TERMS

CONFIG_NAMES = ("default", "tiangong3_full_wbc")


def load_config(name="default"):
    if name == "tiangong3_full_wbc":
        from supermimic.tasks.tracking.configs.tiangong3 import full_wbc
        return full_wbc()
    if name != "default":
        raise ValueError(f"Unknown training config: {name}")
    env, runner = TrackingEnvCfg(), SuperMimicRunnerCfg()
    env.scene.robot = os.environ.get("SUPERMIMIC_ROBOT", "g1")
    env.events.ranges = randomization_ranges(env.scene.robot)
    return env, runner


def cli_defaults(name="default"):
    """Only the adapter flattens groups; leaves have the original CLI names."""
    env, runner = load_config(name)
    result = {}
    def visit(obj):
        for f in fields(obj):
            value = getattr(obj, f.name)
            if f.name in ("rewards", "terminations", "ranges"):
                continue
            if is_dataclass(value):
                visit(value)
            else:
                if f.name in result:
                    raise ValueError(f"Ambiguous CLI field: {f.name}")
                result[f.name] = value
    visit(env)
    visit(runner)
    result.update(root_pos_z_weight=None, body_pos_z_mode=None)
    return result


def resolve_reward_config(args, checkpoint=None):
    """Resume the saved task objective, including legacy checkpoints.

    Refusing a conflicting preset is intentional: changing reward must be an explicit
    separate experiment, not a side effect of a config-layout migration.
    """
    name = getattr(args, "config", "default")
    env, _ = load_config(name)
    cfg = (reward_cfg(args.dynamics) if name == "default"
           else from_task_cfg(env.rewards, env.terminations))
    overrides = {k: getattr(args, k) for k in ("root_pos_z_weight", "body_pos_z_mode")
                 if getattr(args, k, None) is not None}
    cfg = replace(cfg, **overrides).validate()
    if checkpoint is not None:
        old = checkpoint["args"]
        saved = checkpoint_reward_cfg(old)
        if name != "default" and asdict(saved) != asdict(cfg):
            raise ValueError("Preset rewards/terminations differ from checkpoint; resume with --config default "
                             "to retain its objective, or start a new experiment")
        if any(getattr(saved, key) != value for key, value in overrides.items()):
            raise ValueError("Cannot resume with different reward settings; start a separate reward ablation")
        cfg = saved
    for key in ("root_pos_z_weight", "body_pos_z_mode"):
        setattr(args, key, getattr(cfg, key))
    args.reward_cfg = asdict(cfg)
    return cfg


def saved_randomization_ranges(saved):
    """Restore complete saved ranges, or the historical robot/dynamics defaults."""
    default = randomization_ranges(saved.get("robot") or "g1", saved.get("dynamics") or "default")
    values = saved.get("dr_ranges")
    if values is None:
        return default
    unknown = set(values) - {f.name for f in fields(RandomizationRangesCfg)}
    if unknown:
        raise ValueError(f"Unknown DR range fields: {sorted(unknown)}")
    merged = {**asdict(default), **values}
    for name, value in merged.items():
        if isinstance(value, list):
            merged[name] = tuple(value)  # JSON round-trip
    return RandomizationRangesCfg(**merged).validate()


def requested_randomization_ranges(args):
    name = getattr(args, "config", "default")
    if name == "default":
        return randomization_ranges(args.robot, getattr(args, "dynamics", None) or "default")
    env = load_config(name)[0]
    declared = env.events.ranges.validate()
    original = randomization_ranges(env.scene.robot, env.scene.dynamics or "default")
    selected = randomization_ranges(args.robot, getattr(args, "dynamics", None) or "default")
    # CLI robot/dynamics overrides also select their historical DR defaults;
    # values explicitly customized in the task preset remain authoritative.
    return replace(declared, **{f.name: getattr(selected, f.name)
                                for f in fields(declared)
                                if getattr(declared, f.name) == getattr(original, f.name)}).validate()


def resolve_randomization_config(args, checkpoint=None):
    cfg = requested_randomization_ranges(args)
    if checkpoint is not None:
        saved = saved_randomization_ranges(checkpoint["args"])
        if checkpoint["args"].get("dr_ranges") is not None:
            if getattr(args, "config", "default") != "default" and cfg != saved:
                raise ValueError("Preset DR ranges differ from checkpoint; resume with --config default to retain them")
            cfg = saved
        # Old checkpoints had no adjustable range config: retain their original
        # dynamics-preset/CLI behavior (including intentionally changing dynamics).
    args.dr_ranges = asdict(cfg)
    return cfg


def config_snapshot(args):
    """Read the actual namespace back into groups, including CLI overrides.

    Before setup, nulls mean 'resolve from checkpoint or the documented fresh-run
    default'. After setup, timing/normalization choices and rewards are resolved.
    """
    env, runner = load_config(getattr(args, "config", "default"))
    def update(obj):
        for f in fields(obj):
            value = getattr(obj, f.name)
            if f.name in ("rewards", "terminations", "ranges"):
                continue
            if is_dataclass(value):
                update(value)
            elif hasattr(args, f.name):
                setattr(obj, f.name, getattr(args, f.name))
    update(env)
    update(runner)
    env.events.ranges = (saved_randomization_ranges(vars(args)) if getattr(args, "dr_ranges", None) is not None
                         else requested_randomization_ranges(args))
    saved = getattr(args, "reward_cfg", None)
    if saved is not None:
        cfg = reward_cfg(getattr(args, "dynamics", None) or "default", saved)
    elif getattr(args, "config", "default") == "default":
        cfg = reward_cfg(getattr(args, "dynamics", None) or "default")
    else:
        cfg = from_task_cfg(env.rewards, env.terminations)
    if saved is None:
        cfg = replace(cfg, **{k: getattr(args, k) for k in ("root_pos_z_weight", "body_pos_z_mode")
                             if getattr(args, k, None) is not None}).validate()
    env.rewards.root_pos_z_weight = cfg.root_pos_z_weight
    env.rewards.body_pos_z_mode = cfg.body_pos_z_mode
    env.rewards.body_point_offsets = cfg.body_point_offsets
    for name in TRACKING_TERMS:
        term = getattr(env.rewards, name)
        term.weight, term.std = getattr(cfg, "w_" + name), getattr(cfg, name + "_std")
    for name in PENALTY_TERMS:
        getattr(env.rewards, name).weight = getattr(cfg, "w_" + name)
    env.rewards.soft_limit_factor = cfg.soft_limit_factor
    env.terminations.term_height, env.terminations.term_tilt = cfg.term_height, cfg.term_tilt
    result = {"config": getattr(args, "config", "default"), "env": asdict(env), "runner": asdict(runner)}
    result["resolved"] = saved is not None and hasattr(args, "simulation_timing")
    result["runtime"] = {key: getattr(args, key) for key in
                         ("world_size", "simulation_timing", "self_collisions", "goal_offsets", "max_step_reward")
                         if hasattr(args, key)}
    return result
