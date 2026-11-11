"""Rotating GPU reference library; references owned by replay remain resident.

Adapted from WBT dvt_29dof_training_base's active/resident distinction. Interpolation,
FK and return-based bin sampling remain those of this repository, not WBT's loader.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from supermimic.motions.library import BIN_FRAMES, CONTROL_DT, MotionLib
from supermimic.robots import JOINT_NAMES, ROBOT_NAME


class MotionPool(MotionLib):
    FRAME_FIELDS = ("root_p", "root_R", "root_v", "root_w", "q", "qd",
                    "body_p", "body_R", "body_v", "body_w", "root_v_local", "root_w_local", "root_gz")

    BODY_FIELDS = ("body_p", "body_R", "body_v", "body_w")
    GPU_FIELDS = tuple(k for k in FRAME_FIELDS if k not in ("body_p", "body_R", "body_v", "body_w"))

    def __init__(self, files, kin, device, *, active_ratio=.3, replace_ratio=.25, seed=0,
                 stats_cache=".cache/motion_pool", state=None):
        if not 0 < active_ratio <= 1 or not 0 <= replace_ratio <= 1:
            raise ValueError("motion active ratio must be in (0,1], replacement ratio in [0,1]")
        if not files or any(not f.endswith(".npz") for f in files):
            raise ValueError("The rotating pool needs validated named NPZ motions")
        self.files, self.kin, self.device = list(files), kin, device
        self.body_storage = "cpu"
        self._num_tracked = len(kin.tracked_idx)
        self._body_widths = tuple(self._num_tracked*n for n in (3,9,3,3))
        self.active_ratio, self.replace_ratio = active_ratio, replace_ratio
        self.rng = np.random.default_rng(seed)
        self.catalog_names = [Path(f).stem for f in files]
        lengths, fingerprint = [], hashlib.sha256()
        for f in files:
            meta = json.loads(Path(f + ".metadata.json").read_text())
            if meta["robot"] != ROBOT_NAME or meta["quaternion_format"] != "wxyz":
                raise ValueError(f"Motion metadata does not match selected robot: {f}")
            n = int(np.floor((meta["frames"] - 1) / meta["fps"] / CONTROL_DT)) + 1
            if n < 2:
                raise ValueError(f"Motion needs at least two control frames: {f}")
            lengths.append(n)
            st = os.stat(f)
            fingerprint.update(json.dumps([Path(f).name, meta, st.st_size, st.st_mtime_ns], sort_keys=True).encode())
        fingerprint.update(json.dumps(["motion_pool_v1", CONTROL_DT, ROBOT_NAME, JOINT_NAMES]).encode())
        self.catalog_hash = fingerprint.hexdigest()
        self.catalog_len = torch.tensor(lengths, dtype=torch.long)
        self.catalog_num_frames = sum(lengths)
        self.catalog_num_clips = len(files)
        nb = (self.catalog_len + BIN_FRAMES - 1) // BIN_FRAMES
        self.catalog_bin_start = torch.cumsum(nb, 0) - nb
        self.bin_returns = torch.zeros(int(nb.sum()))
        self.bin_weights = torch.full_like(self.bin_returns, 1 / len(self.bin_returns))
        self.starts_seen = torch.zeros(len(files), dtype=torch.bool, device=device)
        self.starts_drawn = torch.zeros((), dtype=torch.long, device=device)
        self.ever_active = torch.zeros(len(files), dtype=torch.bool)
        self._cache = {}
        self.resident_ids = []
        self.last_refresh_s = self.last_loaded = self.refresh_count = 0
        self.stats_cache = stats_cache
        count = max(1, round(len(files) * active_ratio))
        if state is not None:
            if state["catalog_hash"] != self.catalog_hash:
                raise ValueError("Motion catalog differs from the pool checkpoint")
            self.active_ids = list(state["active_ids"])
            self.rng.bit_generator.state = state["rng"]
            self.bin_returns.copy_(state["bin_returns"])
            self.bin_weights.copy_(state["bin_weights"])
            self.ever_active.copy_(state["ever_active"])
            self.starts_seen.copy_(state["starts_seen"])
            self.starts_drawn.copy_(state["starts_drawn"])
            self.refresh_count = state["refresh_count"]
            if len(self.active_ids) != count:
                raise ValueError("Resume with the checkpoint's motion_active_ratio")
        else:
            self.active_ids = sorted(self.rng.choice(len(files), count, replace=False).tolist())
        self._rebuild(set())

    @classmethod
    def from_dir(cls, path, kin, device, rank=0, world=1, exclude=None, **kwargs):
        files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.npz")))
        if exclude:
            files = [f for f in files if not re.search(exclude, os.path.basename(f))]
        if len(files) >= world:
            files = files[rank::world]
        return cls(files, kin, device, **kwargs)

    def _load_missing(self, ids):
        missing = [i for i in ids if i not in self._cache]
        # Only staging batches are bounded; there is no capacity cap on the active pool.
        for b in range(0, len(missing), 128):
            block = missing[b:b + 128]
            lib = MotionLib([self.files[i] for i in block], self.kin, self.device, body_storage="gpu")
            for j, cid in enumerate(block):
                s, n = int(lib.clip_start[j]), int(lib.clip_len[j])
                if n != int(self.catalog_len[cid]):
                    raise ValueError(f"Metadata frame count does not match loader: {self.files[cid]}")
                self._cache[cid] = {k: getattr(lib, k)[s:s+n].to("cpu", copy=True) for k in self.GPU_FIELDS}
                self._cache[cid]["_body"] = torch.cat([getattr(lib,k)[s:s+n].flatten(1) for k in self.BODY_FIELDS], -1).cpu()
            del lib
            loaded = min(b+len(block), len(missing))
            if loaded % 2048 == 0 or loaded == len(missing):
                print(f"[MotionPool] loaded new references {loaded}/{len(missing)}", flush=True)
        self.last_loaded = len(missing)

    def _rebuild(self, pinned):
        t0 = time.monotonic()
        ids = sorted(set(self.active_ids) | pinned)
        self._cache = {i: v for i, v in self._cache.items() if i in ids}
        # References were snapshotted by refresh(). Drop old GPU tables before loading new ones.
        for k in (*self.FRAME_FIELDS, "clip_id", "frame_bin"):
            if hasattr(self, k):
                delattr(self, k)
        self._load_missing(ids)
        self.resident_ids = ids
        self.names = [self.catalog_names[i] for i in ids]
        self.clip_len = self.catalog_len[ids].to(self.device)
        self.clip_end = self.clip_len.cumsum(0) - 1
        self.clip_start = self.clip_end + 1 - self.clip_len
        self.num_frames = int(self.clip_len.sum())
        for k in self.GPU_FIELDS:
            setattr(self, k, torch.cat([self._cache[i][k] for i in ids]).to(self.device))
        self.clip_id = torch.repeat_interleave(torch.arange(len(ids), device=self.device), self.clip_len)
        # A single (clip,local-bin) identity survives every resident-layout rebuild.
        bc, bs, bg = [], [], []
        for j, cid in enumerate(ids):
            for local in range(math.ceil(int(self.catalog_len[cid]) / BIN_FRAMES)):
                bc.append(j)
                bs.append(int(self.clip_start[j]) + local * BIN_FRAMES)
                bg.append(int(self.catalog_bin_start[cid]) + local)
        self.bin_clip = torch.tensor(bc, device=self.device)
        self.bin_start = torch.tensor(bs, device=self.device)
        self.bin_global = torch.tensor(bg, device=self.device)
        self.num_bins = len(bs)
        local = torch.arange(self.num_frames, device=self.device) - self.clip_start[self.clip_id]
        starts = ((self.clip_len + BIN_FRAMES - 1) // BIN_FRAMES).cumsum(0)
        starts -= (self.clip_len + BIN_FRAMES - 1) // BIN_FRAMES
        self.frame_bin = starts[self.clip_id] + local // BIN_FRAMES
        self.resident_global = torch.tensor(ids, device=self.device)
        active = torch.zeros(len(self.files), dtype=torch.bool, device=self.device)
        active[self.active_ids] = True
        self.active_bins = torch.where(active[self.resident_global[self.bin_clip]])[0]
        self.ever_active[self.active_ids] = True
        # Reference lengths change after a refresh. Keep compiled indexing polymorphic.
        if torch.device(self.device).type == "cuda":
            for k in (*self.GPU_FIELDS, "clip_id", "clip_start", "clip_end", "clip_len", "frame_bin"):
                torch._dynamo.mark_dynamic(getattr(self, k), 0)
        self._host_clip_end = self.clip_end.cpu()
        self._host_clip_start = self.clip_start.cpu()
        self.last_refresh_s = time.monotonic() - t0

    @torch.compiler.disable
    def body_reference(self, f, positions_only=False):
        """Gather cached float32 bodies by clip; avoid duplicating the full CPU library.

        Pool rotation preserves the cached values. Only this transfer batch is pinned,
        and reward/critic kernels receive the GPU result outside torch.compile.
        """
        index = f.detach().to("cpu").reshape(-1)
        width = self._body_widths[0] if positions_only else sum(self._body_widths)
        host = torch.empty((index.numel(), width), dtype=torch.float32,
                           pin_memory=torch.device(self.device).type == "cuda")
        clips = torch.bucketize(index, self._host_clip_end, right=False)
        order = clips.argsort()
        unique, counts = torch.unique_consecutive(clips[order], return_counts=True)
        offset = 0
        for local_clip, count in zip(unique.tolist(), counts.tolist()):
            rows = order[offset:offset+count]
            local_frames = index[rows] - self._host_clip_start[local_clip]
            cached = self._cache[self.resident_ids[local_clip]]["_body"][:, :width]
            host.index_copy_(0, rows, cached.index_select(0, local_frames))
            offset += count
        batch = host.to(self.device, non_blocking=True)
        if positions_only:
            return batch.reshape(*f.shape, self._num_tracked, 3), None, None, None
        p, R, v, w = batch.split(self._body_widths, -1)
        return (p.reshape(*f.shape, self._num_tracked, 3), R.reshape(*f.shape, self._num_tracked, 3, 3),
                v.reshape(*f.shape, self._num_tracked, 3), w.reshape(*f.shape, self._num_tracked, 3))

    def record_starts(self, bins):
        ids = self.resident_global[self.bin_clip[bins]]
        self.starts_seen[ids] = True
        self.starts_drawn.add_(bins.numel())

    def refresh(self, env, replay):
        """Run only between learner iterations, after all backward passes have finished."""
        if self.replace_ratio == 0:
            return False
        # Save sampler statistics even for clips which leave residency.
        self.bin_returns[self.bin_global.cpu()] = env.bin_G.detach().cpu()
        self.bin_weights[self.bin_global.cpu()] = env.bin_w_avg.detach().cpu()
        valid_f = replay.reference_frames()
        old_clip, old_start = self.clip_id, self.clip_start
        env_clip = old_clip[env.f]
        replay_clip = old_clip[valid_f]
        pinned_local = torch.unique(torch.cat((env_clip, replay_clip.flatten()))).cpu().tolist()
        pinned = {self.resident_ids[i] for i in pinned_local}
        available = sorted(set(range(len(self.files))) - set(self.active_ids))
        count = min(max(1, round(self.replace_ratio * len(self.files))), len(self.active_ids), len(available))
        if count:
            drop = set(self.rng.choice(self.active_ids, count, replace=False).tolist())
            add = self.rng.choice(available, count, replace=False).tolist()
            self.active_ids = sorted((set(self.active_ids) - drop) | set(add))
        ids = sorted(set(self.active_ids) | pinned)
        if ids == self.resident_ids and not count:
            return False
        old_ids = self.resident_ids
        # start_bin belongs to episode initialization, which can precede env.f by many bins.
        old_episode_bin = self.bin_global[env.start_bin].clone()
        env_local_f = env.f - old_start[env_clip]
        replay_local_f = valid_f - old_start[replay_clip]
        self._rebuild(pinned)
        index = {cid: j for j, cid in enumerate(self.resident_ids)}
        old_to_new = torch.tensor([index.get(cid, -1) for cid in old_ids], device=self.device)
        env.f.copy_(self.clip_start[old_to_new[env_clip]] + env_local_f)
        valid_f.copy_(self.clip_start[old_to_new[replay_clip]] + replay_local_f)
        global_to_bin = torch.full((len(self.bin_returns),), -1, dtype=torch.long, device=self.device)
        global_to_bin[self.bin_global] = torch.arange(self.num_bins, device=self.device)
        env.start_bin.copy_(global_to_bin[old_episode_bin])
        env.bin_G = self.bin_returns[self.bin_global.cpu()].to(self.device)
        env.bin_w_avg = self.bin_weights[self.bin_global.cpu()].to(self.device)
        self.refresh_count += 1
        return True

    def state_dict(self, env):
        self.bin_returns[self.bin_global.cpu()] = env.bin_G.detach().cpu()
        self.bin_weights[self.bin_global.cpu()] = env.bin_w_avg.detach().cpu()
        return dict(catalog_hash=self.catalog_hash, active_ids=self.active_ids,
                    rng=self.rng.bit_generator.state, bin_returns=self.bin_returns,
                    bin_weights=self.bin_weights, ever_active=self.ever_active,
                    starts_seen=self.starts_seen.cpu(), starts_drawn=self.starts_drawn.cpu(),
                    refresh_count=self.refresh_count)

    def reference_joint_stats(self):
        """Full-catalog action statistics, independent of active subset and rotation.

        Stream once and persist moments; no whole-catalog GPU tensors/FK are needed.
        """
        path = Path(self.stats_cache) / f"{self.catalog_hash}.pt"
        if path.is_file():
            moments = torch.load(path, map_location="cpu", weights_only=True)
        else:
            count = 0
            total, square = torch.zeros(29, dtype=torch.float64), torch.zeros(29, dtype=torch.float64)
            for i, f in enumerate(self.files):
                # Statistics need only q. Preserve validation/hash and the exact interpolation,
                # without decoding all body/velocity arrays for every inactive clip.
                meta = json.loads(Path(f + ".metadata.json").read_text())
                if hashlib.sha256(Path(f).read_bytes()).hexdigest() != meta["sha256"]:
                    raise ValueError(f"Motion changed; regenerate validation metadata: {f}")
                with np.load(f, allow_pickle=False) as data:
                    values = np.asarray(data["joint_pos"], dtype=np.float32)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite joint reference: {f}")
                    order = [meta["joint_names"].index(n) for n in JOINT_NAMES]
                    q = torch.from_numpy(values.copy())[:, order]
                    fps = float(data["fps"].reshape(-1)[0])
                n = int(self.catalog_len[i])
                if fps <= 0 or n != int(np.floor((len(q)-1)/fps/CONTROL_DT))+1:
                    raise ValueError(f"Invalid motion fps/frame metadata: {f}")
                t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * fps
                i0 = t.floor().long().clamp(max=len(q)-2)
                a = (t-i0.float()).clamp(0, 1)[:, None]
                q = (q[i0]*(1-a) + q[i0+1]*a).double()
                total += q.sum(0)
                square += q.square().sum(0)
                count += n
                if (i+1) % 5000 == 0:
                    print(f"[MotionPool] full-catalog action statistics {i+1}/{len(self.files)}", flush=True)
            mean = total / count
            moments = dict(count=count, mean=mean, var=(square/count-mean.square()).clamp(min=0))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(moments, tmp)
            os.replace(tmp, path)
        if moments["count"] != self.catalog_num_frames or not (
                torch.isfinite(moments["mean"]).all() and torch.isfinite(moments["var"]).all()):
            raise ValueError(f"Invalid full-catalog action statistics cache: {path}")
        return moments["var"].to(self.device), moments["mean"].to(self.device), moments["count"]

    def metrics(self):
        return dict(pool_active=len(self.active_ids), pool_resident=len(self.resident_ids),
                    pool_legacy=len(self.resident_ids)-len(self.active_ids), pool_frames=self.num_frames,
                    pool_gpu_gib=sum(getattr(self, k).numel()*getattr(self, k).element_size()
                                     for k in self.GPU_FIELDS) / 2**30,
                    pool_cpu_gib=sum(x.numel()*x.element_size() for v in self._cache.values() for x in v.values()) / 2**30,
                    pool_refresh_s=self.last_refresh_s, pool_loaded=self.last_loaded,
                    pool_catalog=len(self.files), pool_selected=int(self.ever_active.sum()),
                    pool_starts_seen=int(self.starts_seen.sum()), pool_starts_drawn=int(self.starts_drawn),
                    pool_refreshes=self.refresh_count)
