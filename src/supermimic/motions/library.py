"""Reference motion library: retargets resampled to the control rate, with FK of tracked bodies."""

from __future__ import annotations

import glob
import json
import hashlib
import os
import re

import numpy as np
import torch

from supermimic.robots.kinematics import G1Kinematics
from supermimic.robots import JOINT_NAMES, ROBOT_NAME
from supermimic.utils.rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.01  # 100 Hz policy, replay transitions and world-model steps
BIN_FRAMES = round(1.0 / CONTROL_DT)  # retain 1 s return-based reset bins
# per-frame goal parts in MotionLib.goal order, with their sizes
GOAL_PARTS = {"root_pos": 3, "root_rot": 6, "root_lin_vel": 3, "root_ang_vel": 3, "q": 29, "qd": 29}


def parse_goal_drop(s: str) -> tuple[str, ...]:
    """``"root_pos,root_lin_vel"`` -> the goal parts to leave out, in GOAL_PARTS order."""
    names = {n for n in s.split(",") if n}
    if names - GOAL_PARTS.keys():
        raise ValueError(f"unknown goal part(s) {sorted(names - GOAL_PARTS.keys())}; choose from {list(GOAL_PARTS)}")
    if names == GOAL_PARTS.keys():
        raise ValueError("the goal needs at least one part")
    return tuple(n for n in GOAL_PARTS if n in names)


def goal_frame_dim(drop: tuple[str, ...] = ()) -> int:
    return sum(d for n, d in GOAL_PARTS.items() if n not in drop)


def motion_group(name: str) -> str:
    """``walk__qingtong1000__BG_Brisk_walking_00012`` -> ``walk``; names without ``__`` drop the trailing index."""
    return name.split("__")[0] if "__" in name else re.sub(r"_\d+$", "", name)


class MotionLib:
    # Standalone/legacy callers keep consecutive frames; training/playback set this from run metadata.
    goal_stride = 1
    goal_start = None  # Standalone/legacy loaders start one stride ahead unless configured.

    def __init__(self, files: list[str], kin: G1Kinematics, device: str, src_fps: float = 30.0,
                 body_storage: str = "cpu"):
        self.device = device
        if body_storage not in ("cpu", "gpu"):
            raise ValueError(f"unknown body_storage {body_storage!r}")
        self.kin, self.body_storage = kin, body_storage
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js, vs = [], [], [], []
        lens = []
        for f in files:
            velocities = None
            if f.endswith(".npz"):
                p, quat, j, velocities, clip_fps = _load_npz(f)
            else:
                if ROBOT_NAME != "g1":
                    raise ValueError("CSV joint layout is only defined for G1")
                raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
                p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
                quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
                clip_fps = src_fps
            if len(p) < 2:
                raise ValueError(f"Motion must contain at least two frames: {f}")
            # Resample to the control rate, preserving the source clip's duration.
            T_src = p.shape[0]
            dur = (T_src - 1) / clip_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * clip_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            vs.append(tuple(x[i0] * (1-a[:,None]) + x[i0+1] * a[:,None] for x in velocities) if velocities is not None else None)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j, supplied_vel in zip(ps, qs, js, vs):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            if supplied_vel is not None:
                v, w, qd = supplied_vel
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        self._cache_body_reference()
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            nb = int(np.ceil(lens[c] / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    def _cache_body_reference(self):
        """Use the original 65536-frame GPU FK batches, storing identical float32 outputs on CPU.

        Preserving the precomputation batch size also preserves TF32's shape-dependent rounding.
        Root/joint trajectories and all sampling/goal tables remain on the GPU.
        """
        with torch.no_grad():
            if self.body_storage == "cpu":
                self._num_tracked = len(self.kin.tracked_idx)
                self._body_widths = tuple(self._num_tracked * n for n in (3, 9, 3, 3))
                # The whole library can exceed 70 GiB; only transfer batches need pinned memory.
                self._body_cache = torch.empty((self.num_frames, sum(self._body_widths)), dtype=torch.float32,
                                              device="cpu")
                for s in range(0, self.num_frames, 65536):
                    e = min(s + 65536, self.num_frames)
                    out = self.kin.forward(*self.ref_state(slice(s, e)))
                    packed = torch.cat([x.flatten(1) for x in out], -1)
                    self._body_cache[s:e].copy_(packed)
                return
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = self.kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            # Keep identical concatenated tensors; release each temporary chunk list immediately.
            self.body_p = torch.cat(bp)
            bp.clear()
            self.body_R = torch.cat(bR)
            bR.clear()
            self.body_v = torch.cat(bv)
            bv.clear()
            self.body_w = torch.cat(bw)
            bw.clear()

    @torch.compiler.disable
    def body_reference(self, f: torch.Tensor, positions_only: bool = False):
        """Gather a pinned CPU batch and transfer it before compiled reward/critic computation."""
        if self.body_storage == "gpu":
            if positions_only:
                return self.body_p[f], None, None, None
            return self.body_p[f], self.body_R[f], self.body_v[f], self.body_w[f]
        index = f.detach().to("cpu").reshape(-1)
        columns = self._body_cache[:, :self._body_widths[0]] if positions_only else self._body_cache
        host = torch.empty((index.numel(), columns.shape[1]), dtype=columns.dtype,
                           pin_memory=torch.device(self.device).type == "cuda")
        torch.index_select(columns, 0, index, out=host)
        batch = host.to(self.device, non_blocking=True)
        if positions_only:
            return batch.reshape(*f.shape, self._num_tracked, 3), None, None, None
        p, R, v, w = batch.split(self._body_widths, -1)
        return (p.reshape(*f.shape, self._num_tracked, 3), R.reshape(*f.shape, self._num_tracked, 3, 3),
                v.reshape(*f.shape, self._num_tracked, 3), w.reshape(*f.shape, self._num_tracked, 3))

    @staticmethod
    def from_dir(path: str, kin: G1Kinematics, device: str, rank: int = 0, world: int = 1, exclude: str | None = None) -> MotionLib:
        """Load all motions, or with ``world > 1`` every ``world``-th file starting at ``rank``
        (all files on every rank when there are fewer files than ranks). ``exclude``: regex; files whose
        name it matches (``re.search``) are skipped."""
        files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.csv")) + glob.glob(os.path.join(path, "*.npz")))
        if exclude:
            files = [f for f in files if not re.search(exclude, os.path.basename(f))]
        if not files:
            raise FileNotFoundError(f"No motion files found at {path}" + (f" outside --motion_exclude {exclude!r}" if exclude else ""))
        if len(files) >= world:
            files = files[rank::world]
        return MotionLib(files, kin, device)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """K frames spaced by goal_stride, clamped within this clip, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        start = self.goal_stride if self.goal_start is None else self.goal_start
        fut = f.unsqueeze(-1) + start + self.goal_stride * torch.arange(K, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int, drop: tuple[str, ...] = ()) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame,
        without the GOAL_PARTS named in ``drop``."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        feats = []
        if "root_pos" not in drop:
            feats.append((Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1))
        if "root_rot" not in drop:
            feats.append(mat_to_6d(Ht @ self.root_R[fut]))
        if "root_lin_vel" not in drop:
            feats.append(self.root_v_local[fut])
        if "root_ang_vel" not in drop:
            feats.append(self.root_w_local[fut])
        if "q" not in drop:
            feats.append(self.q[fut])
        if "qd" not in drop:
            feats.append(self.qd[fut] * 0.1)
        return torch.cat(feats, -1).reshape(f.shape[0], -1)  # (B, K * goal_frame_dim)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def _load_npz(path):
    """Read a validated, named NPZ in wxyz convention and reorder to simulator joints."""
    metadata = path + '.metadata.json'
    if not os.path.isfile(metadata):
        raise ValueError(f"Missing validated motion metadata: {metadata}; run scripts/assets/prepare_tiangong3.py first")
    with open(metadata) as stream:
        meta = json.load(stream)
    if meta['robot'] != ROBOT_NAME or meta['quaternion_format'] != 'wxyz':
        raise ValueError('Motion metadata does not match selected robot/quaternion format')
    with open(path, 'rb') as stream:
        if hashlib.sha256(stream.read()).hexdigest() != meta['sha256']:
            raise ValueError('Motion file changed; regenerate its validation metadata')
    with np.load(path, allow_pickle=False) as data:
        order = [meta['joint_names'].index(n) for n in JOINT_NAMES]
        root = meta['root_body_index']
        def tensor(key):
            x = np.asarray(data[key], dtype=np.float32)
            if not np.isfinite(x).all(): raise ValueError(f'Nonfinite {key}')
            return torch.from_numpy(x.copy())
        p, quat = tensor('body_pos_w')[:,root], tensor('body_quat_w')[:,root]
        q, qd = tensor('joint_pos')[:,order], tensor('joint_vel')[:,order]
        v, w = tensor('body_lin_vel_w')[:,root], tensor('body_ang_vel_w')[:,root]
        fps = float(data['fps'].reshape(-1)[0])
    if fps <= 0 or len(p) != len(q): raise ValueError('Invalid NPZ fps/frame count')
    if not torch.allclose(quat.norm(dim=-1), torch.ones(len(quat)), atol=1e-3):
        raise ValueError('NPZ root quaternions are not normalized')
    return p, quat, q, (v,w,qd), fps
