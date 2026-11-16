"""Replace selected link origins with one rigidly attached reward point per link."""
from __future__ import annotations

import torch


class TrackingPoints:
    def __init__(self, kin, offsets):
        names = [kin.body_names[i] for i in kin.tracked_idx]
        missing = set(offsets) - set(names)
        if missing:
            raise ValueError(f"Reward point offsets require tracked links: {sorted(missing)}")
        self.offsets = torch.tensor([offsets.get(name, (0., 0., 0.)) for name in names],
                                    dtype=torch.float32, device=kin.q_lo.device)

    def apply(self, positions, rotations, linear_velocities, angular_velocities):
        # Same transform for simulator FK and motion references, with no in-place cache edits.
        lever = (rotations @ self.offsets.unsqueeze(-1)).squeeze(-1)
        return (positions + lever, rotations,
                linear_velocities + torch.cross(angular_velocities, lever, dim=-1), angular_velocities)
