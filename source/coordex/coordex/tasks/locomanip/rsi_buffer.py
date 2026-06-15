from __future__ import annotations

import torch


class NoDemoRSIBuffer:
    """Simple per-stage ring buffer for reset state snapshots.
    """

    def __init__(self, num_stages: int, capacity_per_stage: int, device: str = "cpu"):
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}.")
        if capacity_per_stage <= 0:
            raise ValueError(
                f"capacity_per_stage must be positive, got {capacity_per_stage}.")
        self.num_stages = int(num_stages)
        self.capacity_per_stage = int(capacity_per_stage)
        self.storage_device = torch.device(device)
        self.dtype = torch.float32

        self._storage: list[torch.Tensor | None] = [None] * self.num_stages
        self._write_ptr = torch.zeros(self.num_stages, dtype=torch.long)
        self._size = torch.zeros(self.num_stages, dtype=torch.long)
        self._add_count = torch.zeros(self.num_stages, dtype=torch.long)

    def add(self, stage: int, env_ids: torch.Tensor, snapshot: torch.Tensor) -> None:
        """Insert snapshot(s) for a given stage."""
        stage_idx = int(stage)
        if stage_idx < 0 or stage_idx >= self.num_stages:
            return
        if snapshot is None:
            return

        _ = env_ids

        snap = snapshot.to(device=self.storage_device, dtype=self.dtype)
        if snap.ndim == 1:
            snap = snap.unsqueeze(0)
        if snap.numel() == 0 or snap.shape[0] == 0:
            return

        cap = self.capacity_per_stage
        if snap.shape[0] > cap:
            snap = snap[-cap:]

        if self._storage[stage_idx] is None or self._storage[stage_idx].shape[1] != snap.shape[1]:
            self._storage[stage_idx] = torch.zeros(
                (cap, snap.shape[1]
                 ), device=self.storage_device, dtype=self.dtype
            )
            self._write_ptr[stage_idx] = 0
            self._size[stage_idx] = 0

        size = int(self._size[stage_idx].item())
        ptr = int(self._write_ptr[stage_idx].item())
        count = snap.shape[0]

        # Build index tensor on the storage device to avoid cross-device indexing.
        indices = (torch.arange(count, device=self.storage_device) + ptr) % cap
        self._storage[stage_idx][indices] = snap
        self._write_ptr[stage_idx] = (ptr + count) % cap
        self._size[stage_idx] = min(cap, size + count)
        self._add_count[stage_idx] += count

    def sample(self, stage: int, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Sample ``n`` snapshots (with replacement) from the buffer for the given stage."""
        stage_idx = int(stage)
        if stage_idx < 0 or stage_idx >= self.num_stages:
            raise IndexError(f"Stage index out of range: {stage_idx}.")
        size = int(self._size[stage_idx].item())
        if size == 0:
            raise RuntimeError(
                f"No snapshots available for stage {stage_idx}.")
        if self._storage[stage_idx] is None:
            raise RuntimeError(
                f"Storage for stage {stage_idx} not initialized.")

        n = int(n)
        if n <= 0:
            return torch.zeros(
                (0, self._storage[stage_idx].shape[1]), dtype=self.dtype, device=self.storage_device
            )

        indices = torch.randint(0, size, (n,), device=self.storage_device)
        samples = self._storage[stage_idx][indices]

        target_device = self.storage_device if device is None else torch.device(
            device)

        return samples.to(device=target_device)

    def size(self, stage: int) -> int:
        """Return the number of stored snapshots for a given stage."""
        stage_idx = int(stage)
        if stage_idx < 0 or stage_idx >= self.num_stages:
            return 0
        return int(self._size[stage_idx].item())

    def sizes(self) -> torch.Tensor:
        """Return a CPU tensor of sizes for all stages (no CUDA sync)."""
        return self._size.clone()

    def add_count(self, stage: int) -> int:
        """Return the cumulative number of inserted snapshots for a given stage."""
        stage_idx = int(stage)
        if stage_idx < 0 or stage_idx >= self.num_stages:
            return 0
        return int(self._add_count[stage_idx].item())

    def add_counts(self) -> torch.Tensor:
        """Return cumulative inserted snapshot counts for all stages."""
        return self._add_count.clone()
