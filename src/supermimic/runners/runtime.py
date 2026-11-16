"""Graceful signals; a stop on any distributed rank stops the same iteration."""
import signal


class StopController:
    def __init__(self):
        self.signals = 0
        self.world, self.device = 1, "cpu"

    def _request_stop(self, sig, _frame):
        self.signals += 1
        if self.signals > 1:
            raise KeyboardInterrupt
        print(f"\n[{signal.Signals(sig).name}] Stopping after the current iteration and saving a checkpoint "
              "(send it again to quit without saving).", flush=True)

    def __enter__(self):
        self.previous = {sig: signal.signal(sig, self._request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        return self

    def __exit__(self, *exc):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def requested(self):
        if self.world == 1:
            return bool(self.signals)
        import torch
        import torch.distributed as dist
        flag = torch.tensor(self.signals, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag)
