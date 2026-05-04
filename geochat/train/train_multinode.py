"""Entry point for multi-node DeepSpeed training.

Applies two monkey-patches needed to run transformers 4.37.2 on top of
torch >= 2.6 and DeepSpeed ZeRO-2/3:

  1. accelerate.Accelerator.no_sync -> nullcontext when backend is DeepSpeed.
     Otherwise the trainer's `accelerator.accumulate(model)` raises:
       AssertionError: no_sync context manager is incompatible with
       gradient partitioning logic of ZeRO stage 2

  2. torch.optim.lr_scheduler.LRScheduler._update_lr -> drop strict=True in
     zip(). Torch >= 2.6 enforces param_groups and values to have the same
     length, but transformers 4.37's scheduler was created with N groups
     and DeepSpeed may consolidate to M < N after optimizer wrapping.
     Pre-2.6 torch truncated silently; we restore that behaviour.

Both issues are fixed upstream in newer transformers (>= 4.44) but we keep
4.37.2 to preserve GeoChat compatibility. Both patches are safe because
DeepSpeed handles accumulation/sync internally, and LoRA param groups share
the same base LR.
"""
import contextlib

from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.optim.lr_scheduler import LRScheduler, _enable_get_lr_call

# --- Patch 1: DeepSpeed-safe no_sync -----------------------------------------
_orig_no_sync = Accelerator.no_sync


@contextlib.contextmanager
def _no_sync_patched(self, model):
    if self.distributed_type == DistributedType.DEEPSPEED:
        yield
    else:
        with _orig_no_sync(self, model):
            yield


Accelerator.no_sync = _no_sync_patched


# --- Patch 2: non-strict zip in LR scheduler (keep torch 2.11 semantics) -----
# Replicates torch 2.11 _update_lr EXACTLY, only removes strict=True in the
# final zip. The critical bits are: (1) increment last_epoch inside the
# `_enable_get_lr_call` context when epoch is None, (2) only call
# `_get_closed_form_lr` when epoch is provided.
def _update_lr_nonstrict(self, epoch=None):
    with _enable_get_lr_call(self):
        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr()
        else:
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_lr"):
                values = self._get_closed_form_lr()
            else:
                values = self.get_lr()

    # DeepSpeed may consolidate param_groups after scheduler creation, so
    # values can be longer than param_groups. Non-strict zip truncates safely.
    for param_group, lr in zip(self.optimizer.param_groups, values):
        param_group["lr"] = lr

    self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


LRScheduler._update_lr = _update_lr_nonstrict

from geochat.train.train import train

if __name__ == "__main__":
    train()
