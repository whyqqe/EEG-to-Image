#!/usr/bin/env python
"""Fail fast instead of silently training on the CPU.

The cluster's GPU nodes do not share a driver version.  A job that landed on
dgx-09 logged

    CUDA initialization: The NVIDIA driver on your system is too old
    (found version 12080)

while this environment's torch is built for CUDA 13.0, so `torch.cuda.is_available()`
returned False.  Every stage here used to pick its device with

    torch.device(args.device if torch.cuda.is_available() else "cpu")

which turns that hardware mismatch into a silent 10-20x slowdown.  A one-epoch S1
run that takes 75 s on dgx-13 took over 10 minutes on dgx-09 without finishing, and
the diffusion generation stage would take days.  Nothing in the logs said "CPU" --
the only evidence was a warning line from `torch.cuda` printed before training began.

So: if a GPU was requested and cannot be used, raise.  `allow_cpu=True` is the
explicit opt-out for a deliberate CPU run.
"""

from __future__ import annotations

import os


class DeviceError(RuntimeError):
    """Raised when a requested accelerator is not usable."""


def pick_device(requested: str = "cuda:0", allow_cpu: bool = False):
    """Return a torch.device, or raise if `requested` is a GPU we cannot use."""
    import torch

    want_cuda = requested.startswith("cuda")
    have_cuda = torch.cuda.is_available()
    if want_cuda and not have_cuda and not allow_cpu:
        reason = "torch.cuda.is_available() is False"
        try:  # surface the underlying cause when torch exposes one
            import subprocess
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20)
            if out.stdout.strip():
                reason += (f"; nvidia-smi reports driver {out.stdout.strip()} but this "
                           f"torch build needs a newer one "
                           f"(torch {torch.__version__})")
        except Exception:  # noqa: BLE001
            pass
        hosts = os.environ.get("SLURMD_NODENAME", "?")
        raise DeviceError(
            f"requested {requested} but no usable GPU on {hosts}: {reason}.\n"
            f"The GPU nodes have mixed driver versions, so this is a scheduling "
            f"lottery, not a code error -- resubmit (a different node will be "
            f"allocated), or pass --allow-cpu to accept a very slow CPU run on "
            f"purpose.")
    return torch.device(requested if want_cuda else "cpu")
