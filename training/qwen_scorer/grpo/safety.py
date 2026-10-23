"""Explicit busy-GPU guard; never launch over an existing training job."""
import os
import subprocess


def require_idle_devices(config):
    logical = {int(config[k].split(":")[1]) for k in ("qwen_device","n0_device")}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",") if visible is not None else None
    ids = [physical[i] if physical is not None else str(i) for i in logical]
    for device in ids:
        result = subprocess.run(
            ["nvidia-smi","-i",device,"--query-compute-apps=pid","--format=csv,noheader,nounits"],
            check=True,text=True,capture_output=True)
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
        if pids:
            raise RuntimeError(f"GPU {device} already has compute processes {pids}; choose idle GPUs")
