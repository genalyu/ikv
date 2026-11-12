"""Independent frozen Qwen3.5-0.8B scorer and area/time provenance mapping.

This encoder never shares weights/features with HWM's native Planner. Image
features are immutable and reusable; task/subtask-conditioned scores are always
recomputed. All causal registered image features participate in scoring.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from .types import Candidate


@dataclass(frozen=True)
class _Frame:
    time: float
    digest: str
    shape: tuple
    features: torch.Tensor


def _import_source(source_root):
    root = Path(source_root).expanduser().resolve()
    if not (root / "training/parallel_scorer/runtime.py").is_file():
        raise FileNotFoundError("scorer source_root must contain training/parallel_scorer")
    # Reject silently importing a different deployment already on sys.path.
    for name in ("training", "training.parallel_scorer"):
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if filename and root not in Path(filename).resolve().parents:
            raise RuntimeError("A different scorer source is already imported")
    sys.path.insert(0, str(root))
    runtime_module = importlib.import_module("training.parallel_scorer.runtime")
    encoder_module = importlib.import_module("training.parallel_scorer.encoders")
    for module in (runtime_module, encoder_module):
        filename = getattr(module, "__file__", None)
        if not filename or root not in Path(filename).resolve().parents:
            raise RuntimeError("Loaded scorer module is outside the configured source root")
    return runtime_module.RuntimeScorer, encoder_module.QwenEncoders


class ScorerBridge:
    def __init__(self, source_root=None, checkpoint=None, encoder_path=None,
                 device="cuda:0", *, encoder=None, runtime=None):
        if (encoder is None) != (runtime is None):
            raise ValueError("Explicit test injection requires both encoder and runtime")
        if encoder is None:
            if not all((source_root, checkpoint, encoder_path)):
                raise ValueError("IKV requires scorer source, checkpoint, and independent encoder")
            if not Path(checkpoint).expanduser().is_file():
                raise FileNotFoundError(checkpoint)
            if not Path(encoder_path).expanduser().is_dir():
                raise FileNotFoundError(encoder_path)
            RuntimeScorer, QwenEncoders = _import_source(source_root)
            runtime = RuntimeScorer(str(Path(checkpoint).expanduser()), device=device)
            revision = runtime.info.get("encoder_revision")
            if not revision:
                raise ValueError("Scorer checkpoint lacks pinned encoder_revision")
            encoder = QwenEncoders(str(Path(encoder_path).expanduser()), revision,
                                   device=device, train_vision=False, train_text=False,
                                   image_batch=3)
        self.runtime = runtime
        self.encoder = encoder
        self.revision = runtime.info.get("encoder_revision")
        if not self.revision or getattr(encoder, "revision", None) != self.revision:
            raise ValueError("Scorer/encoder revision mismatch")
        for obj in (getattr(runtime, "model", None), encoder):
            if hasattr(obj, "eval"):
                obj.eval()
            if hasattr(obj, "requires_grad_"):
                obj.requires_grad_(False)
        self._frames = {}
        self._frame_times = {}
        self.encoded_images = 0
        self.score_calls = 0

    def fork(self):
        """Share immutable model weights, never per-episode image features."""
        return type(self)(encoder=self.encoder, runtime=self.runtime)

    def reset(self):
        self._frames = {}
        self._frame_times = {}
        self.encoded_images = 0
        self.score_calls = 0

    @property
    def registered_frames(self):
        return tuple(sorted(self._frame_times, key=lambda f: (self._frame_times[f], f)))

    @torch.inference_mode()
    def register_frame(self, frame_id, timestamp, images):
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("frame_id must be a nonempty string")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Frame timestamp must be finite and nonnegative")
        if frame_id in self._frame_times and self._frame_times[frame_id] != timestamp:
            raise ValueError("Frame identity changed timestamp")
        if not isinstance(images, dict) or not images:
            raise ValueError("images must map sensor IDs to RGB arrays")
        pending = []
        for sensor, image in sorted(images.items()):
            if type(sensor) is not int or sensor < 0:
                raise ValueError("Invalid sensor ID")
            rgb = np.asarray(image)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 1:
                raise ValueError("Scorer images must be uint8 HWC RGB")
            rgb = np.ascontiguousarray(rgb)
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            key = (frame_id, sensor)
            old = self._frames.get(key)
            if old is not None:
                if (old.time, old.digest, old.shape) != (timestamp, digest, rgb.shape):
                    raise ValueError("Frame identity changed RGB content/shape")
            else:
                image = Image.fromarray(rgb).resize((256, 256), Image.Resampling.BICUBIC)
                pending.append((key, digest, rgb.shape, image))
        if pending:
            features = self.encoder.images([p[3] for p in pending])
            if not isinstance(features, torch.Tensor) or features.ndim != 2:
                raise ValueError("Scorer encoder must emit a feature matrix")
            if features.shape[0] != len(pending) * 64 or not torch.isfinite(features).all().item():
                raise ValueError("Scorer encoder must emit 64 finite patches per image")
            features = features.detach().cpu().reshape(len(pending), 64, -1).clone()
            if self._frames and next(iter(self._frames.values())).features.shape[-1] != features.shape[-1]:
                raise ValueError("Scorer feature width changed")
            additions = {key: _Frame(timestamp, digest, shape, feature)
                         for (key, digest, shape, _), feature in zip(pending, features)}
            self._frames.update(additions)
            self.encoded_images += len(pending)
        self._frame_times[frame_id] = timestamp

    @staticmethod
    def _condition(task, subtask, memory):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Scoring requires the task instruction")
        # Empty feedback reproduces the trained task-only text path exactly.
        text = task
        if subtask:
            text += "\nCurrent subtask: " + str(subtask)
        if memory:
            text += "\nSemantic memory: " + str(memory)
        return text

    @staticmethod
    def _region_score(grid, box):
        y0, x0, y1, x1 = box
        edges = torch.arange(9, dtype=torch.float64) / 8
        wy = (torch.minimum(edges[1:], torch.tensor(y1)) -
              torch.maximum(edges[:-1], torch.tensor(y0))).clamp_min(0)
        wx = (torch.minimum(edges[1:], torch.tensor(x1)) -
              torch.maximum(edges[:-1], torch.tensor(x0))).clamp_min(0)
        weights = wy[:, None] * wx[None, :]
        if weights.sum() <= 0:
            raise ValueError("Region does not overlap the scorer grid")
        return float((grid.double() * weights).sum() / weights.sum())

    @torch.inference_mode()
    def score(self, candidates, task, query_time, subtask="", memory=""):
        candidates = tuple(candidates)
        if not candidates:
            return {}
        if any(not isinstance(c, Candidate) for c in candidates):
            raise TypeError("Expected native KV Candidate records")
        if len({c.uid for c in candidates}) != len(candidates):
            raise ValueError("Duplicate scored candidate IDs")
        query_time = float(query_time)
        if not math.isfinite(query_time) or query_time < 0:
            raise ValueError("Invalid query time")
        # Keep future registered observations out of both scores and context.
        keys = sorted((key for key, f in self._frames.items() if f.time <= query_time),
                      key=lambda key: (self._frames[key].time, key[0], key[1]))
        if not keys:
            raise ValueError("No causal registered observations")
        lookup = {key: i for i, key in enumerate(keys)}
        for c in candidates:
            if c.time > query_time:
                raise ValueError("A future candidate cannot be scored")
            for region in c.regions:
                key = (region.frame_id, region.sensor)
                if key not in lookup:
                    raise ValueError("Missing or future observation in KV provenance")
                if self._frames[key].time > c.time + 1e-6:
                    raise ValueError("Candidate predates a supporting observation")
        width = 64
        xy = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8),
                                       indexing="ij"), -1).reshape(-1, 2).float() / 7
        text = self.encoder.task(self._condition(task, subtask, memory)).detach().cpu()
        packet = dict(
            features=torch.cat([self._frames[key].features for key in keys]),
            text=text,
            times=torch.tensor([self._frames[key].time for key in keys],
                               dtype=torch.float64).repeat_interleave(width),
            xy=xy.repeat(len(keys), 1),
            modalities=torch.zeros(len(keys) * width, dtype=torch.long),
            sensors=torch.tensor([key[1] for key in keys]).repeat_interleave(width),
            query_time=query_time,
            encoder_revision=self.revision,
            ids=[f"{key[0]}:c{key[1]}:p{p}" for key in keys for p in range(width)],
        )
        result = self.runtime.score(packet, {0: len(keys) * width, 1: 0})
        scores = torch.as_tensor(result["scores"]).detach().cpu().float()
        if scores.shape != (len(keys) * width,) or not torch.isfinite(scores).all().item():
            raise ValueError("Invalid learned scorer output")
        grids = scores.reshape(len(keys), 8, 8)
        mapped = {}
        for c in candidates:
            value = sum(r.weight * self._region_score(grids[lookup[(r.frame_id, r.sensor)]], r.box)
                        for r in c.regions)
            mapped[c.uid] = value / sum(r.weight for r in c.regions)
        self.score_calls += 1
        return mapped
