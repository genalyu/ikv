"""Strict Sys2 scoring protocol. Feature references never enter Sys1 Q/K/V."""
from __future__ import annotations
import json
import math
from urllib.request import Request, urlopen
import torch


def binary_contact(value, count, device="cpu"):
    if value is None:
        raise ValueError("Sys2 tactile indices require explicit binary contact labels")
    try:
        value = torch.as_tensor(value, device=device)
    except (TypeError, ValueError) as exc:
        raise ValueError("contact labels must be binary, not text or unknown") from exc
    if value.ndim > 1 or value.numel() not in (1, count):
        raise ValueError("contact labels require one label per patch or an explicit sensor-wide scalar")
    if not ((value == 0) | (value == 1)).all():
        raise ValueError("contact labels must be exactly 0 or 1")
    return value.reshape(-1).expand(count).bool().clone()


def validate_candidates(candidates):
    ids = set()
    for candidate in candidates:
        required = {"id", "time", "modality", "frame", "patch", "grid"}
        if not required <= candidate.keys():
            raise ValueError("candidate lacks ID, time, modality or spatial mapping")
        uid = candidate["id"]
        if not isinstance(uid, str) or not uid or uid in ids:
            raise ValueError("candidate IDs must be unique nonempty strings")
        ids.add(uid)
        if candidate["modality"] not in ("visual", "tactile"):
            raise ValueError("Sys2 ranks visual and tactile candidates only")
        if not isinstance(candidate["time"], (int, float)) or not math.isfinite(candidate["time"]):
            raise ValueError("candidate time must be finite")
        if type(candidate["frame"]) is not int or candidate["frame"] < 0:
            raise ValueError("candidate frame must refer to an encoded frame")
        grid, patch = candidate["grid"], candidate["patch"]
        if (len(grid) != 2 or any(type(n) is not int or n < 1 for n in grid)
                or type(patch) is not int or not 0 <= patch < grid[0]*grid[1]):
            raise ValueError("invalid Sys1 spatial patch mapping")
        if candidate["modality"] == "tactile":
            if type(candidate.get("contact")) is not bool:
                raise ValueError("tactile contact must be an explicit JSON boolean")
    return candidates


def unwrap_json_fence(text):
    """Remove only one complete Markdown JSON wrapper; never extract partial JSON."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    return text


def validate_scores(output, candidates):
    validate_candidates(candidates)
    if isinstance(output, str):
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        output = json.loads(output, object_pairs_hook=unique_pairs)
    if not isinstance(output, dict) or set(output) != {"visual_scores", "tactile_scores"}:
        raise ValueError("Sys2 must return only visual_scores and tactile_scores")
    result = {}
    for modality in ("visual", "tactile"):
        expected = {c["id"] for c in candidates if c["modality"] == modality}
        entries = output[modality + "_scores"]
        if not isinstance(entries, list):
            raise ValueError("scores must be arrays")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "score"}:
                raise ValueError("score entry must contain only id and score")
            uid, score = entry["id"], entry["score"]
            if not isinstance(uid, str) or uid not in expected or uid in seen:
                raise ValueError("unknown, duplicate or wrong-modality score ID")
            if type(score) is not int or not 0 <= score <= 100:
                raise ValueError("score must be an integer in [0,100]")
            seen.add(uid)
            result[uid] = score
        if seen != expected:
            raise ValueError("Sys2 omitted candidate IDs; no partial scores accepted")
    return result


class Sys2Client:
    def __init__(self, endpoint, task, timeout=300):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Sys2 endpoint must be HTTP(S)")
        self.endpoint, self.task, self.timeout = endpoint.rstrip("/"), task, timeout

    def call(self, route, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        request = Request(self.endpoint + route, data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def encode(self, rgb):
        import base64
        import io
        from PIL import Image
        rgb = torch.as_tensor(rgb).detach().cpu()
        if rgb.ndim != 3 or rgb.shape[0] != 3 or not torch.isfinite(rgb).all():
            raise ValueError("Qwen encoding requires finite CHW RGB")
        if (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("Qwen RGB must be in [0,1]")
        buffer = io.BytesIO()
        pixels = (rgb.permute(1,2,0)*255).round().byte().numpy()
        Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
        result = self.call("/encode", {"image": base64.b64encode(buffer.getvalue()).decode()})
        if type(result.get("frame")) is not int or result["frame"] < 0:
            raise ValueError("invalid encoded frame reference")
        return result["frame"]

    def index(self, videos, anchors, target, *, tactile=False, contact=None, device="cpu"):
        """Native full images are encoded once; map Sys1 rows without averaging Qwen tokens."""
        h, w = target
        if len(anchors) == 0 or h < 1 or w < 1:
            raise ValueError("empty Sys2 frame or patch grid")
        streams = []
        for video in videos:
            streams.append(torch.tensor([self.encode(video[:, t]) for t in anchors], dtype=torch.long))
        ids = torch.stack(streams)[:, :, None, None].expand(-1, -1, h, w)
        patches = torch.arange(h*w).reshape(1,1,h,w).expand_as(ids)
        # Visual WAN layout: frame, row, camera, column. Tactile: sensor, frame, row, column.
        layout = (lambda x: x.flatten()) if tactile else (lambda x: x.permute(1,2,0,3).flatten())
        frame, patch = layout(ids).to(device), layout(patches).to(device)
        n = len(frame)
        result = dict(sys2_frame=frame, sys2_patch=patch,
                      sys2_height=torch.full((n,), h, dtype=torch.long, device=device),
                      sys2_width=torch.full((n,), w, dtype=torch.long, device=device),
                      observation_flag=torch.ones(n, dtype=torch.bool, device=device),
                      duration=torch.ones(n, device=device), task_relevance=torch.zeros(n, device=device),
                      dino=torch.empty(n, 0, device=device), neoforce=torch.empty(n,0,device=device))
        if tactile:
            result["contact_present"] = binary_contact(contact, n, device)
        return result

    def score(self, candidates, current_time):
        validate_candidates(candidates)
        output = self.call("/score", dict(task=self.task, current_time=current_time,
                                          candidates=candidates))
        return validate_scores(output, candidates)

    def __call__(self, data, rows, next_uid):
        """Archive callback: send all archived real rows plus protected incoming rows."""
        candidates = []
        for source, incoming in ((data, False), (rows, True)):
            n = len(source["kind"])
            for i in range(n):
                kind = int(source["kind"][i])
                if kind == 1:
                    continue
                if not bool(source["observation_flag"][i]):
                    raise ValueError("predicted KV cannot enter Sys2 history")
                uid = next_uid+i if incoming else int(source["token_uid"][i])
                item = dict(id=str(uid), time=float(source["world_time_id"][i]),
                            modality="visual" if kind == 0 else "tactile",
                            frame=int(source["sys2_frame"][i]),
                            patch=int(source["sys2_patch"][i]),
                            grid=[int(source["sys2_height"][i]), int(source["sys2_width"][i])])
                if kind == 2:
                    item["contact"] = bool(source["contact_present"][i])
                candidates.append(item)
        if not candidates:
            return torch.zeros(len(data["kind"]))
        scores = self.score(candidates, max(c["time"] for c in candidates))
        self.last_scores = scores
        return torch.tensor([scores[str(int(uid))] if int(kind) != 1 else 0.
                             for uid, kind in zip(data["token_uid"], data["kind"])])
