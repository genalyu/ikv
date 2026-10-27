"""Strict frozen-scorer plan artifact binding; no formula fallback."""
import hashlib
import json
from pathlib import Path
import torch
from functools import lru_cache
from .selection import VERSION


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


@lru_cache(maxsize=4096)
def cached_sha256(path, size, mtime_ns):
    return sha256(path)


def load_plan(config, task, episode, start, end, payloads, cameras, root):
    task = getattr(config, "memory_task_alias", {}).get(task, task)
    path = Path(config.memory_plan_root)/task/f"episode_{episode:06d}_{start}_{end}.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)
    expected = dict(version=VERSION, scorer_sha256=config.memory_scorer_sha256,
                    cameras=list(cameras), budgets=list(config.memory_budgets),
                    binding_sha256=config.memory_binding_sha256,chunk_size=config.memory_chunk_size)
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"Memory plan {key} mismatch: {path}")
    anchors = None
    for camera in cameras:
        files=list((Path(root)/"latents").glob(f"chunk-*/{camera}/episode_{episode:06d}_{start}_{end}.pth"))
        if len(files)!=1:
            raise ValueError("Missing or ambiguous VAE source for scorer plan")
        stat=files[0].stat()
        if cached_sha256(str(files[0]),stat.st_size,stat.st_mtime_ns)!=data["latent_sha256"][camera]:
            raise ValueError("VAE source changed after scorer plan generation")
        ids = torch.as_tensor(payloads[f"{camera}.frame_ids"], dtype=torch.long)[::4]
        if anchors is not None and not torch.equal(anchors, ids):
            raise ValueError("Camera clocks disagree")
        anchors = ids
    if not torch.equal(anchors, data["frame_ids"]):
        raise ValueError("Memory plan and actual VAE frame anchors differ")
    F = int(payloads[f"{cameras[0]}.latent_num_frames"])
    if data["selected"].shape != (2*((F+config.memory_chunk_size-1)//config.memory_chunk_size), sum(config.memory_budgets)):
        raise ValueError("Memory plan shape mismatch")
    return data["selected"]


def verify_binding(spec):
    binding = json.loads(Path(spec["binding"]).read_text())
    if binding.get("version") != VERSION:
        raise ValueError("Incompatible scorer binding")
    checkpoint = Path(binding["checkpoint"])
    if sha256(checkpoint) != binding["scorer_sha256"]:
        raise ValueError("Scorer changed after plan generation")
    marker = json.loads((Path(binding["plan_root"])/"COMPLETE.json").read_text())
    if marker.get("binding_sha256") != sha256(spec["binding"]):
        raise ValueError("All train/validation plans must finish before training")
    return binding


def write_training_manifest(trainer):
    cfg=trainer.config
    manifest=dict(version=VERSION,experiment=dict(cfg.memory_experiment),
        step=trainer.step,cameras=list(cfg.obs_cam_keys),normalization=dict(cfg.norm_stat),
        action_dimension=cfg.action_dim,action_per_frame=cfg.action_per_frame,
        chunk_size=cfg.memory_chunk_size,action_seed="measured-observation-v1",rgb_only=True,mode=cfg.task_mode,
        scorer_sha256=getattr(cfg,"memory_scorer_sha256",None),
        budgets=getattr(cfg,"memory_budgets",None))
    target=Path(cfg.save_root)/"checkpoints"/f"checkpoint_step_{trainer.step}"/"memory_manifest.json"
    target.write_text(json.dumps(manifest,indent=2))


def verify_encoder(root, revision, metadata_path):
    """Verify pinned Hub revision against bytes, including processor/tokenizer."""
    metadata=json.loads(Path(metadata_path).read_text())
    if metadata["sha"] != revision.split(":",1)[0]:
        raise ValueError("Scorer encoder revision differs from pinned model metadata")
    hashes={}
    for item in metadata["siblings"]:
        name=item["rfilename"]
        if not name.endswith((".safetensors", ".json", ".txt", ".model")) or name.startswith("assets/"):
            continue
        path=Path(root)/name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned encoder file missing: {path}")
        digest=sha256(path)
        if "lfs" in item:
            expected=item["lfs"]["sha256"]
            if digest!=expected:raise ValueError(f"Pinned encoder checksum mismatch: {name}")
        else:
            data=path.read_bytes()
            blob=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            if blob!=item["blobId"]:raise ValueError(f"Pinned processor checksum mismatch: {name}")
        hashes[name]=digest
    if not any(k.endswith(".safetensors") for k in hashes):
        raise ValueError("No verified encoder weights")
    return hashes
