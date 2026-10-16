"""Task overlays; optimizer/model defaults stay in the official posttrain config."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

OFFICIAL_REVISION = "cdd87b6a141667123ad2c25f452478afdb71e287"
MODES = {
    "baseline": (False, False),
    "motion": (True, False),
    "ikv": (False, True),
    "motion_ikv": (True, True),
}
RECIPE = dict(
    batch_size=1,
    num_steps=2000,
    save_interval=500,
    gc_interval=50,
    learning_rate=1e-4,
    lr_schedule="cosine",
    lr_min_ratio=0.1,
    warmup_steps=20,
    weight_decay=0.1,
    max_latent_frames=0,
    max_tactile_frames=0,
    load_worker=4,
    num_init_worker=1,
)


def fingerprint(value):
    # A byte-identical frozen model may move without invalidating derived features
    # or optimizer checkpoints.  Keep the original task identity only when the
    # replacement DINO weight has been verified against the recorded digest.
    if isinstance(value, dict) and isinstance(value.get("runtime"), dict):
        migration = value["runtime"].get("dino_model_fingerprint_compat")
        if migration is not None:
            if set(migration) != {"original_path", "model_sha256"}:
                raise ValueError("Invalid DINO model fingerprint migration")
            replacement = Path(value["runtime"]["dino_model"]) / "model.safetensors"
            digest = hashlib.sha256()
            with replacement.open("rb") as weights:
                for block in iter(lambda: weights.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != migration["model_sha256"]:
                raise ValueError("Relocated DINO model differs from feature source")
            value = deepcopy(value)
            value["runtime"]["dino_model"] = migration["original_path"]
            del value["runtime"]["dino_model_fingerprint_compat"]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def load_task(path):
    path = Path(path).resolve()

    def read(p):
        value = json.loads(p.read_text())
        for field in ("robot", "runtime"):
            if isinstance(value.get(field), str):
                value[field] = json.loads((p.parent / value[field]).read_text())
        return value

    task = read(path)
    for name in ("name", "format", "source", "prompt", "robot", "runtime"):
        if name not in task:
            raise ValueError(f"task missing {name}")
    if not task["prompt"].strip():
        raise ValueError("An explicit generic task prompt is required")
    if task["format"] not in ("collector_v06", "lerobot_v21", "neosim_hdf5"):
        raise ValueError("Unsupported format; define an explicit adapter")
    if isinstance(task["source"], list):
        if (
            task["format"] != "collector_v06"
            or not task["source"]
            or not all(isinstance(item, str) for item in task["source"])
            or len(set(task["source"])) != len(task["source"])
        ):
            raise ValueError("Source lists require distinct collector_v06 paths")
    semantic = task["runtime"].get("semantic_encoder")
    if semantic:
        for field in ("repo", "model", "backbone_weights", "head_weights", "text_weights", "bpe"):
            if field in semantic:
                semantic[field] = str((path.parent / semantic[field]).resolve())
    allowed = task.get("allowed_quality_labels")
    if allowed is not None and (
        task["format"] != "collector_v06"
        or not isinstance(allowed, list)
        or not allowed
        or not all(isinstance(x, str) for x in allowed)
    ):
        raise ValueError(
            "allowed_quality_labels requires collector_v06 and a nonempty string list"
        )
    for owner, names in (
        (task, ("source",)),
        (task["runtime"], ("work_root", "base_checkpoint", "model_path", "dino_model")),
    ):
        for key in names:
            if key in owner:
                values = owner[key] if isinstance(owner[key], list) else [owner[key]]
                resolved = []
                for value in values:
                    raw = os.path.expandvars(os.path.expanduser(value))
                    if "$" in raw:
                        raise ValueError(f"Unresolved environment variable: {key}={raw}")
                    p = Path(raw)
                    resolved.append(
                        str((path.parent / p).resolve() if not p.is_absolute() else p)
                    )
                owner[key] = resolved if isinstance(owner[key], list) else resolved[0]
    robot = task["robot"]
    if robot.get("action_representation") != "absolute_eef_rot6d":
        raise ValueError(
            "This recipe requires commanded absolute EE targets with column rot6d"
        )
    if robot.get("position_unit") != "m":
        raise ValueError(
            "Explicit meter coordinates required; convert other units in the adapter"
        )
    if robot.get("arms", 1) not in (1, 2):
        raise ValueError("arms must be 1 or 2")
    if not robot.get("cameras") or not robot.get("tactile"):
        raise ValueError("Official recipe requires RGB and real tactile image streams")
    destinations = list(robot["cameras"].values()) + list(robot["tactile"].values())
    if len(set(destinations)) != len(destinations):
        raise ValueError("Camera destination keys must be unique")
    return task


def paths(task):
    root = Path(task["runtime"]["work_root"]) / task["name"]
    return dict(
        root=root,
        dataset=root / "dataset",
        pool=root / "pool",
        reports=root / "reports",
        runs=root / "runs",
    )


def accumulation(world_size):
    if world_size not in (1, 2, 4, 8):
        raise ValueError("Supported GPU/process counts: 1, 2, 4, 8")
    return 32 // world_size


def training_config(task, mode, world_size, *, require_ready=True):
    from n0_twam.configs.twam_posttrain_cfg import twam_posttrain_cfg

    cfg = deepcopy(twam_posttrain_cfg)
    cfg.update(RECIPE)
    cfg.gradient_accumulation_steps = accumulation(world_size)
    motion, ikv = MODES[mode]
    p = paths(task)
    runtime, robot = task["runtime"], task["robot"]
    cfg.dataset_path = str(p["pool"] / "train")
    cfg.val_dataset_path = str(p["pool"] / "val")
    cfg.val_interval = 9999
    cfg.obs_cam_keys = list(robot["cameras"].values())
    cfg.tactile_keys = list(robot["tactile"].values())
    cfg.tactile_sensor_id_map = {k: i for i, k in enumerate(cfg.tactile_keys)}
    cfg.used_action_channel_ids = list(range(10 * robot.get("arms", 1)))
    cfg.inverse_used_action_channel_ids = [len(cfg.used_action_channel_ids)] * 20
    for i in cfg.used_action_channel_ids:
        cfg.inverse_used_action_channel_ids[i] = i
    cfg.use_rgb_motion_tokens, cfg.use_ikv_training = motion, ikv
    cfg.kv_cache_policy = "global" if ikv else "fifo"
    cfg.ikv_train_capacity = int(task.get("features", {}).get("ikv_capacity", 4096))
    # Sparse mask granularity is independent of the attention matmul tile sizes.
    cfg.ikv_train_block_size = int(task.get("features", {}).get("ikv_block_size", 64))
    # Runtime override permits switching kernels without invalidating prepared
    # task-data fingerprints, which include every field in the task JSON.
    cfg.ikv_train_compact_attention = bool(ikv and (
        task.get("features", {}).get("ikv_compact_attention", False)
        or os.environ.get("IKV_COMPACT_ATTENTION") == "1"))
    cfg.ikv_train_compact_max_packed_keys = int(os.environ.get(
        "IKV_COMPACT_MAX_PACKED_KEYS",
        task.get("features", {}).get("ikv_compact_max_packed_keys", 65536)))
    if cfg.ikv_train_compact_max_packed_keys < 1:
        raise ValueError("ikv_compact_max_packed_keys must be positive")
    if cfg.ikv_train_block_size not in (64, 128):
        raise ValueError("ikv_block_size must be 64 or 128 for the current attention kernels")
    cfg.ikv_train_execution = (task.get("features", {}).get("ikv_training_execution", "masked")
                               if ikv else "recurrent")
    cfg.ikv_train_sample_capacity = bool(ikv and cfg.ikv_train_execution == "masked"
        and task.get("features", {}).get("ikv_sample_capacity", True))
    cfg.ikv_train_min_capacity = (int(task.get("features", {}).get("ikv_min_capacity",
                                           (cfg.ikv_train_capacity + 1) // 2)) if ikv else 0)
    if ikv:
        cfg.kv_retention = dict(cfg.kv_retention)
        cfg.kv_retention.update(query_weight=0.0, action_query_weight=0.0,
                                tactile_query_weight=0.0)
    cfg.ikv_index_root_name = "ikv_index" if ikv else None
    cfg.rgb_motion_root_name = "rgb_motion"
    cfg.rgb_motion_input_mode = "rgb"
    cfg.rgb_motion_max_tokens = (
        cfg.height
        // (16 * cfg.patch_size[1])
        * cfg.width
        // (16 * cfg.patch_size[2])
        * len(cfg.obs_cam_keys)
    )
    cfg.kv_semantic_encoder = dict(runtime.get("semantic_encoder", {}) or {})
    semantic_enabled = cfg.kv_semantic_encoder.get("backend", "dinov2") != "dinov2"
    if semantic_enabled and ikv:
        if cfg.kv_retention.get("version", 1) != 2:
            raise ValueError("semantic patch scoring requires IKV retention version 2")
        cfg.kv_retention["task_weight"] = float(task.get("features", {}).get("task_weight", 1.))
        cfg.kv_retention["content_threshold"] = float(task.get("features", {}).get("content_threshold", .9))
        manifest_path = paths(task)["dataset"] / "features.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Build semantic features before training")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("task_fingerprint") != fingerprint(task):
            raise ValueError("Semantic feature manifest differs from task configuration")
        cfg.kv_semantic_provenance = manifest.get("semantic_provenance")
        if not cfg.kv_semantic_provenance:
            raise ValueError("Feature manifest lacks semantic provenance")
        from n0_twam.preprocessing.semantic_patch import _identity
        if cfg.kv_semantic_provenance.get("assets") != _identity(cfg.kv_semantic_encoder):
            raise ValueError("Semantic model assets changed since feature build; regenerate features")
    cfg.rgb_motion_dino_model_name_or_path = runtime.get("dino_model", "")
    cfg.rgb_motion_rgb_threshold = float(
        task.get("features", {}).get("motion_threshold", 0.02)
    )
    cfg.wan22_pretrained_model_name_or_path = runtime.get(
        "model_path", runtime["base_checkpoint"]
    )
    cfg.empty_emb_path = str(
        Path(cfg.wan22_pretrained_model_name_or_path) / "empty_emb.pt"
    )
    cfg.resume_from = runtime["base_checkpoint"]
    cfg.save_root = str(p["runs"] / mode)
    cfg.eval_prompt = task["prompt"]
    cfg.seed = int(task.get("seed", 42))
    cfg.norm_stat_path = str(p["pool"] / "norm_stat_absee.json")
    norm = Path(cfg.norm_stat_path)
    if require_ready and not norm.is_file():
        raise FileNotFoundError(f"Run prepare --stage pool first: {norm}")
    if norm.exists():
        cfg.norm_stat = json.loads(norm.read_text())
    cfg.per_repo_norm_stat = {}
    cfg.task_fingerprint = fingerprint(task)
    cfg.task_mode = mode
    cfg.enable_wandb = bool(task.get("monitoring", {}).get("wandb", False))
    cfg.official_revision = OFFICIAL_REVISION
    return cfg


def source_inventory(source):
    if isinstance(source, list):
        return [{"source": path, "files": source_inventory(path)} for path in source]
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(root)
    files = (
        [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    )
    result = []
    for p in files:
        st = p.stat()
        entry = dict(
            path=str(p.relative_to(root)) if root.is_dir() else p.name,
            bytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
        )
        if root.is_file() or st.st_size < 1024 * 1024:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(block)
            entry["sha256"] = h.hexdigest()
        result.append(entry)
    return result


def conversion_identity(task):
    fields = {
        k: task.get(k)
        for k in (
            "format",
            "source",
            "prompt",
            "robot",
            "action_labels",
            "on_invalid_alignment",
            "allowed_quality_labels",
        )
    }
    fields["source_inventory"] = source_inventory(task["source"])
    return fields
