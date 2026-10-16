"""Unified task CLI. GPU work is only performed by explicit prepare/train stages."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from .config import (
    load_task,
    paths,
    training_config,
    MODES,
    OFFICIAL_REVISION,
    fingerprint,
)

REPO = Path(__file__).resolve().parents[2]


def run(command):
    print(shlex.join([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], check=True, cwd=REPO)


def check(task, mode="baseline"):
    import torch
    from n0_twam.dataset.ikv_index import load_dense_index

    p = paths(task)
    root = p["dataset"]
    errors = []
    info_path = root / "meta/info.json"
    if not info_path.exists():
        return ["Missing converted dataset: prepare --stage convert"]
    info = json.loads(info_path.read_text())
    if any(MODES[mode]):
        f = root / "features.json"
        if not f.exists() or json.loads(f.read_text()).get(
            "task_fingerprint"
        ) != fingerprint(task):
            errors.append(
                "Feature build is missing or stale for this task; run prepare --stage features"
            )
    if (
        task.get("expected_episodes")
        and info.get("source_total_episodes", info["total_episodes"])
        != task["expected_episodes"]
    ):
        errors.append(
            f"Expected {task['expected_episodes']} episodes; found {info['total_episodes']}. Download full dataset or explicitly configure a subset."
        )
    if (
        task.get("expected_usable_episodes")
        and info["total_episodes"] != task["expected_usable_episodes"]
    ):
        errors.append(
            f"Expected {task['expected_usable_episodes']} usable episodes; "
            f"converted {info['total_episodes']}"
        )
    cameras = list(task["robot"]["cameras"].values())
    tactile = list(task["robot"]["tactile"].values())
    if info["fps"] != 30:
        errors.append("Official action clock must be 30 Hz")
    for key in cameras + tactile:
        if key not in info["features"]:
            errors.append(f"Missing camera {key}")
    for ep in (root / "meta/episodes.jsonl").read_text().splitlines():
        ep = json.loads(ep)
        idx = ep["episode_index"]
        chunk = f"chunk-{idx // 1000:03d}"
        name = f"episode_{idx:06d}_0_{ep['length']}.pth"
        reference = None
        for key in cameras:
            f = root / "latents" / chunk / key / name
            if not f.exists():
                errors.append(f"Missing RGB latent {f}")
                continue
            v = torch.load(f, map_location="cpu", weights_only=True)
            if v.get("text") != task["prompt"]:
                errors.append(f"Stale prompt {f}")
            if reference is not None and reference["frame_ids"] != v["frame_ids"]:
                errors.append(f"Misaligned cameras {f}")
            reference = v if reference is None else reference
        for stream in ("global", "local"):
            for key in tactile:
                if not (
                    root / "latents_tactile" / stream / chunk / key / name
                ).exists():
                    errors.append(f"Missing {stream} tactile: {idx}/{key}")
        motion, ikv = MODES[mode]
        if reference is not None:
            frames = reference["latent_num_frames"]
            grid = (
                reference["latent_height"] // 2,
                reference["latent_width"] // 2 * len(cameras),
            )
            if motion:
                f = root / "rgb_motion" / chunk / name
                if not f.exists():
                    errors.append(f"Missing motion {f}")
                else:
                    v = torch.load(f, map_location="cpu", weights_only=True)
                    if (
                        v["frame_ids"] != reference["frame_ids"]
                        or v["camera_keys"] != cameras
                    ):
                        errors.append(f"Misaligned motion {f}")
                    if v["motion_indices"].shape[0] != frames:
                        errors.append(f"Wrong motion frame count {f}")
            if ikv:
                f = root / "ikv_index" / chunk / name
                try:
                    load_dense_index(
                        f,
                        camera_keys=cameras,
                        patch_size=[1, 2, 2],
                        grid_shape=grid,
                        latent_frame_ids=reference["frame_ids"],
                        full_frames=frames,
                    )
                except (FileNotFoundError, ValueError) as e:
                    errors.append(str(e))
    if not (p["pool"] / "norm_stat_absee.json").exists():
        errors.append("Missing normalization: prepare --stage pool")
    for split in ("train", "val"):
        if not (p["pool"] / split / root.name).exists():
            errors.append(f"Missing {split} pool link")
    for part in ("transformer", "vae", "tokenizer", "text_encoder", "empty_emb.pt"):
        base = Path(
            task["runtime"]["base_checkpoint"]
            if part == "transformer"
            else task["runtime"].get("model_path", task["runtime"]["base_checkpoint"])
        )
        if not (base / part).exists():
            errors.append(f"Missing base component: {base / part}")
    return errors


def prepare(task, stage, device, dry_run):
    p = paths(task)
    root = p["dataset"]
    runtime = task["runtime"]
    model = runtime.get("model_path", runtime["base_checkpoint"])
    rgb = [
        sys.executable,
        "script/encode_lerobot_n0_latents.py",
        "--dataset-root",
        root,
        "--model-path",
        model,
        "--target-fps",
        "10",
        "--height",
        "256",
        "--width",
        "256",
        "--device",
        device,
        "--prompt",
        task["prompt"],
        "--video-keys",
        *task["robot"]["cameras"].values(),
    ]
    tactile = [
        sys.executable,
        "script/encode_tactile_latent.py",
        "--dataset-root",
        root,
        "--model-path",
        model,
        "--target-fps",
        "10",
        "--height",
        "128",
        "--width",
        "128",
        "--device",
        device,
        "--mode",
        "both",
        "--local-mode",
        "current",
        "--tactile-keys",
        *task["robot"]["tactile"].values(),
    ]
    pool = [
        sys.executable,
        "script/build_task_pool.py",
        "--data",
        root,
        "--pool",
        p["pool"],
        "--mode",
        "absee",
        "--anchor-cam",
        next(iter(task["robot"]["cameras"].values())),
    ]
    if task["robot"].get("arms", 1) == 2:
        pool.append("--dual-arm")
    for name in ("convert", "rgb", "tactile", "features", "pool"):
        if stage not in (name, "all"):
            continue
        if dry_run:
            print(
                shlex.join(
                    [
                        str(x)
                        for x in {"rgb": rgb, "tactile": tactile, "pool": pool}[name]
                    ]
                )
                if name in ("rgb", "tactile", "pool")
                else f"{name}: {root}"
            )
            continue
        if name == "convert":
            from .convert import convert

            convert(task)
        elif name == "features":
            from .features import build_features

            build_features(task, device)
        else:
            run({"rgb": rgb, "tactile": tactile, "pool": pool}[name])


def snapshot(task, mode, world_size, destination):
    cfg = training_config(task, mode, world_size)
    from n0_twam.configs.twam_posttrain_cfg import twam_posttrain_cfg

    official = dict(twam_posttrain_cfg)
    changes = {
        k: {"template": str(official.get(k)), "resolved": str(v)}
        for k, v in cfg.items()
        if str(official.get(k)) != str(v)
    }
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "resolved_config.json").write_text(
        json.dumps(dict(cfg), indent=2, default=str)
    )
    (dest / "task.json").write_text(json.dumps(task, indent=2))
    (dest / "recipe_diff.json").write_text(
        json.dumps(
            dict(official_revision=OFFICIAL_REVISION, differences=changes), indent=2
        )
    )
    hashes = {}
    for f in [
        paths(task)["pool"] / "norm_stat_absee.json",
        paths(task)["dataset"] / "conversion.json",
        paths(task)["dataset"] / "features.json",
        *sorted((paths(task)["dataset"] / "meta").glob("*")),
    ]:
        if f.is_file():
            hashes[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    (dest / "provenance.json").write_text(
        json.dumps(
            dict(
                core_commit=commit,
                hashes=hashes,
                official_revision=OFFICIAL_REVISION,
                effective_batch=world_size
                * cfg.batch_size
                * cfg.gradient_accumulation_steps,
                task_fingerprint=fingerprint(task),
            ),
            indent=2,
        )
    )
    serve = dict(
        use_rgb_motion_tokens=cfg.use_rgb_motion_tokens,
        kv_cache_policy=cfg.kv_cache_policy,
        use_ikv_training=cfg.use_ikv_training,
        rgb_motion_input_mode="rgb",
        rgb_motion_rgb_threshold=cfg.rgb_motion_rgb_threshold,
        rgb_motion_online_preprocess=cfg.use_rgb_motion_tokens,
        obs_cam_keys=cfg.obs_cam_keys,
        tactile_keys=cfg.tactile_keys,
        used_action_channel_ids=cfg.used_action_channel_ids,
        eval_prompt=cfg.eval_prompt,
        action_delta_mode="none",
        pi05_action_horizon=12,
        action_per_frame=12,
        use_local_tactile=True,
        local_tactile_mode="current",
        tactile_global_zero=False,
        norm_stat=cfg.norm_stat,
        rgb_motion_max_tokens=cfg.rgb_motion_max_tokens,
        rgb_motion_dino_model_name_or_path=cfg.rgb_motion_dino_model_name_or_path,
        ikv_train_capacity=cfg.ikv_train_capacity,
        kv_retention=dict(cfg.kv_retention),
        kv_index_dino_online=cfg.use_ikv_training,
        kv_semantic_encoder=dict(getattr(cfg, "kv_semantic_encoder", {}) or {}),
        kv_semantic_provenance=getattr(cfg, "kv_semantic_provenance", None),
        kv_index_dino_model_name_or_path=cfg.rgb_motion_dino_model_name_or_path,
        inverse_used_action_channel_ids=cfg.inverse_used_action_channel_ids,
        tactile_sensor_id_map=dict(cfg.tactile_sensor_id_map),
        action_norm_method=cfg.action_norm_method,
        norm_stat_path=cfg.norm_stat_path,
    )
    (dest / "serve_overrides.json").write_text(json.dumps(serve, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["analyze", "prepare", "check", "train"])
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument(
        "--stage",
        choices=["convert", "rgb", "tactile", "features", "pool", "all"],
        default="convert",
    )
    ap.add_argument("--mode", choices=list(MODES), default="baseline")
    ap.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args(argv)
    task = load_task(args.task)
    if args.command == "analyze":
        from .report import analyze

        analyze(task)
    elif args.command == "prepare":
        prepare(task, args.stage, args.device, args.dry_run)
    elif args.command == "check":
        errors = check(task, args.mode)
        print(json.dumps({"ready": not errors, "errors": errors}, indent=2))
        if errors:
            raise SystemExit(1)
    else:
        cfg = training_config(task, args.mode, args.gpus, require_ready=False)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.gpus}",
            "-m",
            "n0_twam.train",
            "--task-config",
            str(args.task.resolve()),
            "--mode",
            args.mode,
        ]
        if args.resume:
            command.extend(["--resume-training", str(args.resume.resolve())])
        if args.dry_run:
            print(
                json.dumps(
                    dict(
                        gpus=args.gpus,
                        gradient_accumulation=cfg.gradient_accumulation_steps,
                        effective_batch=32,
                        optimizer_updates=2000,
                        mode=args.mode,
                    ),
                    indent=2,
                )
            )
            print(shlex.join(command))
            return
        errors = check(task, args.mode)
        if errors:
            raise RuntimeError("Training preflight failed:\n" + "\n".join(errors[:25]))
        import torch

        if torch.cuda.device_count() != args.gpus:
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly --gpus devices")
        if Path(cfg.save_root).exists() and not args.resume:
            raise FileExistsError(
                "Existing run directory; resume it or use a new work root"
            )
        if args.resume:
            manifest = json.loads((args.resume / "complete.json").read_text())
            expected = dict(
                world_size=args.gpus, task_fingerprint=fingerprint(task), mode=args.mode
            )
            for key, value in expected.items():
                if manifest[key] != value:
                    raise ValueError(f"Resume {key} mismatch")
            if args.resume.resolve().parents[2] != Path(cfg.save_root).resolve():
                raise ValueError("Resume checkpoint must belong to this run directory")
        else:
            snapshot(task, args.mode, args.gpus, cfg.save_root)
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(REPO)
            + os.pathsep
            + str(REPO / "n0_twam")
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TOKENIZERS_PARALLELISM"] = "false"
        if cfg.enable_wandb:
            if not env.get("WANDB_API_KEY"):
                raise RuntimeError("WandB enabled but WANDB_API_KEY is missing")
            env.setdefault(
                "WANDB_PROJECT",
                task.get("monitoring", {}).get("project", "ikv-phone-weight"),
            )
            env.setdefault("WANDB_RUN_NAME", f"{task['name']}-{args.mode}")
        subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
