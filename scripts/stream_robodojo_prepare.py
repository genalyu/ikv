"""Overlap episode conversion and persistent GPU feature extraction."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def gpu_worker(root, gpu):
    import cv2
    import pandas as pd
    import torch
    import encode_lerobot_n0_latents as enc
    from n0_twam.task_pipeline.config import load_task, paths, fingerprint
    from n0_twam.task_pipeline.features import selected_rgb, build_payloads, _existing_sidecars_match
    from n0_twam.preprocessing.semantic_patch import load_patch_encoder
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    tasks = [load_task(root / "tasks" / (name + ".json"))
             for name in json.loads((root / "tasks.json").read_text())]
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model_path = Path(tasks[0]["runtime"]["base_checkpoint"])
    vae = enc.load_vae(model_path / "vae", torch_dtype=dtype, torch_device=device)
    tokenizer = enc.load_tokenizer(model_path / "tokenizer")
    text_encoder = enc.load_text_encoder(model_path / "text_encoder",
                                         torch_dtype=dtype, torch_device=device)
    semantic = load_patch_encoder(tasks[0]["runtime"]["semantic_encoder"],
                                 device=device, prompt=tasks[0]["prompt"])
    cache = {}
    completed = 0
    print(f"GPU {gpu}: models resident, accepting converted episodes", flush=True)
    while True:
        if (root / "CPU_FAILED.json").exists():
            raise RuntimeError("CPU conversion failed; see CPU_FAILED.json")
        found = False
        for task in tasks:
            data = paths(task)["dataset"]
            for marker in sorted((data / "conversion_episodes").glob("episode_*.json")):
                job = root / "episode_jobs" / (task["name"] + "_" + marker.stem)
                done = job.with_suffix(".done")
                if done.exists():
                    continue
                lock = job.with_suffix(".lock")
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    continue
                os.close(fd)
                try:
                    idx = int(marker.stem.split("_")[1])
                    chunk = f"chunk-{idx // 1000:03d}"
                    parquet = data / "data" / chunk / (marker.stem + ".parquet")
                    length = len(pd.read_parquet(parquet, columns=["frame_index"]))
                    keys = list(task["robot"]["cameras"].values())
                    outputs = []
                    for key in keys:
                        out = data / "latents" / chunk / key / f"{marker.stem}_0_{length}.pth"
                        if not out.exists():
                            request = enc.EpisodeRequest(idx, length, task["prompt"],
                                data / "videos" / chunk / key / (marker.stem + ".mp4"),
                                out, 0.0, length / 30)
                            frames, ids = enc.extract_episode_frames(request=request,
                                width=256, height=256, target_fps=10, ori_fps=30)
                            with torch.inference_mode():
                                latent, nf, nh, nw, provenance = enc.encode_video(
                                    frames, ids, vae, device, dtype)
                                text = enc.get_text_emb(task["prompt"], tokenizer,
                                    text_encoder, device, dtype, 512, cache)
                            payload = dict(latent=latent.to(dtype), latent_num_frames=int(nf),
                                latent_height=int(nh), latent_width=int(nw),
                                video_num_frames=len(ids), video_height=256, video_width=256,
                                text_emb=text.to(dtype), text=task["prompt"], frame_ids=list(ids),
                                temporal_provenance=provenance, start_frame=0, end_frame=length,
                                fps=10, ori_fps=30)
                            out.parent.mkdir(parents=True, exist_ok=True)
                            tmp = out.with_suffix(".tmp")
                            torch.save(payload, tmp)
                            tmp.replace(out)
                        outputs.append(out)
                    reference = torch.load(outputs[0], map_location="cpu", weights_only=True)
                    ids = reference["frame_ids"]
                    anchors = reference["temporal_provenance"]["latent_anchor_indices"]
                    semantic.set_prompt(task["prompt"])
                    motion_path = data / "rgb_motion" / chunk / outputs[0].name
                    dense_path = data / "ikv_index" / chunk / outputs[0].name
                    threshold = float(task.get("features", {}).get("motion_threshold", .02))
                    valid = _existing_sidecars_match(motion_path, dense_path, ids, anchors, keys, threshold)
                    if valid:
                        prior = torch.load(dense_path, map_location="cpu", weights_only=True)
                        valid = prior.get("semantic_provenance") == semantic.provenance()
                    if not valid:
                        streams = {key: selected_rgb(data / "videos" / chunk / key /
                                   (marker.stem + ".mp4"), ids) for key in keys}
                        with torch.inference_mode():
                            motion, dense = build_payloads(streams, ids, anchors, semantic, threshold)
                        for path, payload in ((motion_path, motion), (dense_path, dense)):
                            path.parent.mkdir(parents=True, exist_ok=True)
                            tmp = path.with_suffix(".tmp")
                            torch.save(payload, tmp)
                            tmp.replace(path)
                    atomic_json(done, dict(task=task["name"], episode=idx, gpu=gpu,
                        latent=str(outputs[0]), index=str(dense_path),
                        task_fingerprint=fingerprint(task)))
                    completed += 1
                    print(f"GPU {gpu}: READY {task['name']}/{idx}, worker_total={completed}", flush=True)
                finally:
                    lock.unlink(missing_ok=True)
                found = True
                break
            if found:
                break
        if not found:
            if (root / "CONVERSION_COMPLETE").exists():
                pending = sum(1 for task in tasks
                    for marker in (paths(task)["dataset"] / "conversion_episodes").glob("episode_*.json")
                    if not (root / "episode_jobs" / (task["name"] + "_" + marker.stem + ".done")).exists())
                if not pending:
                    return
            time.sleep(2)

def orchestrate(root, cpu_workers, role="all"):
    from n0_twam.task_pipeline.config import load_task, paths, fingerprint
    from n0_twam.task_pipeline.cli import check
    (root / "episode_jobs").mkdir(exist_ok=True)
    if role != "cpu":
        for lock in (root / "episode_jobs").glob("*.lock"):
            lock.unlink()
    names = json.loads((root / "tasks.json").read_text())
    children, handles = [], []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    try:
        for gpu in (range(4) if role != "cpu" else []):
            log = (root / "logs" / f"gpu_features_{gpu}.log").open("a")
            handles.append(log)
            children.append(subprocess.Popen([sys.executable, "-u", __file__,
                "--root", str(root), "--gpu", str(gpu)],
                cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=log, stderr=subprocess.STDOUT))
        def convert(name):
            with (root / "logs" / (name + ".log")).open("a") as log:
                subprocess.run([sys.executable, "script/task_pipeline.py", "prepare",
                    "--task", str(root / "tasks" / (name + ".json")), "--stage", "convert"],
                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("CONVERTED_TASK", name, flush=True)
        if role != "gpu":
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as pool:
                futures = [pool.submit(convert, name) for name in names]
                while not all(f.done() for f in futures):
                    for child in children:
                        if child.poll() is not None and child.returncode:
                            raise RuntimeError(f"GPU worker failed: pid={child.pid}")
                    time.sleep(3)
                for future in futures:
                    future.result()
            (root / "CONVERSION_COMPLETE").write_text("all tasks converted")
        if role == "cpu":
            return
        while any(child.poll() is None for child in children):
            for child in children:
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"GPU worker failed: pid={child.pid}")
            time.sleep(2)
        if any(child.returncode for child in children):
            raise RuntimeError("GPU preprocessing failed")
        task = load_task(root / "all35.json")
        for name in names:
            item = load_task(root / "tasks" / (name + ".json"))
            data = paths(item)["dataset"]
            dense = sorted((data / "ikv_index").glob("chunk-*/*.pth"))
            if len(dense) != 100:
                raise RuntimeError(f"Incomplete GPU corpus: {name}")
            import torch
            provenance = torch.load(dense[0], map_location="cpu", weights_only=True)["semantic_provenance"]
            atomic_json(data / "features.json", dict(task_fingerprint=fingerprint(item),
                episodes=100, dino_model=item["runtime"]["dino_model"],
                semantic_provenance=provenance, neoforce="unavailable; zero width",
                threshold=item.get("features", {}).get("motion_threshold", .02)))
            subprocess.run([sys.executable, "script/task_pipeline.py", "prepare",
                "--task", str(root / "tasks" / (name + ".json")), "--stage", "pool"],
                cwd=REPO, env=env, check=True)
        errors = check(task, "ikv")
        if errors:
            raise RuntimeError(str(errors[:25]))
        p = paths(task)
        for split in ("train", "val"):
            (p["pool"] / split).mkdir(parents=True, exist_ok=True)
            for name in names:
                link = p["pool"] / split / name
                if not link.exists():
                    link.symlink_to(root / "prepared" / name / "dataset")
        (root / "PREPROCESS_COMPLETE").write_text("3500 episodes verified")
        while not (root / "FOUR_GPU_PREFLIGHT_PASSED").exists():
            time.sleep(10)
        with (root / "logs" / "train.log").open("a") as log:
            subprocess.run([sys.executable, "script/task_pipeline.py", "train",
                "--task", str(root / "all35.json"), "--mode", "ikv", "--gpus", "4"],
                cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES="0,1,2,3",
                    IKV_COMPACT_ATTENTION="1", IKV_COMPACT_MAX_PACKED_KEYS="65536"),
                stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for log in handles:
            log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-workers", type=int, default=24)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--role", choices=["all", "cpu", "gpu"], default="all")
    args = parser.parse_args()
    if args.gpu is None:
        try:
            orchestrate(args.root, args.cpu_workers, args.role)
        except Exception as error:
            atomic_json(args.root / (args.role.upper() + "_FAILED.json"), {"error": repr(error)})
            raise
    else:
        gpu_worker(args.root, args.gpu)
