"""Create an independent inference override; never mutate the source config."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n0_twam.models.global_kv_retention import RetentionConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gib", type=float, default=8)
    parser.add_argument("--overflow", choices=("error", "disk"), default="error")
    parser.add_argument("--directory", default="")
    parser.add_argument("--sys2-endpoint")
    parser.add_argument("--sys2-task")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("output must differ from source")
    cfg = json.loads(args.source.read_text())
    retention = dict(cfg["kv_retention"])
    retention.update(history_storage="cpu_archive", archive_max_bytes=int(args.max_gib * 1024**3),
                     archive_overflow=args.overflow, archive_directory=args.directory,
                     query_weight=0., action_query_weight=0., tactile_query_weight=0.)
    if args.sys2_endpoint:
        if not args.sys2_task:
            raise ValueError("--sys2-endpoint requires --sys2-task")
        retention.update(score_backend="qwen_sys2",sys2_endpoint=args.sys2_endpoint,
                         sys2_task=args.sys2_task,visual_weight=0.,persistence_weight=0.,
                         class_recency_weight=0.,task_weight=0.,contact_weight=0.)
        cfg.update(kv_index_dino_online=False,kv_index_predicted_dino=False,
                   kv_neoforce_online=False,use_rgb_motion_tokens=False)
    RetentionConfig(**retention)
    cfg["kv_retention"] = retention
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(cfg, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__": main()
