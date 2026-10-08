"""Create independent, ready-to-prepare task JSONs for the three semantic backends."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n0_twam.task_pipeline.config import load_task
from validate_semantic_backends import configurations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    original = load_task(args.task)
    for backend, encoder in configurations(args.model_root.resolve()).items():
        root = args.out_root.resolve() / backend
        root.mkdir(parents=True, exist_ok=True)
        task = deepcopy(original)
        task["runtime"]["work_root"] = str(root / "work")
        task["runtime"]["semantic_encoder"] = encoder
        task["runtime"].pop("dino_model_fingerprint_compat", None)
        task.setdefault("features", {}).update(task_weight=1., content_threshold=.9)
        dest = root / "task.json"
        if dest.exists() and json.loads(dest.read_text()) != task:
            raise FileExistsError(f"Refusing to replace a different task: {dest}")
        dest.write_text(json.dumps(task, indent=2) + "\n")
        print(dest)


if __name__ == "__main__": main()
