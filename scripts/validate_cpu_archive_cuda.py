"""Real CUDA archive smoke test using tiny two-layer MoT, no downloaded weights."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import torch
from test_cpu_kv_archive import model, policy
from test_global_kv_retention import model_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    torch.set_num_threads(2)
    m = model().to(device="cuda", dtype=torch.bfloat16)
    cfg = policy(m).config
    from dataclasses import asdict
    m.create_empty_cache("test", 2, 16, 8, torch.device("cuda"), torch.bfloat16, 1)
    m.configure_global_retention("test", **asdict(cfg))
    def forward(time, mode=2):
        data = model_input(time)
        data = {key: value.to(device="cuda", dtype=torch.bfloat16) if value.is_floating_point()
                else value.cuda() for key, value in data.items()}
        return m(data, update_cache=mode, cache_name="test")
    with torch.no_grad():
        for t in range(4): forward(t)
        p = policy(m)
        record = p.archive.records[0]
        assert all(x.is_pinned() for pair in record["kv"].values() for x in pair)
        uid = record["metadata"]["token_uid"][0].item()
        def ranked(self, slots, **kwargs):
            return (self.data["token_uid"][slots] == uid).float() * 100
        with patch.object(type(p), "scores", ranked): forward(5)
        active = m.get_global_retention("test")
        slots = active["slot_indices"][active["token_uid"] == uid]
        assert len(slots) == 1
        for layer, attention in enumerate(m.mot.shared_attn):
            for name, expected in zip(("k", "v"), record["kv"][layer]):
                torch.testing.assert_close(attention.attn_caches["test"][name][:, slots].cpu(),
                                           expected[:, :1], rtol=0, atol=0)
        count = p.archive.stats["archived_tokens"]
        forward(6, mode=1)
        assert p.archive.stats["archived_tokens"] == count
        m.clear_pred_cache("test")
        stats = dict(p.archive.stats)
        m.clear_cache("test")
    report = dict(status="passed", device=torch.cuda.get_device_name(), dtype="bfloat16", stats=stats,
                  checks=["pinned D2H archive", "exact H2D recall", "prediction isolation", "reset"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__": main()
