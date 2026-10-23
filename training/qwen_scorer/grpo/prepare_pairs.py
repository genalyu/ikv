"""Export causal query/target prefixes from the existing N0 dataset.

Pair specifications must contain verified RGB/latent temporal and spatial
provenance. Never guess camera tiling or temporal VAE compression.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from torch.utils.data import default_collate
from .train import load_symbol


def crop_sample(sample, end):
    # Unbatched existing dataset tensor axes.
    axes = dict(latents=1, actions=1, actions_mask=1,
                tactile_global_latent=2, tactile_local_latent=2,
                dense_dino_features=0, dino_features=0, neoforce_features=0,
                frame_neoforce_features=0, task_relevance=0)
    static = {"text_emb","tactile_sensor_ids","_ikv_sample_index"}
    result = {}
    for key,value in sample.items():
        if not torch.is_tensor(value):
            raise ValueError("Expected tensor-only N0 sample: "+key)
        if key in axes:
            axis = axes[key]
            if value.shape[axis] < end+1:
                raise ValueError("Incomplete temporal alignment: "+key)
            index = [slice(None)] * value.ndim
            index[axis] = slice(0,end+1)
            result[key] = value[tuple(index)].clone()
        elif key in static:
            result[key] = value
        else:
            raise ValueError("Unrecognized field; add a verified axis mapping: "+key)
    return result


def export(config_symbol, pairs, output):
    from n0_twam.dataset import MultiLatentLeRobotDataset
    config = deepcopy(load_symbol(config_symbol))
    # Prefix slicing must start at episode origin, not a random training crop.
    config.max_latent_frames = 0
    config.max_tactile_frames = 0
    dataset = MultiLatentLeRobotDataset(config=config)
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    records = []
    for number,pair in enumerate(pairs):
        record = dict(pair["scorer_record"])
        if not pair.get("alignment_provenance"):
            raise ValueError("Verified alignment provenance is required")
        sample = dataset[int(pair["dataset_index"])]
        end = int(pair["target_latent_frame"])
        raw_ids = pair["latent_raw_frame_ids"]
        source_max = pair["latent_source_max_raw_frame_ids"]
        if end < 1 or len(raw_ids) <= end or len(source_max) <= end:
            raise ValueError("History plus one target required")
        if any(a>b for a,b in zip(raw_ids,raw_ids[1:])):
            raise ValueError("Latent frame provenance must be chronological")
        query = record["query_frame"]
        if max(raw_ids[:end] + source_max[:end]) > query or raw_ids[end] <= query:
            raise ValueError("Expert future leaks into observation history")
        cropped = crop_sample(sample,end)
        B = default_collate([cropped])
        patch = config.patch_size
        if patch[0] != 1:
            raise ValueError("Temporal patch size must be 1")
        per_frame = (B["latents"].shape[-2]//patch[1])*(B["latents"].shape[-1]//patch[2])
        spatial = pair["video_patch_coordinates"]
        if len(spatial) != per_frame:
            raise ValueError("Spatial map differs from native N0 patch layout")
        if any(len(v)!=3 or not (0<=v[0]<3 and 0<=v[1]<8 and 0<=v[2]<8) for v in spatial):
            raise ValueError("Expected scorer camera/row/column coordinates")
        coordinates = [[raw_ids[f],*address] for f in range(end) for address in spatial]
        coordinates += [[-1]*4 for _ in spatial]
        batch_path = output/f"{number:06d}.pt"
        map_path = output/f"{number:06d}.coordinates.json"
        torch.save(B,batch_path)
        map_path.write_text(json.dumps(coordinates))
        record.update(n0_batch=str(batch_path),kv_coordinates=str(map_path),
                      target_raw_frame=raw_ids[end],
                      alignment_provenance=pair["alignment_provenance"])
        records.append(record)
    (output/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n0-config",required=True)
    parser.add_argument("--pairs",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    pairs=[json.loads(s) for s in Path(args.pairs).read_text().splitlines() if s.strip()]
    export(args.n0_config,pairs,args.output)


if __name__=="__main__":
    main()
