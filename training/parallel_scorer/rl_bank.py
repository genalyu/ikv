"""Read the existing SFT RGB/tactile bank with an explicit causal raw-frame prefix."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import validate_packet


def load_bank(record, revision):
    spec=record['scorer_bank']; root=Path(spec['directory'])
    if spec['encoder_revision']!=revision:raise ValueError('Encoder revision mismatch')
    ready=json.loads((root/'READY.json').read_text())
    if ready['contract']!=spec['contract'] or ready['key']!=record['episode_id']:
        raise ValueError('SFT feature bank identity changed')
    ids=np.load(root/'frame_ids.npy');ids=ids[ids<=record['query_frame']]
    if not len(ids) or ids[-1]!=record['query_frame']:raise ValueError('Missing query feature')
    bank=np.load(root/'visual.npy',mmap_mode='r');views=spec['views'];n=len(ids);v=len(views)
    if bank.shape[1:]!=(v,64,1024):raise ValueError('SFT bank shape changed')
    features=torch.from_numpy(np.array(bank[:n],copy=True)).reshape(-1,1024)
    xy=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij'),-1).reshape(-1,2).float()/7
    p=dict(features=features,text=torch.load(root/'text.pt',weights_only=True,map_location='cpu'),
        times=torch.tensor(ids/spec['fps'],dtype=torch.float32).repeat_interleave(v*64),
        xy=xy.repeat(n*v,1),sensors=torch.tensor(spec['sensors']).repeat_interleave(64).repeat(n),
        modalities=torch.tensor(spec['modalities']).repeat_interleave(64).repeat(n),
        query_time=record['query_frame']/spec['fps'],encoder_revision=revision,
        ids=[f'{f}:{name}:{k}' for f in ids for name in views for k in range(64)])
    validate_packet(p);return p
