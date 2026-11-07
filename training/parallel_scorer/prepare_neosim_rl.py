import json,hashlib,pathlib,torch,numpy as np
from easydict import EasyDict
from n0_twam.dataset.lerobot_latent_dataset import LatentLeRobotDataset
import argparse
parser=argparse.ArgumentParser(description="Verify SFT banks and build causal NeoSim FM manifests")
parser.add_argument('--output',required=True);parser.add_argument('--sft-root',required=True);parser.add_argument('--n0-root',required=True)
a=parser.parse_args();P=pathlib.Path(a.output);S=pathlib.Path(a.sft_root);N=pathlib.Path(a.n0_root)
P.mkdir(parents=True,exist_ok=False)
C=EasyDict(json.loads((N/'runs/ikv/resolved_config.json').read_text()));contract=json.loads((S/'ENCODING_CONTRACT.json').read_text())
episodes=[e for e in json.loads((S/'EPISODES.json').read_text()) if e['dataset']=='neosim_mem']
(P/'batches').mkdir(exist_ok=True);records={'train':[],'validation':[]};ds={}
for e in episodes:
 task=e['task'];ep=int(e['key'].rsplit('_',1)[-1]);root=pathlib.Path(e['feature_dir']);ready=json.loads((root/'READY.json').read_text())
 if ready['contract']!=e['contract_sha256'] or ready['key']!=e['key']:raise ValueError('bank identity')
 # Pin existing SFT artifacts, without modifying the feature bank.
 for name,expected in ready['hashes'].items():
  h=hashlib.sha256()
  with (root/name).open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  if h.hexdigest()!=expected:raise ValueError('bank hash mismatch')
 if task not in ds:ds[task]=LatentLeRobotDataset(str(N/'pool/train'/task),C)
 d=ds[task]
 for view in e['views']:
  expected=d.root/'videos'/f'chunk-{ep//int(d.meta.info.get("chunks_size",1000)):03d}'/view['view']/f'episode_{ep:06d}.mp4'
  if pathlib.Path(view['path']).resolve()!=expected.resolve():raise ValueError('SFT/native video source mismatch')
 hits=[i for i,m in enumerate(d.new_metas) if m['episode_index']==ep and m['start_frame']==0]
 if len(hits)!=1:raise ValueError(('complete episode required',task,ep,hits))
 i=hits[0];m=d.new_metas[i];raw=d._get_range_latent_data(0,m['end_frame'],ep)
 keys=d.used_video_keys;anchors=raw[keys[0]+'.temporal_provenance']['latent_anchor_frame_ids'];F=len(anchors)
 for key in keys:
  if raw[key+'.temporal_provenance']['latent_anchor_frame_ids']!=anchors:raise ValueError('camera time alignment')
 batch=d[i];source=np.load(root/'frame_ids.npy');views=[v['view'] for v in e['views']];V=len(views)
 candidates=[j for j in range(8,F) if anchors[j-1] in source]
 if not candidates:continue
 queries=sorted(set(candidates[round(f*(len(candidates)-1))] for f in (.0,.33,.67,1.)))
 for target in queries:
  q=anchors[target-1];f=target+1;packet_index={int(frame):j for j,frame in enumerate(source[source<=q])}
  vmap=[]
  for j in range(f):
   for y in range(8):
    for camera in keys:
     for x in range(8):
      row=-1 if j==target else (packet_index[anchors[j]]*V+views.index(camera))*64+y*8+x
      vmap.append([row]*4)
  tmap=[]
  for sensor in d.used_tactile_keys:
   for j in range(f):
    for y in range(4):
     for x in range(4):
      tmap.append([-1]*4 if j==target else [(packet_index[anchors[j]]*V+views.index(sensor))*64+(y*2+dy)*8+x*2+dx for dy,dx in [(0,0),(0,1),(1,0),(1,1)]])
  b={}
  for k,value in batch.items():
   if k in ('latents','actions','actions_mask'):value=value[:,:f]
   elif k in ('tactile_global_latent','tactile_local_latent'):value=value[:,:,:f]
   elif k not in ('text_emb','tactile_sensor_ids','tactile_cond_drop'):value=value[:f]
   b[k]=value.contiguous().unsqueeze(0)
  path=P/'batches'/f'{task}-{ep:06d}-t{target}.pt';torch.save(b,path)
  row=dict(id=f'{e["key"]}:t{target}',episode_id=e['key'],task=task,query_frame=int(q),target_raw_frame=int(anchors[target]),
   n0_batch=str(path),alignment_provenance=dict(kind='stored-causal-chunk-end',anchors=anchors[:f],video_order='frame-row-camera-column',tactile_order='sensor-frame-row-column',tactile_score='mean of aligned 2x2 Qwen score cells'),
   video_candidate_map=vmap,tactile_candidate_map=tmap,
   scorer_bank=dict(directory=str(root),encoder_revision=contract['encoder_revision'],contract=e['contract_sha256'],views=views,fps=e['fps'],sensors=[contract['sensor_to_id'][v] for v in views],modalities=[int('tact' in v.lower()) for v in views],hashes=ready['hashes']))
  records[e['split']].append(row)
 print(json.dumps(dict(event='episode',task=task,episode=ep,split=e['split'],queries=len(queries),frames=F)),flush=True)
 for split,rows in records.items():
  temp=P/(split+'.aligned.jsonl.tmp');temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(P/(split+'.aligned.jsonl'))
a={r['episode_id'] for r in records['train']};b={r['episode_id'] for r in records['validation']};assert not a&b
assert records['train'] and records['validation'], 'Both splits required'
(P/'DATA_READY.json').write_text(json.dumps({k:len(v) for k,v in records.items()},indent=2))
