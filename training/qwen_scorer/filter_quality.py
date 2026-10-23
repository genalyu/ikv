"""Revalidate frozen labels and select informative intact examples, preserving histories."""
from pathlib import Path
import json, hashlib, collections, time, argparse
import numpy as np
from common import decode_scores, CAMS
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
args=parser.parse_args()
D=Path(args.dataset); OUT=D/'quality_v1'; OUT.mkdir(exist_ok=True)
def real(p): return Path(p)
rgb={}; seen=set(); episodes={}; reject=[]; kept=[]; stats=collections.Counter()
for split in ('train','validation','test'):
 for line in (D/f'{split}.ready.jsonl').read_text().splitlines():
  r=json.loads(line); reasons=[]
  try:
   assert r['id'] not in seen, 'duplicate_id'
   seen.add(r['id'])
   assert episodes.setdefault(r['episode_id'],split)==split,'episode_leakage'
   q=r['query_frame']; n=q+1
   assert type(q) is int and q>=0 and r['fps']>0,'invalid_frame_or_fps'
   assert r['split']==split and r['num_images']==3*n and r['camera_order']==CAMS,'contract'
   assert 0<r['target_tokens']<r['sequence_length']<=524288,'context'
   assert hashlib.sha256(r['target'].encode()).hexdigest()==r['target_sha256'],'target_hash'
   a=decode_scores(json.loads(r['target']),n)
   meta=json.loads(real(r['teacher_meta']).read_text())
   assert meta['fingerprint']==r['teacher_fingerprint'] and meta['frame']==q and meta['candidate_count']==a.size,'teacher_meta'
   with np.load(real(r['teacher_scores'])) as z: assert np.array_equal(a,z['scores']),'teacher_scores'
   p=real(r['rgb_npy'])
   if str(p) not in rgb:
    arr=np.load(p,mmap_mode='r')
    assert arr.ndim==5 and arr.shape[1:]==(3,256,256,3) and arr.dtype==np.uint8,'rgb_shape'
    rgb[str(p)]=arr
   arr=rgb[str(p)]; assert len(arr)>=n,'missing_rgb'
   for f in set((0,q//2,q)):
    for cam in range(3):
     image=np.asarray(arr[f,cam])
     assert not (image.max()<=1 or image.min()>=254),'blank_rgb'
   unique=np.unique(a); current=np.unique(a[-1])
   if len(unique)<3 or int(unique[-1])-int(unique[0])<20: reasons.append('low_label_information')
   if len(current)<2 or int(current[-1])-int(current[0])<10: reasons.append('undifferentiated_current_frame')
  except Exception as exc: reasons.append('invalid:'+str(exc))
  if reasons:
   reject.append(dict(id=r['id'],split=split,reasons=reasons)); stats.update(reasons)
  else: kept.append(r)
 print(json.dumps(dict(split=split,checked=len(seen),kept=len(kept),rejected=len(reject))),flush=True)
hashes={}
for split in ('train','validation','test'):
 p=OUT/f'{split}.ready.jsonl'
 p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept if r['split']==split))
 hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'rejected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reject))
report=dict(created=time.time(),source_samples=len(seen),accepted_queries=len(kept),rejected=len(reject),reasons=dict(stats),split_samples=dict(collections.Counter(r['split'] for r in kept)),file_sha256=hashes,rgb_episodes=len(rgb),max_length=max(r['sequence_length'] for r in kept),criteria=['strict schema/bounds and target hash','exact teacher score equality and metadata','episode split isolation','RGB uint8 shape and first/middle/current nonblank checks','at least 3 effective scores and score range >=20','current-frame score range >=10'],limitations='Heuristic quality screen, not independent semantic relabeling; RGB pixel check samples three frames per prefix.',preserved='all original frames, 3 cameras, 256 pixels and full targets')
(OUT/'QUALITY_DATASET_READY.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
