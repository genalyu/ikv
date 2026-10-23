"""Freeze validated teacher labels; episode-stratified splits; never truncate RGB history."""
from pathlib import Path
import argparse,json,hashlib,concurrent.futures as cf,collections,time
import numpy as np
from common import decode_scores,CAMS

def main():
 p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=24);args=p.parse_args()
 root=Path(args.labels).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 selection=json.loads((root/'selected_20_per_task.json').read_text());run=root/'runs/rgb16_full_history_8x8_v1'
 split_by_ep={};jobs=[];rgb_jobs=[]
 for task in selection['selection']:
  eps=sorted(task['selected'],key=lambda e:hashlib.sha256(('qwen-scorer-42:'+e['episode_id']).encode()).hexdigest())
  for ordinal,e in enumerate(eps):
   split='validation' if ordinal<2 else 'test' if ordinal<4 else 'train';eid=e['episode_id'];split_by_ep[eid]=split
   manifest_path=root/'episodes'/eid/'manifest.json'
   if not manifest_path.exists():continue
   ep=json.loads(manifest_path.read_text());assert ep['temporal_sampling']=='none' and ep['cameras']==CAMS
   valid=[]
   for q in sorted(set(int(x) for x in np.linspace(0,e['source_frames']-1,16,dtype=int))):
    folder=run/eid/f'{q:06d}'
    if (folder/'meta.json').exists():jobs.append((ep,q,folder,split));valid.append(q)
   if valid:rgb_jobs.append(dict(episode_id=eid,videos=ep['videos'],frames=max(valid)+1,path=str(out/'rgb'/f'{eid}.npy')))
 def check(job):
  ep,q,folder,split=job
  try:
   meta=json.loads((folder/'meta.json').read_text());obj=json.loads((folder/'assignment.json').read_text())
   a=decode_scores(obj,q+1);saved=np.load(folder/'scores.npz')['scores']
   assert a.shape==saved.shape and np.array_equal(a,saved)
   assert meta['candidate_count']==a.size and meta['frame']==q
   target=json.dumps(obj,separators=(',',':'),ensure_ascii=False)
   return dict(id=f'{ep["episode_id"]}:q{q}',episode_id=ep['episode_id'],task=ep['task'],query_frame=q,fps=ep['fps'],num_images=3*(q+1),camera_order=CAMS,rgb_npy=str(out/'rgb'/f'{ep["episode_id"]}.npy'),target=target,split=split,teacher_meta=str(folder/'meta.json'),teacher_scores=str(folder/'scores.npz'),teacher_fingerprint=meta['fingerprint'],target_sha256=hashlib.sha256(target.encode()).hexdigest()),None
  except Exception as exc:return None,dict(path=str(folder),error=str(exc))
 records=[];errors=[]
 with cf.ThreadPoolExecutor(args.workers) as pool:
  for record,error in pool.map(check,jobs):
   if record:records.append(record)
   else:errors.append(error)
 records.sort(key=lambda r:(r['episode_id'],r['query_frame']))
 for split in ['train','validation','test']:
  with (out/(split+'.jsonl')).open('w') as f:
   for r in records:
    if r['split']==split:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 (out/'all.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 (out/'rgb_jobs.json').write_text(json.dumps(rgb_jobs,indent=2));(out/'episode_splits.json').write_text(json.dumps(split_by_ep,indent=2))
 report=dict(created=time.time(),labels_root=str(root),selection_sha256=hashlib.sha256((root/'selected_20_per_task.json').read_bytes()).hexdigest(),snapshot_records=len(records),split_samples=dict(collections.Counter(r['split'] for r in records)),split_episodes=dict(collections.Counter(split_by_ep[e['episode_id']] for e in rgb_jobs)),invalid=len(errors),errors=errors,rgb_episodes=len(rgb_jobs),total_prompt_image_tokens=sum(r['num_images']*64 for r in records),max_prompt_image_tokens=max(r['num_images']*64 for r in records),note='Snapshot of completed selected queries only. Missing/failed teacher labels excluded; no imputation. Splits fixed by whole episode, stratified by task. Exact model-context audit is performed before training; excess samples are quarantined, not truncated.')
 (out/'snapshot_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='errors'}),flush=True)
 for r in records:
  if r['query_frame']>0:
   example=dict(input=dict(task=r['task'],current_frame=r['query_frame'],rgb_history=dict(npy=r['rgb_npy'],frames=[0,r['query_frame']+1],interval='half-open; all frames',cameras=CAMS),per_camera_resolution=[256,256],per_camera_grid=[8,8]),output=json.loads(r['target']),expanded_score_shape=[r['query_frame']+1,3,8,8],teacher_meta=r['teacher_meta'])
   (out/'example.json').write_text(json.dumps(example,indent=2));break
if __name__=='__main__':main()
