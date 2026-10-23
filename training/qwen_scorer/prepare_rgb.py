"""Materialize clean RGB once as episode arrays, retaining every source frame."""
from pathlib import Path
import argparse,json,concurrent.futures as cf,time,os
import numpy as np,av,cv2
cv2.setNumThreads(1)

def extract(job):
 p=Path(job['path']);shape=(job['frames'],3,256,256,3)
 if p.exists():
  old=np.load(p,mmap_mode='r')
  if old.dtype==np.uint8 and old.shape[1:]==shape[1:] and old.shape[0]>=shape[0]:return
  del old
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial.npy')
 a=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=shape)
 for c,video in enumerate(job['videos']):
  seen=0
  with av.open(video) as container:
   stream=container.streams.video[0];stream.codec_context.thread_count=1
   for f,frame in enumerate(container.decode(stream)):
    if f>=job['frames']:break
    a[f,c]=cv2.resize(frame.to_ndarray(format='rgb24'),(256,256),interpolation=cv2.INTER_AREA);seen=f+1
  assert seen==job['frames'],(video,seen,job['frames'])
 a.flush();del a;tmp.replace(p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--workers',type=int,default=16);args=p.parse_args();r=Path(args.dataset);jobs=json.loads((r/'rgb_jobs.json').read_text());done=0;errors=[]
 with cf.ThreadPoolExecutor(args.workers) as pool:
  fs={pool.submit(extract,j):j for j in jobs}
  for f in cf.as_completed(fs):
   try:f.result();done+=1
   except Exception as e:errors.append(dict(episode=fs[f]['episode_id'],error=str(e)))
   (r/'rgb_progress.json').write_text(json.dumps(dict(completed=done,total=len(jobs),failed=len(errors),updated=time.time())))
 if errors:
  (r/'RGB_FAILED.json').write_text(json.dumps(errors,indent=2));raise SystemExit(1)
 (r/'RGB_READY.json').write_text(json.dumps(dict(episodes=done,updated=time.time())))
if __name__=='__main__':main()
