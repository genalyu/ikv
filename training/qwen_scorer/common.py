"""Shared training/inference contract: clean RGB, complete history, fixed 8x8 grid."""
from pathlib import Path
import json
import numpy as np
CAMS=['observation.images.top','observation.images.wrist_l','observation.images.wrist_r']
GRID=8
SYSTEM='''You score visual evidence for the robot next action. Given the task and all observed RGB images through query frame i, reassess EVERY historical frame/camera/patch. Preserve evidence needed for unresolved historical dependencies and accurate current control; lower redundant or superseded evidence. Do not rank by age alone or infer success from the instruction. Use only provided observations, never future frames. Each clean RGB camera image is 256x256 with an 8-row by 8-column grid; row increases downward, column rightward, each cell is 32x32.\nReturn ONLY JSON with EXACT keys default_score and blocks. default_score is an integer 0..100. Each block has EXACT keys frames:[start,end_exclusive],camera:0/1/2,rows:[start,end_exclusive],cols:[start,end_exclusive],score:integer 0..100. Frame intervals use original RGB frame numbers, not query numbers. Row and column bounds are 0..8 exclusive. Initialize every candidate to default_score, then apply blocks in order, later blocks overwrite. Serialize equal scores compactly but do not sacrifice meaningful distinctions. This must define a considered score for every historical patch. No reasoning or prose.'''

def user_text(record):
 q=record['query_frame'];eid=record['episode_id']
 return json.dumps(dict(task=record['task'],current_frame=q,current_time=q/record['fps'],fps=record['fps'],image_order='frame-major then camera-major; image number = 3*frame+camera. All original frames 0..current_frame are supplied, no temporal subsampling.',cameras=CAMS,grid=dict(rows=8,columns=8,cell_pixels=32),visual_candidates=dict(total=(q+1)*192,domain=f'frame 0..{q}, camera 0..2, row 0..7, column 0..7',id_format=f'{eid}:f{{frame}}:c{{camera}}:p{{row*8+column}}')),ensure_ascii=False,separators=(',',':'))

def messages(record,images=None):
 count=(record['query_frame']+1)*3
 content=[dict(type='text',text=user_text(record))]
 if images is None:content += [dict(type='image') for _ in range(count)]
 else:
  assert len(images)==count
  content += [dict(type='image',image=im) for im in images]
 return [dict(role='system',content=SYSTEM),dict(role='user',content=content)]

def prompt_text(processor,record):
 return processor.apply_chat_template(messages(record),tokenize=False,add_generation_prompt=True,enable_thinking=False)

def decode_scores(obj,n):
 assert isinstance(obj,dict) and set(obj)=={'default_score','blocks'}
 assert type(obj['default_score']) is int and 0<=obj['default_score']<=100
 assert isinstance(obj['blocks'],list)
 a=np.full((n,3,8,8),obj['default_score'],dtype=np.uint8)
 for b in obj['blocks']:
  assert set(b)=={'frames','camera','rows','cols','score'}
  assert all(isinstance(b[k],list) and len(b[k])==2 for k in ['frames','rows','cols'])
  f0,f1=b['frames'];r0,r1=b['rows'];c0,c1=b['cols'];cam=b['camera'];score=b['score']
  assert all(type(v) is int for v in [f0,f1,r0,r1,c0,c1,cam,score])
  assert 0<=f0<f1<=n and 0<=r0<r1<=8 and 0<=c0<c1<=8 and 0<=cam<3 and 0<=score<=100
  a[f0:f1,cam,r0:r1,c0:c1]=score
 return a

def canonical_entries(scores,eid):
 for f,c,r,k in np.ndindex(scores.shape):yield dict(id=f'{eid}:f{f}:c{c}:p{r*8+k}',score=int(scores[f,c,r,k]))
