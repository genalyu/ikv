from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from common import prompt_text

class PrefixDataset(torch.utils.data.Dataset):
 def __init__(self,path):
  self.records=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
  if not self.records:raise ValueError('No validated samples: '+str(path))
 def __len__(self):return len(self.records)
 def __getitem__(self,index):return self.records[index]

class RGBCollator:
 def __init__(self,processor,max_length=262144):self.processor=processor;self.max_length=max_length
 def __call__(self,records):
  if len(records)!=1:raise ValueError('Variable full-history batches require micro_batch_size=1')
  r=records[0];array=np.load(r['rgb_npy'],mmap_mode='r');n=r['query_frame']+1
  assert array.ndim==5 and array.shape[1:]==(3,256,256,3) and array.dtype==np.uint8 and array.shape[0]>=n
  images=[Image.fromarray(array[f,c]) for f in range(n) for c in range(3)]
  prompt=prompt_text(self.processor,r)
  batch=self.processor(text=[prompt],images=images,return_tensors='pt',size={'shortest_edge':65536,'longest_edge':65536})
  grids=batch['image_grid_thw'];assert grids.shape==(n*3,3) and torch.equal(grids,torch.tensor([[1,16,16]]*(n*3)))
  image_id=self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
  assert int((batch['input_ids']==image_id).sum())==n*3*64
  answer=self.processor.tokenizer(r['target'],add_special_tokens=False,return_tensors='pt')['input_ids']
  eos=torch.tensor([[self.processor.tokenizer.eos_token_id]],dtype=answer.dtype);answer=torch.cat([answer,eos],dim=1)
  prompt_len=batch['input_ids'].shape[1];seq_len=prompt_len+answer.shape[1]
  if seq_len>self.max_length:raise ValueError(f'{r["id"]}: {seq_len} exceeds {self.max_length}; refusing to truncate RGB/history or output')
  batch['input_ids']=torch.cat([batch['input_ids'],answer],dim=1)
  batch['attention_mask']=torch.ones_like(batch['input_ids'])
  if 'mm_token_type_ids' in batch:batch['mm_token_type_ids']=torch.cat([batch['mm_token_type_ids'],torch.zeros_like(answer)],dim=1)
  labels=torch.full_like(batch['input_ids'],-100);labels[:,prompt_len:]=answer
  batch['labels']=labels;batch['sample_id']=r['id'];return batch
