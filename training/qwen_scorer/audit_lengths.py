"""Exact prompt token accounting at fixed 64 merged tokens per clean image."""
from pathlib import Path
import argparse,json,collections,time
import numpy as np
from transformers import AutoProcessor
from common import prompt_text

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',required=True);p.add_argument('--max-length',type=int,default=262144);a=p.parse_args();r=Path(a.dataset)
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 records=[json.loads(x) for x in (r/'all.jsonl').read_text().splitlines()];good=[];over=[]
 for rec in records:
  text=prompt_text(proc,rec);ids=proc.tokenizer(text,add_special_tokens=False)['input_ids'];img=proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
  assert ids.count(img)==rec['num_images']
  count=len(ids)+rec['num_images']*63+len(proc.tokenizer(rec['target'],add_special_tokens=False)['input_ids'])+1
  rec['sequence_length']=count;rec['target_tokens']=len(proc.tokenizer(rec['target'],add_special_tokens=False)['input_ids'])+1
  (good if count<=a.max_length else over).append(rec)
 for split in ['train','validation','test']:
  (r/(split+'.ready.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in good if x['split']==split))
 (r/'over_context.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in over))
 lengths=np.array([x['sequence_length'] for x in good]);train=[x for x in good if x['split']=='train']
 report=dict(verified=time.time(),max_length=a.max_length,samples=len(good),split_samples=dict(collections.Counter(x['split'] for x in good)),quarantined_over_context=len(over),length_p50_p90_p99_max=[int(x) for x in np.percentile(lengths,[50,90,99,100])],train_tokens=sum(x['sequence_length'] for x in train),train_squared_lengths=sum(x['sequence_length']**2 for x in train),training_samples=len(train),note='No input or target is truncated. All over-context records remain in over_context.jsonl for a separately configured longer-context run.')
 (r/'length_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
