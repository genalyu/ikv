from pathlib import Path
import argparse,json,gzip
import numpy as np,torch
from PIL import Image
from transformers import AutoProcessor,AutoConfig,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from common import prompt_text,decode_scores,canonical_entries

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter',required=True);p.add_argument('--record',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 record=json.loads(Path(a.record).read_text());proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 array=np.load(record['rgb_npy'],mmap_mode='r');n=record['query_frame']+1
 images=[Image.fromarray(array[f,c]) for f in range(n) for c in range(3)]
 batch=proc(text=[prompt_text(proc,record)],images=images,return_tensors='pt',size={'shortest_edge':65536,'longest_edge':65536})
 model_cfg=AutoConfig.from_pretrained(a.adapter,local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,config=model_cfg,dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True).cuda()
 model=PeftModel.from_pretrained(model,a.adapter)
 model.base_model.model.model.visual.load_state_dict(torch.load(Path(a.adapter)/'vision_state.pt',map_location='cpu',weights_only=True),strict=True)
 model.eval()
 batch={k:v.to(model.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(model.device) for k,v in batch.items()}
 if batch['input_ids'].shape[1]+4096>model.config.text_config.max_position_embeddings:raise ValueError('Context too long; no silent history truncation')
 with torch.inference_mode():ids=model.generate(**batch,max_new_tokens=4096,do_sample=False,use_cache=True)
 text=proc.tokenizer.decode(ids[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip();obj=json.loads(text);scores=decode_scores(obj,n)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'assignment.json').write_text(json.dumps(obj,indent=2));np.savez_compressed(out/'scores.npz',scores=scores)
 with gzip.open(out/'visual_scores.json.gz','wt') as f:json.dump(dict(visual_scores=list(canonical_entries(scores,record['episode_id']))),f)
 print(json.dumps(dict(shape=list(scores.shape),candidates=scores.size,output=str(out))))
if __name__=='__main__':main()
