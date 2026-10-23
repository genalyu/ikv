"""BF16 LoRA SFT with assistant-only loss, full RGB prefixes and DDP."""
from pathlib import Path
import argparse,json,time,math,contextlib,os
import torch,yaml
from datetime import timedelta
from torch.nn import functional as F
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import LoraConfig,get_peft_model,PeftModel
from accelerate import Accelerator,DistributedDataParallelKwargs
from accelerate.utils import set_seed,InitProcessGroupKwargs
from full_history_sampler import FullHistoryLengthSampler
from data import PrefixDataset,RGBCollator
from context_config import load_context_config
from vision_chunking import enable_chunked_vision
from vision_patch_linear import enable_linear_patch_embed
from activation_offload import ActivationOffload
from gradient_sync import average_gradients
from resume_position import restored_batch_index
from mlp_chunking import enable_chunked_mlp
from grouped_checkpoint import enable_grouped_checkpoints
from segmented_delta import enable_segmented_delta
from numerics import install_numerics_checks,install_kernel_checks,remove_numerics_checks,check_finite

TARGETS=r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))'

def attach_lora(model,rank=16,alpha=32):
 cfg=LoraConfig(task_type='CAUSAL_LM',r=rank,lora_alpha=alpha,lora_dropout=0.0,target_modules=TARGETS,bias='none')
 model=get_peft_model(model,cfg)
 assert any(p.requires_grad for p in model.parameters())
 assert not any(p.requires_grad for n,p in model.named_parameters() if '.visual.' in n)
 return model

def vision_module(model):
 return model.base_model.model.model.visual

def configure_vision(model,cfg,resume=None):
 visual=vision_module(model)
 if cfg.get('vision_train_dtype')=='float32':visual.float()
 if resume:
  visual.load_state_dict(torch.load(Path(resume)/'vision_state.pt',map_location='cpu',weights_only=True),strict=True)
 visual.requires_grad_(cfg['train_vision'])
 visual.train(cfg['train_vision'])
 assert all(p.requires_grad==cfg['train_vision'] for p in visual.parameters())

def assistant_loss(model,batch):
 # Never allocate [full history length, 248320] vocabulary logits.
 labels=batch['labels'];assert labels.shape[0]==1
 target_positions=torch.nonzero(labels[0]!=-100,as_tuple=False).flatten()
 assert target_positions.numel()>0 and target_positions.min()>0
 kwargs={k:v for k,v in batch.items() if k not in ['labels','sample_id']}
 output=model(**kwargs,use_cache=False,logits_to_keep=target_positions-1)
 logits=output.logits[0];truth=labels[0,target_positions]
 check_finite(logits,'assistant_logits')
 assert logits.shape[0]==truth.numel()
 # Chunk softmax computation while retaining exact token-weighted CE.
 loss=logits.new_zeros((),dtype=torch.float32)
 for start in range(0,truth.numel(),128):loss=loss+F.cross_entropy(logits[start:start+128].float(),truth[start:start+128],reduction='sum')
 return loss/truth.numel()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--max-updates',type=int);args=p.parse_args()
 if args.resume and not (Path(args.resume)/'COMPLETE.json').is_file():raise RuntimeError('Refusing incomplete checkpoint resume')
 cfg=yaml.safe_load(Path(args.config).read_text())
 final_marker=Path(cfg.get('final_dataset_marker', str(Path(cfg['model']).parent/'dataset/FINAL_DATASET_READY.json')))
 if not final_marker.exists():raise RuntimeError('Labeling is not finalized: refusing SFT on provisional labels')
 final=json.loads(final_marker.read_text())
 if final.get('missing_queries')!=0 or final.get('over_context')!=0:raise RuntimeError('Final dataset must include all planned queries and fit the configured context')
 quality_marker=Path(cfg['train_data']).parent/'QUALITY_DATASET_READY.json'
 if not quality_marker.exists():raise RuntimeError('Quality screening must complete before training')
 import hashlib
 quality=json.loads(quality_marker.read_text())
 for key in ['train_data','validation_data']:
  path=Path(cfg[key])
  assert hashlib.sha256(path.read_bytes()).hexdigest()==quality['file_sha256'][path.name], 'Quality dataset hash mismatch'
 out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
 acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=cfg['gradient_accumulation'],kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False),InitProcessGroupKwargs(timeout=timedelta(seconds=cfg.get('collective_timeout_seconds',3600)))])
 if not torch.cuda.is_available():raise RuntimeError('Production training requires CUDA')
 import flash_attn
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qmod
 if qmod.chunk_gated_delta_rule is None:raise RuntimeError('FLA gated-delta training kernel unavailable; refusing slow Python long-context fallback')
 set_seed(cfg['seed'])
 proc=AutoProcessor.from_pretrained(cfg['model'],local_files_only=True)
 proc.tokenizer.model_max_length=cfg['max_length']
 dataset=PrefixDataset(cfg['train_data'])
 model_cfg=load_context_config(cfg['model'],cfg['max_length'],cfg.get('rope_factor',1.0))
 model=Qwen3_5ForConditionalGeneration.from_pretrained(cfg['model'],config=model_cfg,dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True)
 if cfg['max_length']>model.config.text_config.max_position_embeddings:raise ValueError('Configured length exceeds model native context; no implicit RoPE changes')
 model.config.use_cache=False
 if args.resume:model=PeftModel.from_pretrained(model,args.resume,is_trainable=True)
 else:model=attach_lora(model,cfg['lora_rank'],cfg['lora_alpha'])
 configure_vision(model,cfg,args.resume)
 model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
 if cfg.get("linear_patch_embed",False):enable_linear_patch_embed(vision_module(model))
 enable_chunked_vision(vision_module(model),cfg.get('vision_images_per_chunk',6),deterministic_attention=cfg.get('vision_deterministic_attention',False))
 if cfg.get('checkpoint_group_size'):
  enable_grouped_checkpoints(model.base_model.model.model.language_model,cfg['checkpoint_group_size'],cfg.get('checkpoint_group_min_tokens',100000))
 if cfg.get('mlp_chunk_tokens'):enable_chunked_mlp(model,cfg['mlp_chunk_tokens'],gpu_recompute=cfg.get('local_recompute_on_gpu',False))
 if cfg.get('mlp_no_inner_remat_max_tokens'):
  from mlp_remat_candidate import enable_mlp_without_inner_remat
  count=enable_mlp_without_inner_remat(model,cfg['mlp_no_inner_remat_max_tokens'],cfg['mlp_chunk_tokens'])
  print(json.dumps(dict(event='mlp_inner_remat_disabled_short',rank=acc.process_index,max_tokens=cfg['mlp_no_inner_remat_max_tokens'],modules=count)),flush=True)
 import bounded_delta
 bounded_delta.GPU_SEGMENT_RECOMPUTE=cfg.get('local_recompute_on_gpu',False)
 if cfg.get('delta_segment_tokens'):
  enable_segmented_delta(model,cfg['delta_segment_tokens'],cfg.get('bounded_delta_backward',False))
 detailed_handles=[];detailed_kernel_originals=[]
 if cfg.get('numerics_checks',False):
  detailed_handles=install_numerics_checks(model);detailed_kernel_originals=install_kernel_checks(model)
 if cfg.get("autotune_memory_headroom",False):
  from autotune_memory import enable_autotune_memory_headroom
  enable_autotune_memory_headroom()
 if cfg.get("bucketed_autotune",False):
  from autotune_buckets import enable_bucketed_autotune
  enable_bucketed_autotune()
 generator=torch.Generator().manual_seed(cfg['seed'])
 Sampler=FullHistoryLengthSampler;sampler_extra={}
 if cfg.get('cost_balanced_sampler',False):
  from balanced_sampler_candidate import CostBalancedLengthSampler
  Sampler=CostBalancedLengthSampler;sampler_extra=dict(world_size=acc.num_processes,cost_model=cfg.get('cost_balance_cost_model'))
  print(json.dumps(dict(event='cost_balanced_sampler_enabled',rank=acc.process_index,effective_batch=acc.num_processes*cfg['gradient_accumulation'],per_update_sample_sets_unchanged=True)),flush=True)
 sampler=Sampler(batch_size=acc.num_processes*cfg['gradient_accumulation'],lengths=[x['sequence_length'] for x in dataset.records],generator=generator,mega_batch_mult=cfg.get('length_group_mega_batches'),**sampler_extra)
 rgb_collator=RGBCollator(proc,cfg['max_length'])
 loader=torch.utils.data.DataLoader(dataset,batch_size=1,sampler=sampler,collate_fn=(lambda records: records) if cfg.get('lazy_rgb_collation',False) else rgb_collator,num_workers=0,pin_memory=False)
 visual_params=[p for n,p in model.named_parameters() if p.requires_grad and '.visual.' in n]
 language_params=[p for n,p in model.named_parameters() if p.requires_grad and '.visual.' not in n]
 opt=torch.optim.AdamW([dict(params=language_params,lr=cfg['learning_rate']),dict(params=visual_params,lr=cfg['vision_learning_rate'])],betas=(.9,.95),weight_decay=.01)
 model,opt,loader=acc.prepare(model,opt,loader,device_placement=[True,True,False])
 from collections import Counter
 dtype_counts=Counter()
 for parameter in model.parameters():
  if parameter.requires_grad:dtype_counts[str(parameter.dtype)]+=parameter.numel()
 print(json.dumps(dict(event='trainable_parameter_dtypes',rank=acc.process_index,counts=dict(dtype_counts))),flush=True)
 updates_per_epoch=math.ceil(len(loader)/cfg['gradient_accumulation']);total_updates=updates_per_epoch*cfg['epochs'];warm=max(1,int(total_updates*.03))
 def scale(step):
  if step<warm:return max(1,step)/warm
  return .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total_updates-warm))))
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt.optimizer,scale)
 wb=None
 if acc.is_main_process and os.environ.get('WANDB_PROJECT'):
  import wandb
  wb=wandb.init(project=os.environ['WANDB_PROJECT'],name=os.environ.get('WANDB_NAME',out.name),dir=str(out),config=dict(cfg,world_size=acc.num_processes,planned_updates=total_updates),save_code=False)
 step=0;start_epoch=0;skip_batches=0
 if args.resume:
  state=torch.load(Path(args.resume)/'training_state.pt',map_location=acc.device,weights_only=False)
  opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);step=state['step'];start_epoch=state['epoch'];skip_batches=restored_batch_index(state,args.resume,acc.num_processes)
  restored_steps=sorted({int(v["step"]) for v in opt.optimizer.state.values()})
  restored_parameters=sum(v["exp_avg"].numel() for v in opt.optimizer.state.values())
  assert restored_steps==[step] and scheduler.last_epoch==step
  assert restored_parameters==sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(json.dumps(dict(event="training_state_restored",rank=acc.process_index,step=step,epoch=start_epoch,next_batch=skip_batches,world_size=acc.num_processes,optimizer_state_steps=restored_steps,optimizer_parameter_count=restored_parameters,scheduler_step=scheduler.last_epoch,resume_path=args.resume)),flush=True)
 def maybe_remove_detailed_checks():
  nonlocal detailed_handles,detailed_kernel_originals
  until=cfg.get("detailed_numerics_until_step")
  if until is not None and step>=until and (detailed_handles or detailed_kernel_originals):
   remove_numerics_checks(detailed_handles,detailed_kernel_originals)
   detailed_handles=[];detailed_kernel_originals=[]
   print(json.dumps(dict(event="detailed_numerics_disabled",rank=acc.process_index,after_step=step,remaining_guards=["assistant_logits","loss","segmented_delta_output_carry","segmented_delta_gradients","gradient_norm","parameters_first_updates"])),flush=True)
 maybe_remove_detailed_checks()
 if acc.is_main_process:
  (out/'run_config.json').write_text(json.dumps(dict(config=cfg,world_size=acc.num_processes,train_samples=len(dataset),planned_updates=total_updates,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gpu=torch.cuda.get_device_name()),indent=2))
 def checkpoint(epoch,next_batch,final=False):
  acc.wait_for_everyone()
  if acc.is_main_process:
   checkpoint_start=time.time()
   path=out/('final' if final else f'checkpoint-{step:06d}');path.mkdir(exist_ok=True)
   (path/'COMPLETE.json').unlink(missing_ok=True)
   acc.unwrap_model(model).save_pretrained(path,safe_serialization=True);proc.save_pretrained(path)
   acc.unwrap_model(model).config.save_pretrained(path)
   torch.save({k:v.detach().cpu() for k,v in vision_module(acc.unwrap_model(model)).state_dict().items()},path/'vision_state.pt')
   torch.save(dict(optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),step=step,epoch=epoch,next_batch=next_batch,world_size=acc.num_processes),path/'training_state.pt')
   (path/'COMPLETE.json').write_text(json.dumps(dict(step=step,epoch=epoch,next_batch=next_batch,world_size=acc.num_processes,saved=time.time())))
   print(json.dumps(dict(event='checkpoint_complete',step=step,path=str(path),seconds=time.time()-checkpoint_start)),flush=True)
  acc.wait_for_everyone()
 def evaluate():
  acc.wait_for_everyone()
  if acc.is_main_process:
   evaluation_start=time.time()
   val=PrefixDataset(cfg['validation_data']);indices=torch.linspace(0,len(val)-1,min(cfg['eval_samples'],len(val))).long().tolist();values=[];sample_ids=[]
   unwrapped=acc.unwrap_model(model);unwrapped.eval();collator=RGBCollator(proc,cfg['max_length'])
   with torch.inference_mode(),acc.autocast():
    for i in indices:
     b=collator([val[i]]);b={k:v.to(acc.device,dtype=torch.bfloat16) if isinstance(v,torch.Tensor) and v.is_floating_point() else v.to(acc.device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
     value=float(assistant_loss(unwrapped,b))
     if not math.isfinite(value):raise FloatingPointError('Nonfinite validation loss')
     values.append(value);sample_ids.append(b['sample_id'])
   validation_row=dict(event='validation_complete',step=step,loss=sum(values)/len(values),samples=len(values),sample_ids=sample_ids,manifest_sha256=hashlib.sha256(Path(cfg['validation_data']).read_bytes()).hexdigest(),seconds=time.time()-evaluation_start)
   with (out/'validation_metrics.jsonl').open('a') as f:f.write(json.dumps(validation_row)+'\n')
   print(json.dumps(validation_row),flush=True)
   if wb is not None:wb.log(dict(validation_loss=sum(values)/len(values),validation_samples=len(values)),step=step)
   unwrapped.train();vision_module(unwrapped).train(cfg['train_vision'])
  acc.wait_for_everyone()
 if cfg.get("eval_at_start",False) and not args.resume:evaluate()
 offloader=ActivationOffload(model,cfg.get("offload_gpu_budget_gib",0),cfg.get("offload_backward_gpu_budget_gib",0)) if cfg.get("activation_offload_mode")=="pageable_parameters_resident" else None
 if offloader:offloader.start_monitor()
 optimizer_moment_reserve=None
 if cfg.get('reserve_initial_optimizer_moments',False) and not args.resume:
  reserve_bytes=2*sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
  optimizer_moment_reserve=torch.empty(reserve_bytes,device=acc.device,dtype=torch.uint8)
  print(json.dumps(dict(event='optimizer_moment_reserve',rank=acc.process_index,gib=reserve_bytes/2**30)),flush=True)
 opt.zero_grad();start=time.time();last=start;tokens_since=0;loss_sum=torch.zeros((),device=acc.device);loss_samples=0
 for epoch in range(start_epoch,cfg['epochs']):
  generator.manual_seed(cfg['seed']+epoch);loader.set_epoch(epoch);model.train()
  vision_module(acc.unwrap_model(model)).train(cfg['train_vision'])
  active=acc.skip_first_batches(loader,skip_batches) if epoch==start_epoch and skip_batches else loader
  offset=skip_batches if epoch==start_epoch else 0
  sampler.set_epoch(epoch);active.set_epoch(epoch)
  for index,batch in enumerate(active,start=offset):
   if cfg.get('lazy_rgb_collation',False):batch=rgb_collator(batch)
   print(json.dumps(dict(event='microbatch_start',rank=acc.process_index,epoch=epoch,batch=index,sample=batch['sample_id'],sequence_length=batch['input_ids'].shape[1],time=time.time())),flush=True)
   input_tokens=batch['input_ids'].numel();tokens_since+=input_tokens
   batch={k:(v.to(acc.device,dtype=torch.bfloat16) if isinstance(v,torch.Tensor) and v.is_floating_point() else v.to(acc.device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
   offload_this_batch=cfg['activation_offload'] and input_tokens>cfg.get('activation_offload_min_tokens',0)
   ctx=(offloader.context() if offloader else torch.autograd.graph.save_on_cpu(pin_memory=True)) if offload_this_batch else contextlib.nullcontext()
   with acc.accumulate(model), (acc.no_sync(model) if cfg.get("manual_gradient_sync",False) else contextlib.nullcontext()):
    with ctx:
     if offloader:offloader.phase="forward"
     loss=assistant_loss(model,batch)
     if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss for '+batch['sample_id'])
     loss_sum.add_(loss.detach());loss_samples+=1
     print(json.dumps(dict(event='forward_complete',rank=acc.process_index,batch=index,loss=float(loss.detach()),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,activation_offload=offload_this_batch,time=time.time())),flush=True)
     if offloader:offloader.phase="backward";offloader.report()
     acc.backward(loss)
     print(json.dumps(dict(event='backward_complete',rank=acc.process_index,batch=index,time=time.time())),flush=True)
    if acc.sync_gradients:
     if cfg.get("manual_gradient_sync",False):average_gradients(model.parameters())
     norm=acc.clip_grad_norm_(model.parameters(),1.0)
     if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient norm before optimizer update')
     patch_parameter=vision_module(acc.unwrap_model(model)).patch_embed.proj.weight
     patch_before=patch_parameter.detach().clone() if step==0 and cfg['train_vision'] else None
     optimizer_moment_reserve=None
     opt.step();scheduler.step();opt.zero_grad();step+=1
     if step<=cfg.get('verify_first_updates',10):
      for name,parameter in model.named_parameters():
       if parameter.requires_grad and not torch.isfinite(parameter).all():raise FloatingPointError('Nonfinite parameter after update: '+name)
     patch_update_fraction=float((patch_parameter.detach()!=patch_before).float().mean()) if patch_before is not None else None
     del patch_before
     totals=acc.reduce(torch.stack([loss_sum,loss_sum.new_tensor(loss_samples)]),reduction='sum');avg_loss=float(totals[0]/totals[1]);count=float(acc.reduce(torch.tensor(float(tokens_since),device=acc.device),reduction='sum'));now=time.time()
     row=dict(step=step,epoch=epoch,loss=avg_loss,grad_norm=float(norm),lr=scheduler.get_last_lr()[0],update_seconds=now-last,input_tokens=count,tokens_per_second=count/max(.001,now-last),sequence_length=batch['input_ids'].shape[1],peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=now-start)
     if patch_update_fraction is not None:row['vision_patch_update_fraction']=patch_update_fraction
     if acc.is_main_process:
      with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
      print(json.dumps(row),flush=True)
      if wb is not None:wb.log(row,step=step)
      if step==1:(out/'GPU_PREFLIGHT_PASSED.json').write_text(json.dumps(row,indent=2))
     last=now;tokens_since=0;loss_sum.zero_();loss_samples=0
     if step==1 or step%cfg['save_every']==0 or step in cfg.get('save_after_updates',[]):checkpoint(epoch,index+1)
     if step in cfg.get('eval_after_updates',[]) or step%cfg['eval_every']==0:evaluate()
     maybe_remove_detailed_checks()
     if args.max_updates and step>=args.max_updates:
      checkpoint(epoch,index+1,final=True)
      if wb is not None:wb.finish()
      return
  skip_batches=0;checkpoint(epoch+1,0)
 checkpoint(cfg['epochs'],0,final=True)
 if wb is not None:wb.finish()
if __name__=='__main__':main()
