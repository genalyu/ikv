"""Resumable SFT over full-history packets or raw RGB; no prefix truncation."""
import argparse,json,random,os,signal,time,hashlib
from pathlib import Path
import torch
from .model import ScorerConfig,ParallelScorer,save_scorer
from .data import records,load_record,arguments,regression_loss,check_disjoint

def atomic_save(obj,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp);os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True);p.add_argument("--validation",required=True)
    p.add_argument("--output",required=True);p.add_argument("--preset",choices=["p067m","p26m","p59m","p1m","p5m","p8m","p13m","small","medium"],default="small")
    p.add_argument("--device",default="cuda:0");p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--encoder-lr",type=float,default=5e-6)
    p.add_argument("--accumulate",type=int,default=4);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--raw-model");p.add_argument("--revision",required=True)
    p.add_argument("--train-vision",action="store_true");p.add_argument("--train-text",action="store_true")
    p.add_argument("--query-chunk",type=int,default=65536)
    p.add_argument("--checkpoint-steps",type=int,default=50);p.add_argument("--resume",action="store_true")
    p.add_argument("--max-updates",type=int);p.add_argument("--cpu-threads",type=int,default=8)
    a=p.parse_args()
    if min(a.accumulate,a.checkpoint_steps,a.epochs,a.cpu_threads)<1:raise ValueError("Invalid positive setting")
    if (a.train_vision or a.train_text) and not a.raw_model:
        raise ValueError("Trainable encoders require raw RGB, not cached features")
    torch.set_num_threads(a.cpu_threads)
    train=records(a.manifest);val=records(a.validation);check_disjoint(train,val)
    out=Path(a.output)
    config=vars(a).copy()
    config["manifest_sha256"]=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    config["validation_sha256"]=hashlib.sha256(Path(a.validation).read_bytes()).hexdigest()
    if a.resume:
        old=json.loads((out/"config.json").read_text())
        exceptions={"resume","max_updates","device","cpu_threads"}
        if any(old.get(k)!=v for k,v in config.items() if k not in exceptions):
            raise ValueError("Resume configuration or manifests changed")
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/"config.json").write_text(json.dumps(config,indent=2))
    # Refuse concurrent writers to a run even when resuming.
    import fcntl
    lock=(out/"run.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.manual_seed(a.seed);random.seed(a.seed)
    encoder=None
    if a.raw_model:
        from .encoders import QwenEncoders
        encoder=QwenEncoders(a.raw_model,a.revision,a.device,a.train_vision,a.train_text)
    def packet(r):
        return encoder.record(r) if encoder else load_record(r,a.revision)
    first=packet(train[0]);vd=first["features"].shape[-1];td=first["text"].shape[-1];del first
    c=ScorerConfig.preset(a.preset,vision_dim=vd,text_dim=td,query_chunk=a.query_chunk,
                         flash_only=a.device.startswith("cuda"))
    model=ParallelScorer(c).to(a.device)
    groups=[dict(params=list(model.parameters()),lr=a.lr)]
    if encoder:
        params=[v for v in encoder.parameters() if v.requires_grad]
        if params:groups.append(dict(params=params,lr=a.encoder_lr))
    optimizer=torch.optim.AdamW(groups)
    updates=0;first_epoch=0;cursor=0;best_mae=float("inf");training_seconds=0.;validation_seconds=0.
    if a.resume:
        state=torch.load(out/"last-state.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        if encoder:encoder.restore(state["encoders"])
        updates=state["updates"];first_epoch=state["epoch"];cursor=state["cursor"]
        best_mae=state["best_mae"];training_seconds=state.get("training_seconds",0.);validation_seconds=state.get("validation_seconds",0.);torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if a.device.startswith("cuda"):torch.cuda.set_rng_state_all(state["cuda_rng"])
    stop=[False]
    def stopping(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    def context():
        return torch.autocast("cuda",dtype=torch.bfloat16,enabled=a.device.startswith("cuda"))
    def log(row):
        print(json.dumps(row),flush=True)
        with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
    def save(epoch,next_cursor):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            encoders=encoder.export_trainable() if encoder else {},updates=updates,
            epoch=epoch,cursor=next_cursor,best_mae=best_mae,training_seconds=training_seconds,validation_seconds=validation_seconds,
            torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else [])
        atomic_save(state,out/"last-state.pt")
        revision=a.revision+(f":update-{updates}" if a.train_vision or a.train_text else "")
        tmp=out/"last-scorer.tmp";save_scorer(model,tmp,encoder_revision=revision)
        os.replace(tmp,out/"last-scorer.pt")
        if encoder:atomic_save(encoder.export_trainable(),out/"last-encoders.pt")
        (out/"status.json").write_text(json.dumps(dict(updates=updates,epoch=epoch,
            next_cursor=next_cursor,encoder_revision=revision,time=time.time(),training_seconds=training_seconds,validation_seconds=validation_seconds,
            training_gpu_hours=training_seconds/3600 if a.device.startswith("cuda") else 0.),indent=2))
    log(dict(event="start",counts=model.counts(),train_queries=len(train),
        validation_queries=len(val),resume=a.resume,update=updates))
    for epoch in range(first_epoch,a.epochs):
        order=list(range(len(train)));random.Random(a.seed+epoch).shuffle(order)
        model.train()
        if encoder:encoder.train()
        for start in range(cursor if epoch==first_epoch else 0,len(order),a.accumulate):
            begin=time.perf_counter();batch=order[start:start+a.accumulate]
            optimizer.zero_grad(set_to_none=True);loss_sum=0.;tokens=0;data_seconds=0.
            if a.device.startswith("cuda"):torch.cuda.reset_peak_memory_stats()
            for j in batch:
                with context():
                    data_start=time.perf_counter();q=packet(train[j]);data_seconds+=time.perf_counter()-data_start
                    pred=model(**arguments(q,a.device));tokens+=len(pred)
                    mask=q.get("supervised")
                    loss=regression_loss(pred,q["targets"].to(a.device),
                                         None if mask is None else mask.to(a.device))
                (loss/len(batch)).backward();loss_sum+=float(loss.detach());del q,pred,loss
            grad=torch.nn.utils.clip_grad_norm_([v for g in groups for v in g["params"]],1.)
            if not torch.isfinite(grad):raise FloatingPointError("Nonfinite gradient")
            optimizer.step();updates+=1
            if a.device.startswith("cuda"):torch.cuda.synchronize()
            elapsed=time.perf_counter()-begin;training_seconds+=elapsed
            log(dict(event="update",update=updates,epoch=epoch,loss=loss_sum/len(batch),
                     grad_norm=float(grad),tokens=tokens,seconds=elapsed,data_seconds=data_seconds,
                     training_seconds=training_seconds,training_gpu_hours=training_seconds/3600 if a.device.startswith("cuda") else 0.,
                     peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith("cuda") else None))
            if updates%a.checkpoint_steps==0 or stop[0] or (a.max_updates and updates>=a.max_updates):
                save(epoch,start+len(batch))
            if stop[0] or (a.max_updates and updates>=a.max_updates):
                log(dict(event="stopped_at_update_boundary",update=updates));return
        model.eval()
        if encoder:encoder.eval()
        errors=[];validation_start=time.perf_counter()
        with torch.inference_mode():
            for r in val:
                with context():
                    q=packet(r);pred=model(**arguments(q,a.device)).float().sigmoid()*100
                mask=q.get("supervised",torch.ones(len(pred),dtype=torch.bool)).to(a.device).bool()
                if not mask.any():raise ValueError("Empty validation supervision")
                errors.append(float((pred[mask]-q["targets"].to(a.device)[mask]).abs().mean()))
        validation_elapsed=time.perf_counter()-validation_start;validation_seconds+=validation_elapsed
        mae=sum(errors)/len(errors);improved=mae<best_mae;best_mae=min(best_mae,mae)
        log(dict(event="validation",epoch=epoch,update=updates,queries=len(errors),mae=mae,seconds=validation_elapsed))
        save(epoch+1,0)
        # Retain epoch results; only fixed-size last-state is used for resuming.
        import shutil
        shutil.copyfile(out/"last-scorer.pt",out/f"scorer-{epoch}.pt")
        if encoder:shutil.copyfile(out/"last-encoders.pt",out/f"encoders-{epoch}.pt")
        if improved:
            shutil.copyfile(out/"last-scorer.pt",out/"best-scorer.pt")
            if encoder:shutil.copyfile(out/"last-encoders.pt",out/"best-encoders.pt")
        if stop[0]:return
    (out/"COMPLETE.json").write_text(json.dumps(dict(updates=updates,best_validation_mae=best_mae,time=time.time(),training_seconds=training_seconds,validation_seconds=validation_seconds)))
if __name__=="__main__":main()
