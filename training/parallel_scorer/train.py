"""Supervised variable-length scores: cached features OR end-to-end raw RGB.
No changes to existing SFT jobs. A fresh output directory is mandatory.
"""
import argparse,json,random
from pathlib import Path
import torch
from .model import ScorerConfig,ParallelScorer,save_scorer
from .data import records,load_packet,arguments,regression_loss,check_disjoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True);p.add_argument("--validation",required=True)
    p.add_argument("--output",required=True);p.add_argument("--preset",choices=["small","medium"],default="small")
    p.add_argument("--device",default="cuda:0");p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--encoder-lr",type=float,default=5e-6)
    p.add_argument("--accumulate",type=int,default=4);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--raw-model");p.add_argument("--revision",required=True)
    p.add_argument("--train-vision",action="store_true");p.add_argument("--train-text",action="store_true")
    p.add_argument("--query-chunk",type=int,default=2048)
    a=p.parse_args()
    if a.accumulate<1:raise ValueError("accumulate must be positive")
    if (a.train_vision or a.train_text) and not a.raw_model:
        raise ValueError("Trainable encoders require raw RGB, not cached features")
    train=records(a.manifest);val=records(a.validation);check_disjoint(train,val)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/"config.json").write_text(json.dumps(vars(a),indent=2))
    torch.manual_seed(a.seed);rng=random.Random(a.seed)
    encoder=None
    if a.raw_model:
        from .encoders import QwenEncoders
        encoder=QwenEncoders(a.raw_model,a.revision,a.device,a.train_vision,a.train_text)
    def packet(r):
        return encoder.record(r) if encoder else load_packet(r["feature_path"],a.revision)
    first=packet(train[0]);vd=first["features"].shape[-1];td=first["text"].shape[-1];del first
    c=ScorerConfig.preset(a.preset,vision_dim=vd,text_dim=td,query_chunk=a.query_chunk,
                         flash_only=a.device.startswith("cuda"))
    model=ParallelScorer(c).to(a.device)
    groups=[dict(params=model.parameters(),lr=a.lr)]
    if encoder:
        params=[p for p in encoder.parameters() if p.requires_grad]
        if params:groups.append(dict(params=params,lr=a.encoder_lr))
    optimizer=torch.optim.AdamW(groups)
    def context():
        return torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=a.device.startswith("cuda"))
    updates=0
    for epoch in range(a.epochs):
        rng.shuffle(train);model.train()
        if encoder:encoder.train()
        for start in range(0,len(train),a.accumulate):
            batch=train[start:start+a.accumulate];optimizer.zero_grad(set_to_none=True);loss_sum=0.
            for r in batch:
                with context():
                    q=packet(r);pred=model(**arguments(q,a.device))
                    loss=regression_loss(pred,q["targets"].to(a.device),
                         q.get("supervised",None).to(a.device) if q.get("supervised") is not None else None)
                (loss/len(batch)).backward();loss_sum+=float(loss.detach());del q,pred,loss
            grad=torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]],1.)
            if not torch.isfinite(grad):raise FloatingPointError("Nonfinite gradient")
            optimizer.step();updates+=1
            with (out/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(update=updates,epoch=epoch,loss=loss_sum/len(batch)))+"\n")
        model.eval()
        if encoder:encoder.eval()
        errors=[]
        with torch.inference_mode():
            for r in val:
                with context():
                    q=packet(r);pred=model(**arguments(q,a.device)).float().sigmoid()*100
                mask=q.get("supervised",torch.ones(len(pred),dtype=torch.bool)).to(a.device)
                errors.append(float((pred[mask]-q["targets"].to(a.device)[mask]).abs().mean()))
        with (out/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(dict(epoch=epoch,validation_mae=sum(errors)/len(errors)))+"\n")
        # New visual/text weights imply a new cache revision.
        revision=a.revision+(f":epoch-{epoch}" if a.train_vision or a.train_text else "")
        save_scorer(model,out/f"scorer-{epoch}.pt",encoder_revision=revision)
        if encoder:
            torch.save(encoder.export_trainable(),out/f"encoders-{epoch}.pt")
        torch.save(dict(optimizer=optimizer.state_dict(),epoch=epoch,updates=updates),
                   out/f"optimizer-{epoch}.pt")
if __name__=="__main__":main()
