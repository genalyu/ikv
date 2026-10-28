"""Opt-in history-only distillation. Legacy labels retain their provenance."""
import torch
from torch.nn import functional as F
VERSION="memory-history-objective-v2"

def history_mask(packet):
    mask=packet["times"]<float(packet["query_time"])
    return mask & packet["supervised"].bool() if "supervised" in packet else mask

def soft_topk(logits,k,temperature=.5):
    """Fixed-mass sigmoid selection with the implicit threshold derivative."""
    if logits.ndim!=1 or not 0<k<=len(logits) or temperature<=0:raise ValueError("Invalid soft top-k input")
    if k==len(logits):return torch.ones_like(logits)+logits*0
    with torch.no_grad():
        z=logits.detach();lo=z.min()-40*temperature;hi=z.max()+40*temperature
        for _ in range(32):
            threshold=(lo+hi)/2;mass=torch.sigmoid((z-threshold)/temperature).sum()
            lo,hi=(threshold,hi) if mass>k else (lo,threshold)
        threshold=(lo+hi)/2;p=torch.sigmoid((z-threshold)/temperature);slope=p*(1-p)
        weights=slope/slope.sum().clamp_min(torch.finfo(logits.dtype).tiny)
    correction=(weights*logits).sum();threshold=threshold+correction-correction.detach()
    return torch.sigmoid((logits-threshold)/temperature)

def memory_loss(logits,targets,eligible,k=1024,pairs=4096,rank_weight=.1,selection_weight=.1,temperature=.5):
    if logits.shape!=targets.shape or eligible.shape!=logits.shape:raise ValueError("Misaligned targets or eligibility")
    if k<1 or pairs<1 or rank_weight<0 or selection_weight<0:raise ValueError("Invalid loss configuration")
    z=logits[eligible.bool()].float();y=targets[eligible.bool()].float()/100
    if not len(z):raise ValueError("No historical supervision")
    if not torch.isfinite(z).all() or not torch.isfinite(y).all() or ((y<0)|(y>1)).any():raise ValueError("Invalid scores")
    levels,inverse,counts=torch.unique(y,return_inverse=True,return_counts=True)
    weights=.5/len(z)+.5/(len(levels)*counts[inverse].float())
    regression=(F.smooth_l1_loss(z.sigmoid(),y,reduction="none")*weights).sum()
    budget=min(k,len(z));boundary=torch.topk(y,budget).values[-1];positive=torch.where(y>=boundary)[0];half=(pairs+1)//2
    a=torch.cat([positive[torch.randint(len(positive),(half,),device=z.device)],torch.randint(len(z),(pairs-half,),device=z.device)])
    b=torch.randint(len(z),(pairs,),device=z.device);hard=torch.topk(z.detach(),min(2*budget,len(z))).indices
    b[:half]=hard[torch.randint(len(hard),(half,),device=z.device)];delta=y[a]-y[b];valid=delta!=0
    if valid.any():
        importance=delta[valid].abs();ranking=(F.softplus(-delta[valid].sign()*(z[a[valid]]-z[b[valid]]))*importance).sum()/importance.sum()
    else:ranking=z.sum()*0
    inclusion=soft_topk(z,budget,temperature);selection=torch.topk(y,budget).values.mean()-(inclusion*y).sum()/budget
    return regression+rank_weight*ranking+selection_weight*selection,dict(regression=regression.detach(),ranking=ranking.detach(),selection=selection.detach(),historical_tokens=len(z))

@torch.no_grad()
def selection_metrics(logits,targets,eligible,times=None,query_time=None,k=1024):
    mask=eligible.bool();z=logits[mask].float();y=targets[mask].float()
    if not len(z) or k<1:raise ValueError("Empty history or invalid budget")
    budget=min(k,len(z));chosen=torch.argsort(z,descending=True,stable=True)[:budget];ideal=torch.argsort(y,descending=True,stable=True)[:budget]
    boundary=y[ideal[-1]];strict=y>boundary;tied=y==boundary;slack=budget-int(strict.sum())
    overlap=(int(strict[chosen].sum())+min(int(tied[chosen].sum()),slack))/budget
    discount=1/torch.log2(torch.arange(budget,device=z.device).float()+2);idcg=(y[ideal]*discount).sum()
    im,sm,rm=y[ideal].mean(),y[chosen].mean(),y.mean();gain=im-rm
    row=dict(k=budget,candidates=len(z),mae=float((z.sigmoid()*100-y).abs().mean()),ndcg=float((y[chosen]*discount).sum()/idcg) if idcg>0 else 1.,tie_aware_overlap=overlap,selected_teacher_mean=float(sm),ideal_teacher_mean=float(im),regret=float(im-sm),gain_over_random=float((sm-rm)/gain) if gain>0 else None,strict_count=int(strict.sum()),strict_recall=float(strict[chosen].sum()/strict.sum()) if strict.any() else None)
    if times is not None and query_time is not None:
        early=times[mask].to(z.device)<float(query_time)/4;critical=strict & early
        row.update(early_critical_count=int(critical.sum()),early_recalled=int(critical[chosen].sum()),early_recall=float(critical[chosen].sum()/critical.sum()) if critical.any() else None)
    return row
