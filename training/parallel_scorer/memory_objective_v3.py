"""Experimental Top-k-stratified supervision; no age or evidence-group priors."""
import torch
from torch.nn import functional as F
from .memory_objective import history_mask,selection_metrics,soft_topk
VERSION="memory-history-objective-v3"

def memory_loss(logits,targets,eligible,k=1024,pairs=4096,rank_weight=.1,selection_weight=.1,temperature=.03,point_balance="strata",strict_positive=True):
    if logits.shape!=targets.shape or eligible.shape!=logits.shape:raise ValueError("Misaligned loss input")
    if k<1 or pairs<1 or point_balance not in ["strata","levels"]:raise ValueError("Invalid loss configuration")
    z=logits[eligible.bool()].float();y=targets[eligible.bool()].float()/100
    if not len(z):raise ValueError("No historical supervision")
    if not torch.isfinite(z).all() or not torch.isfinite(y).all() or ((y<0)|(y>1)).any():raise ValueError("Invalid scores")
    budget=min(k,len(z));boundary=torch.topk(y,budget).values[-1];positive=y>boundary;ties=y==boundary;negative=y<boundary
    if point_balance=="levels":
        levels,inv,counts=torch.unique(y,return_inverse=True,return_counts=True);weights=.5/len(z)+.5/(len(levels)*counts[inv].float())
    else:
        weights=torch.zeros_like(y);mass=0.
        for mask,weight in [(positive,.4),(ties,.2),(negative,.4)]:
            if mask.any():weights[mask]=weight/mask.sum();mass+=weight
        weights/=mass
    regression=(F.smooth_l1_loss(z.sigmoid(),y,reduction="none")*weights).sum()
    pool=torch.where(positive if strict_positive and positive.any() else y>=boundary)[0];half=(pairs+1)//2
    a=torch.cat([pool[torch.randint(len(pool),(half,),device=z.device)],torch.randint(len(z),(pairs-half,),device=z.device)])
    b=torch.randint(len(z),(pairs,),device=z.device);hard=torch.topk(z.detach(),min(2*budget,len(z))).indices;b[:half]=hard[torch.randint(len(hard),(half,),device=z.device)]
    delta=y[a]-y[b];valid=delta!=0
    if valid.any():
        importance=delta[valid].abs();ranking=(F.softplus(-delta[valid].sign()*(z[a[valid]]-z[b[valid]]))*importance).sum()/importance.sum()
    else:ranking=z.sum()*0
    inclusion=soft_topk(z,budget,temperature);selection=torch.topk(y,budget).values.mean()-(inclusion*y).sum()/budget
    return regression+rank_weight*ranking+selection_weight*selection,dict(regression=regression.detach(),ranking=ranking.detach(),selection=selection.detach(),historical_tokens=len(z))
