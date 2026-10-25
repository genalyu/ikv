"""Without-replacement selection policy; no autoregressive JSON.
Ordered Plackett-Luce paths are the policy actions. An unordered subset
probability is NOT approximated by multiplying independent softmax terms.
"""
import torch

def sample_order(logits,k,generator=None):
    if not torch.isfinite(logits).all() or k<0 or k>len(logits):raise ValueError("Invalid sampling domain")
    if k==0:return torch.empty(0,dtype=torch.long,device=logits.device)
    u=torch.rand(logits.shape,device=logits.device,generator=generator).clamp(1e-7,1-1e-7)
    return torch.topk(logits.float()-torch.log(-torch.log(u)),k,sorted=True).indices

def ordered_logprob(logits,order):
    if len(order)==0:return logits.sum()*0.
    if len(torch.unique(order))!=len(order) or (order<0).any() or (order>=len(logits)).any():
        raise ValueError("Not a valid without-replacement path")
    mask=torch.ones(len(logits),dtype=torch.bool,device=logits.device);mask[order]=False
    # All entries are shifted by the same constant. Reverse logcumsumexp avoids
    # unstable repeated subtraction from the total probability mass.
    z=logits.float()-logits.float().max()
    chosen=z[order]
    rest=torch.logsumexp(z[mask],0)
    remaining=torch.logcumsumexp(chosen.flip(0),0).flip(0)
    denominators=torch.logaddexp(remaining,rest)
    return (chosen-denominators).sum()

def sample_modalities(logits,modalities,budgets):
    result=[]
    for m in (0,1):
        pool=torch.where(modalities==m)[0]
        k=min(int(budgets[m]),len(pool))
        if k<0:raise ValueError("Negative budget")
        result.append(pool[sample_order(logits[pool],k)])
    return result

def path_logprob(logits,modalities,orders):
    value=logits.sum()*0.
    for m,order in enumerate(orders):
        pool=torch.where(modalities==m)[0]
        inverse=torch.full((len(logits),),-1,dtype=torch.long,device=logits.device)
        inverse[pool]=torch.arange(len(pool),device=logits.device)
        if len(order) and (inverse[order]<0).any():raise ValueError("Path modality mismatch")
        value=value+ordered_logprob(logits[pool],inverse[order])
    return value

def group_loss(logits,modalities,paths,old_logprob,rewards,clip=.2,std_floor=1e-4):
    r=torch.as_tensor(rewards,device=logits.device,dtype=torch.float32)
    if not torch.isfinite(r).all():raise ValueError("Nonfinite rewards")
    advantage=(r-r.mean())/r.std(unbiased=False).clamp_min(std_floor)
    new=torch.stack([path_logprob(logits,modalities,p) for p in paths])
    ratio=(new-old_logprob).clamp(-20,20).exp()
    loss=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()
    return loss
