"""Native-cache adapter for the same causal eviction used by offline plans."""
import torch
from n0_twam.models.global_kv_retention import RetentionConfig
from n0_twam.models.multimodal_kv_retention import MultimodalKVRetention
from .selection import evict


def native_ids(rows):
    """Even IDs are RGB frame*192+patch; odd IDs are action frame*12+step."""
    times=rows["world_time_id"].float()
    kind=rows["kind"].long()
    frame=times.floor().long()
    if (frame<0).any() or ((kind<0)|(kind>1)).any():
        raise ValueError("Only nonnegative RGB/action native addresses are allowed")
    ids=torch.empty_like(frame)
    video=kind==0
    grid=rows["grid_position"]
    rc=grid[video,:2]
    if (times[video]!=frame[video]).any() or not torch.equal(rc,rc.round()):
        raise ValueError("RGB address must lie on the native integer grid")
    if ((rc[:,0]<0)|(rc[:,0]>=8)|(rc[:,1]<0)|(rc[:,1]>=24)).any():
        raise ValueError("Memory requires a dense 8x24 native RGB grid")
    ids[video]=2*(frame[video]*192+rc[:,0].long()*24+rc[:,1].long())
    ordinal=((times[~video]-frame[~video])*13).round().long()-1
    if ((ordinal<0)|(ordinal>=12)).any():
        raise ValueError("Memory requires twelve native action positions")
    expected=frame[~video]+(ordinal+1).float()/13
    if not torch.allclose(times[~video],expected,atol=2e-5,rtol=0):
        raise ValueError("Unexpected action clock")
    ids[~video]=2*(frame[~video]*12+ordinal)+1
    if len(ids.unique())!=len(ids):
        raise ValueError("Duplicate native KV address")
    return ids


class FrozenMemoryRetention(MultimodalKVRetention):
    """Read-only planning; temporary denoising passes cannot mutate history."""
    def __init__(self, device, score_prefix, budgets=(3584,512)):
        config=RetentionConfig(version=2,video_capacity=budgets[0],action_capacity=budgets[1],
            tactile_capacity=0,query_weight=0.,action_query_weight=0.,tactile_query_weight=0.,
            visual_weight=0.,time_weight=0.,contact_weight=0.,persistence_weight=0.,
            class_recency_weight=0.,task_weight=0.)
        super().__init__(sum(budgets),device,config)
        self.score_prefix=score_prefix
        self.data["memory_id"]=torch.full((self.capacity,),-1,dtype=torch.long,device=self.device)

    def plan(self,mask,count,rows):
        ids=native_ids(rows)
        if count!=len(ids):raise ValueError("Incoming token count mismatch")
        live=mask.nonzero().flatten()
        kinds=self.data["kind"][live]
        current=self.data["memory_id"][live]
        if torch.isin(ids,current).any():
            raise ValueError("Prediction must be cleared before real grounding")
        incoming=tuple(int((rows["kind"]==k).sum()) for k in (0,1))
        scores=current.float()
        video=kinds==0
        if int(video.sum())+incoming[0]>self.budgets[0]:
            if incoming[0]==0:raise ValueError("Video budget was already exceeded")
            prefix=int(rows["world_time_id"][rows["kind"]==0].min())
            video_ids=current[video]//2
            if (video_ids>=prefix*192).any():
                raise ValueError("Cannot score target/future RGB")
            all_scores=torch.as_tensor(self.score_prefix(prefix),device=self.device).float().flatten()
            if len(all_scores)!=prefix*192 or not torch.isfinite(all_scores).all():
                raise ValueError("Frozen scorer returned a malformed prefix")
            scores[video]=all_scores[video_ids]
        kept=evict(current,kinds,scores,incoming,tuple(self.budgets[:2]))
        victims=live[~torch.isin(current,kept)]
        remaining=mask.clone();remaining[victims]=False
        slots=(~remaining).nonzero().flatten()[:count]
        if len(slots)!=count:raise ValueError("Native KV capacity mismatch")
        return slots,victims

    def commit(self,slots,rows,old_mask):
        ids=native_ids(rows)
        super().commit(slots,rows,old_mask)
        self.data["memory_id"][slots]=ids
