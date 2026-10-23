"""CPU tests: real tiny N0 MoT/FM and exact GRPO gradient arithmetic."""
from copy import deepcopy
from types import SimpleNamespace
import torch
import pytest
from n0_twam.models.ikv_training import build_ikv_support_plan
from training.qwen_scorer.grpo.objective import advantages, grpo_loss, backward_group
from training.qwen_scorer.grpo.fm import FMReward, final_frame_loss_inputs, map_scores
from test_ikv_training import case, training_input, original_loss
from test_global_kv_retention import tiny_model


def test_query_support_is_causal_fixed_budget_and_changes_selection():
    *_, memory = case()
    memory["splits"] = [4,4,2,2,2,2,0]
    memory["config"].update(capacity=4, retention=dict(
        query_weight=0.,action_query_weight=0.,tactile_query_weight=0.))
    plain = build_ikv_support_plan(deepcopy(memory),torch.device("cpu"))
    memory["config"]["scorer_query"] = dict(target_phase=2,video_scores=[1.,9.,0.,0.],
                                           budgets=[1,2,1])
    plan = build_ikv_support_plan(memory,torch.device("cpu"))
    assert torch.equal(plan["noisy"][:2],plain["noisy"][:2])
    assert torch.equal(plan["clean"],plain["clean"])
    assert plan["noisy"][2,5] and not plan["noisy"][2,4]
    selected=plan["noisy"][2]
    assert (memory["layout"]["phase"][selected]<2).all()
    assert selected.sum()<=4
    memory["config"]["scorer_query"]["video_scores"]=[9.,1.,1000.,1000.]
    other=build_ikv_support_plan(memory,torch.device("cpu"))
    assert other["noisy"][2,4] and not other["noisy"][2,5]
    # Future score values do not affect eligible history.
    assert not other["noisy"][2,6:8].any()


def test_four_candidates_real_frozen_n0_and_original_fm():
    torch.manual_seed(19)
    model=tiny_model(False).to(torch.bfloat16)
    trainer=original_loss()
    reward=FMReward(trainer,model)
    data=training_input()
    data["ikv_training"]=dict(capacity=24, execution="masked", sample_capacity=False,
                             retention=dict(query_weight=0.,action_query_weight=0.,
                                            tactile_query_weight=0.))
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    rewards=[];selected=[]
    for candidate in range(4):
        scores=torch.zeros(12)
        scores[candidate]=100
        r,losses,mask=reward.evaluate(data,scores,[1,4,19])
        rewards.append(r);selected.append(tuple(mask))
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert len(set(selected))==4
    assert max(rewards)-min(rewards)>0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    for name,p in model.named_parameters():
        torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    repeated=reward.evaluate(data,torch.tensor([100.]+[0.]*11),[1,4,19])[0]
    assert repeated==rewards[0]
    # Preparation is immutable between candidates.
    assert "scorer_query" not in data["ikv_training"]


def test_final_frame_loss_ignores_historical_targets():
    data=training_input()
    model=tiny_model(False).to(torch.bfloat16)
    data["ikv_training"]=dict(capacity=24,execution="masked",
                             retention=dict(query_weight=0.,action_query_weight=0.,
                                            tactile_query_weight=0.))
    with torch.no_grad():
        predictions=model(data,train_mode=True)
    sliced,pred=final_frame_loss_inputs(data,predictions,(1,2,2))
    expected=original_loss().compute_loss(sliced,pred)["total_loss"]
    changed=deepcopy(data)
    changed["latent_dict"]["targets"][:,:,:-1]+=1000
    changed["action_dict"]["targets"][:,:,:-1]-=1000
    changed["action_dict"]["tactile_global_targets"][:,:,:,:-1]+=1000
    sliced,pred=final_frame_loss_inputs(changed,predictions,(1,2,2))
    actual=original_loss().compute_loss(sliced,pred)["total_loss"]
    torch.testing.assert_close(actual,expected)


def test_mapping_rejects_future_and_preserves_addresses():
    scores=torch.arange(2*3*8*8).reshape(2,3,8,8)
    result=map_scores(scores,[[1,2,7,7],[-1]*4])
    assert result.tolist()==[383.,0.]
    with pytest.raises(ValueError,match="future"):
        map_scores(scores,[[2,0,0,0]])
    with pytest.raises(ValueError):
        map_scores(scores,[[-1,0,0,0]])


def test_grpo_equal_rewards_zero_advantage_and_finite():
    assert advantages([1.,1.,1.,1.]).eq(0).all()
    with pytest.raises(ValueError):
        advantages([1.,float("nan")])


def test_multiple_frames_accumulation_matches_joint_objective():
    class Policy:
        device=torch.device("cpu");std_floor=1e-4;clip=.2;beta=.01
        def __init__(self):
            self.param=torch.nn.Parameter(torch.tensor([.2,-.1,.3,-.2]))
        def logprobs(self,prompt,completion):
            return self.param.log_softmax(0)[completion] + prompt*0
    p=Policy();other=Policy()
    completions=[torch.tensor([i]) for i in range(4)]
    old=[p.logprobs(0,c).detach() for c in completions]
    reference=[v-.1 for v in old]
    rewards=[[1.,2.,3.,4.],[2.,-1.,4.,1.]]
    for frame,r in enumerate(rewards):
        backward_group(p,frame,completions,old,reference,r,2)
    loss=0
    for frame,r in enumerate(rewards):
        adv=advantages(r)
        for i,c in enumerate(completions):
            loss=loss+grpo_loss(other.logprobs(frame,c),old[i],reference[i],adv[i])/8
    loss.backward()
    torch.testing.assert_close(p.param.grad,other.param.grad)
    assert p.param.grad.abs().sum()>0
    before=p.param.detach().clone()
    torch.optim.SGD([p.param],lr=.1).step()
    assert not torch.equal(before,p.param)


def test_fixed_time_preparation_reuses_real_trainer(monkeypatch):
    # Use the actual Trainer (not the heavyweight training constructor).
    from n0_twam.train import Trainer
    from n0_twam.configs.twam_base_cfg import twam_base_cfg
    cfg=deepcopy(twam_base_cfg)
    cfg.local_rank=0;cfg.enable_wandb=False;cfg.rank=0
    cfg.use_mot=True;cfg.use_ikv_training=True;cfg.ikv_train_capacity=24
    cfg.ikv_train_execution="masked";cfg.tactile_cfg_prob=0.
    cfg.kv_retention=dict(query_weight=0.,action_query_weight=0.,tactile_query_weight=0.)
    trainer=Trainer(cfg,inference_only=True)
    trainer.device=torch.device("cpu");trainer.dtype=torch.bfloat16
    trainer.patch_size=(1,2,2)
    model=tiny_model(False).to(torch.bfloat16)
    reward=FMReward(trainer,model)
    template=training_input()
    raw=dict(latents=template["latent_dict"]["latent"],
             actions=template["action_dict"]["latent"],
             actions_mask=template["action_dict"]["actions_mask"],
             text_emb=template["latent_dict"]["text_emb"],
             tactile_global_latent=template["action_dict"]["tactile_global_clean_latent"],
             tactile_sensor_ids=template["action_dict"]["tactile_sensor_ids"])
    first=reward.prepare(raw,123,400)
    second=reward.prepare(raw,123,400)
    torch.testing.assert_close(first["latent_dict"]["noisy_latents"],
                               second["latent_dict"]["noisy_latents"],rtol=0,atol=0)
    assert first["latent_dict"]["timesteps"].unique().numel()==1
    assert first["action_dict"]["timesteps"].unique().numel()==1
    assert not first["ikv_training"]["sample_capacity"]


def test_clean_target_cannot_leak_into_reward():
    model=tiny_model(False).to(torch.bfloat16)
    reward=FMReward(original_loss(),model)
    data=training_input()
    data["ikv_training"]=dict(capacity=24, execution="masked",
                             retention=dict(query_weight=0.,action_query_weight=0.,
                                            tactile_query_weight=0.))
    scores=torch.arange(12).float()
    before=reward.evaluate(data,scores,[1,4,19])[0]
    changed=deepcopy(data)
    changed["latent_dict"]["latent"][:,:,-1:]+=100
    changed["action_dict"]["latent"][:,:,-1:]-=100
    changed["action_dict"]["tactile_global_clean_latent"][:,:,:,-1:]+=100
    after=reward.evaluate(changed,scores,[1,4,19])[0]
    assert before==after


def test_pair_crop_preserves_all_modalities_and_rejects_unknown_layout():
    from training.qwen_scorer.grpo.prepare_pairs import crop_sample
    sample=dict(latents=torch.randn(2,5,4,4),actions=torch.randn(3,5,2,1),
                actions_mask=torch.ones(3,5,2,1),
                tactile_global_latent=torch.randn(2,48,5,4,4),
                text_emb=torch.randn(3,8))
    out=crop_sample(sample,2)
    assert out["latents"].shape[1]==out["actions"].shape[1]==3
    assert out["tactile_global_latent"].shape[2]==3
    assert sample["latents"].shape[1]==5
    with pytest.raises(ValueError,match="Unrecognized"):
        crop_sample(dict(sample,unknown=torch.zeros(5)),2)
