"""Run with the scorer environment (Transformers with Qwen3.5 support)."""
import torch
from training.qwen_scorer.grpo.policy import QwenPolicy


def tiny_policy():
    from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    from peft import get_peft_model,LoraConfig
    text=dict(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
              num_attention_heads=2,num_key_value_heads=1,head_dim=16,
              layer_types=["linear_attention","full_attention"],
              linear_num_key_heads=1,linear_num_value_heads=2,
              linear_key_head_dim=16,linear_value_head_dim=16,
              max_position_embeddings=128,partial_rotary_factor=1.,
              rope_parameters=dict(rope_type="default",rope_theta=10000.,
                                   partial_rotary_factor=1.,mrope_section=[2,3,3]))
    vision=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
                out_hidden_size=32,num_position_embeddings=16,patch_size=2,
                temporal_patch_size=1,spatial_merge_size=2)
    cfg=Qwen3_5Config(text_config=text,vision_config=vision,
                     image_token_id=60,video_token_id=61,vision_start_token_id=62,
                     vision_end_token_id=63)
    cfg._attn_implementation="eager"
    model=Qwen3_5ForConditionalGeneration(cfg).eval()
    lora=LoraConfig(task_type="CAUSAL_LM",r=2,lora_alpha=4,lora_dropout=0.,
                    target_modules=["q_proj","v_proj"])
    model=get_peft_model(model,lora,adapter_name="policy")
    model.add_adapter("reference",lora)
    policy=QwenPolicy.__new__(QwenPolicy)
    policy.model=model;policy.device=torch.device("cpu")
    return policy


def test_real_qwen_logprobs_match_full_logits_and_backprop():
    torch.manual_seed(21)
    policy=tiny_policy()
    prompt=dict(input_ids=torch.tensor([[3,4,5]]),attention_mask=torch.ones(1,3,dtype=torch.long))
    completion=torch.tensor([6,7,8,2])
    actual=policy.logprobs(prompt,completion)
    ids=torch.cat([prompt["input_ids"],completion[None]],1)
    logits=policy.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                        use_cache=False).logits[0,2:-1]
    expected=logits.log_softmax(-1).gather(1,completion[:,None]).squeeze(1)
    torch.testing.assert_close(actual,expected)
    (-actual.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0
               for n,p in policy.model.named_parameters() if ".policy." in n)
    assert all(p.grad is None for n,p in policy.model.named_parameters() if ".reference." in n)


def test_real_qwen_cached_sampling_matches_teacher_forced_probabilities():
    from transformers import GenerationConfig
    torch.manual_seed(23)
    policy=tiny_policy()
    prompt=dict(input_ids=torch.tensor([[3,4,5]]),attention_mask=torch.ones(1,3,dtype=torch.long))
    cfg=GenerationConfig(do_sample=True,temperature=1.,top_k=0,top_p=1.,
                         max_new_tokens=4,pad_token_id=0,use_cache=True,
                         return_dict_in_generate=True,output_scores=True)
    with torch.no_grad():
        out=policy.model.generate(**prompt,generation_config=cfg)
        completion=out.sequences[0,3:]
        sampled=torch.stack(out.scores)[:,0].log_softmax(-1).gather(1,completion[:,None]).squeeze(1)
        replay=policy.logprobs(prompt,completion)
    torch.testing.assert_close(sampled,replay,rtol=1e-4,atol=1e-5)


def test_real_qwen_image_cached_sampling_replay():
    from transformers import GenerationConfig
    torch.manual_seed(29)
    policy=tiny_policy()
    prompt=dict(input_ids=torch.tensor([[3,62,60,63,5]]),
                attention_mask=torch.ones(1,5,dtype=torch.long),
                pixel_values=torch.randn(4,12),
                image_grid_thw=torch.tensor([[1,2,2]]),
                mm_token_type_ids=torch.tensor([[0,0,1,0,0]]))
    cfg=GenerationConfig(do_sample=True,temperature=1.,top_k=0,top_p=1.,
                         max_new_tokens=4,pad_token_id=0,use_cache=True,
                         return_dict_in_generate=True,output_scores=True)
    with torch.no_grad():
        out=policy.model.generate(**prompt,generation_config=cfg)
        completion=out.sequences[0,5:]
        sampled=torch.stack(out.scores)[:,0].log_softmax(-1).gather(1,completion[:,None]).squeeze(1)
        replay=policy.logprobs(prompt,completion)
    torch.testing.assert_close(sampled,replay,rtol=1e-4,atol=1e-5)
