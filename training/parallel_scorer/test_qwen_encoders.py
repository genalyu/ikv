"""Real tiny Qwen3.5 API/gradient test. Run in the Qwen environment."""
from types import SimpleNamespace
import torch
from torch import nn
from training.parallel_scorer.encoders import QwenEncoders
from training.parallel_scorer.model import ScorerConfig,ParallelScorer

def test_real_qwen_visual_task_features_and_gradients():
    from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    text=dict(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
              num_attention_heads=2,num_key_value_heads=1,head_dim=16,
              layer_types=["linear_attention","full_attention"],
              linear_num_key_heads=1,linear_num_value_heads=2,
              linear_key_head_dim=16,linear_value_head_dim=16,
              max_position_embeddings=128,partial_rotary_factor=1.,
              rope_parameters=dict(rope_type="default",rope_theta=10000.,
                                   partial_rotary_factor=1.,mrope_section=[2,3,3]))
    vision=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
                out_hidden_size=32,num_position_embeddings=256,patch_size=2,
                temporal_patch_size=1,spatial_merge_size=2)
    cfg=Qwen3_5Config(text_config=text,vision_config=vision,
                     image_token_id=60,video_token_id=61,vision_start_token_id=62,
                     vision_end_token_id=63)
    cfg._attn_implementation="eager"
    encoder=QwenEncoders.__new__(QwenEncoders);nn.Module.__init__(encoder)
    encoder.core=Qwen3_5ForConditionalGeneration(cfg).model
    encoder.train_vision=True;encoder.train_text=True;encoder.image_batch=2
    encoder._task_cache={}
    def image_processor(images,**kwargs):
        return dict(pixel_values=torch.randn(len(images)*256,12),
                    image_grid_thw=torch.tensor([[1,16,16]]*len(images)))
    def tokenizer(text,**kwargs):
        return dict(input_ids=torch.tensor([[3,4,5]]),
                    attention_mask=torch.ones(1,3,dtype=torch.long))
    encoder.processor=SimpleNamespace(image_processor=image_processor,tokenizer=tokenizer)
    encoder.train()
    features=encoder.images([None,None]);task=encoder.task("pick")
    assert features.shape==(128,32) and task.shape==(3,32)
    scorer=ParallelScorer(ScorerConfig(vision_dim=32,text_dim=32,width=16,layers=2,heads=2))
    z=scorer(features,task,torch.zeros(128),torch.rand(128,2),
             torch.arange(128)%2,torch.zeros(128,dtype=torch.long),0.)
    z.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.core.visual.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.core.language_model.parameters())
    encoder.train_text=False;encoder.core.zero_grad();encoder._task_cache.clear()
    with torch.inference_mode():encoder.task("frozen")
    cached=encoder.task("frozen")
    assert not cached.requires_grad
    # Cached inference-time features remain valid inputs to trainable projections.
    nn.Linear(32,1)(cached).sum().backward()
