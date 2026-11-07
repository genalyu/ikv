import torch
import pytest
from training.qwen_scorer.grpo.local_tactile import observed_local_mask


def test_final_action_uses_only_observed_current_tactile():
    mask=observed_local_mask((1,20,5,12,1),(1,2,5,4,4),3,'cpu')[0,0]
    keys=torch.arange(5).repeat_interleave(16).repeat(2)
    assert mask.shape==(60,160)
    assert torch.equal(mask[-1],keys==3)
    assert not mask[:,keys==4].any()
    assert torch.equal(mask[12],keys==1)
    # Altering hidden target local features cannot affect masked attention.
    q=torch.randn(1,1,60,8); k=torch.randn(1,1,160,8);v=torch.randn_like(k)
    a=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    k[:,:,keys==4]*=100;v[:,:,keys==4]+=100
    b=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    torch.testing.assert_close(a,b)


def test_future_local_frame_is_rejected():
    with pytest.raises(ValueError):
        observed_local_mask((1,20,5,12,1),(1,2,5,4,4),4,'cpu')
