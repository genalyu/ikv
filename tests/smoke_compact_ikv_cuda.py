"""Small full-model CUDA A/B for sparse Flex vs exact packed retained KV."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from n0_twam.models.mot import WanMoTTransformer3DModel
from test_ikv_training import training_input, original_loss

def main():
    torch.manual_seed(123)
    model = WanMoTTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=1, attention_head_dim=32,
        in_channels=2, out_channels=2, action_dim=3, text_dim=8, freq_dim=4,
        ffn_dim=32, num_layers=2, rope_max_seq_len=32, attn_mode="torch",
        use_local_tactile=False, use_rgb_motion_tokens=False
    ).to(device="cuda", dtype=torch.bfloat16).train()
    packed = deepcopy(model)
    model.mot.gradient_checkpointing = True
    packed.mot.gradient_checkpointing = True
    data = training_input("cuda", False)
    data["ikv_training"] = dict(
        capacity=12, retention=dict(top_k=2, query_weight=0.0, action_query_weight=0.0, tactile_query_weight=0.0),
        execution="masked", sample_capacity=False, block_size=64)
    target = original_loss()
    output = model(data, train_mode=True)
    loss = target.compute_loss(data, output)["total_loss"]
    loss.backward()
    packed_data = deepcopy(data)
    packed_data["ikv_training"]["compact_attention"] = True
    packed_data["ikv_training"]["compact_max_packed_keys"] = 512
    candidate = packed(packed_data, train_mode=True)
    candidate_loss = target.compute_loss(packed_data, candidate)["total_loss"]
    candidate_loss.backward()
    print("losses", float(loss), float(candidate_loss), flush=True)
    torch.testing.assert_close(candidate_loss, loss, rtol=0.02, atol=0.01)
    total_error = 0.0
    total_reference = 0.0
    for (name, a), (_, b) in zip(model.named_parameters(), packed.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
            continue
        assert torch.isfinite(b.grad).all(), name
        total_error += float((a.grad.float()-b.grad.float()).square().sum())
        total_reference += float(a.grad.float().square().sum())
    relative = (total_error / total_reference)**0.5
    print("gradient_relative_l2", relative, flush=True)
    assert relative < 0.03

if __name__ == "__main__":
    main()
