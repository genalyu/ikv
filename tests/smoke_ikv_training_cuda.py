"""Single-GPU real FSDP/activation-checkpoint smoke, separate from CPU pytest.

Run from the repository root: python tests/smoke_ikv_training_cuda.py
Uses a tiny randomly initialized MoT, not released weights or a task dataset.
"""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from n0_twam.models.mot import WanMoTTransformer3DModel
from n0_twam.distributed.fsdp import apply_ac, shard_model
from test_ikv_training import training_input, original_loss


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires one CUDA GPU")
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group('nccl', init_method='file://' + tmp + '/rendezvous',
                                rank=0, world_size=1)
        try:
            run()
        finally:
            dist.destroy_process_group()


def run():
    trainer = original_loss()
    trainer.gradient_accumulation_steps = 2
    for motion, ikv in [(False, False), (True, False), (False, True), (True, True)]:
        torch.manual_seed(123)
        # 32 is compatible with both WAN's 3-axis RoPE and FlexAttention CUDA.
        model = WanMoTTransformer3DModel(
            patch_size=(1, 2, 2), num_attention_heads=1, attention_head_dim=32,
            in_channels=2, out_channels=2, action_dim=3, text_dim=8, freq_dim=4,
            ffn_dim=32, num_layers=2, rope_max_seq_len=32, attn_mode='torch',
            use_local_tactile=False, use_rgb_motion_tokens=motion,
            rgb_motion_require_index=False).to(device='cuda', dtype=torch.bfloat16).train()
        apply_ac(model)
        reference = deepcopy(model)
        model = shard_model(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                data = training_input('cuda', motion)
                if ikv:
                    data['ikv_training'] = dict(capacity=12, retention=dict(top_k=2))
                model.set_requires_gradient_sync(micro == 1)
                output = model(data, train_mode=True)
                loss = trainer.compute_loss(data, output)['total_loss']
                loss.backward()
                expected = reference(data, train_mode=True)
                ref_loss = trainer.compute_loss(data, expected)['total_loss']
                ref_loss.backward()
                torch.testing.assert_close(loss, ref_loss, rtol=0.02, atol=0.01)
            ref_params = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                wanted = ref_params[name].grad
                assert (parameter.grad is None) == (wanted is None), name
                if wanted is None:
                    continue
                actual = parameter.grad.full_tensor().float()
                torch.testing.assert_close(actual, wanted.float(), rtol=0.05, atol=0.01,
                                           msg=lambda msg: name + '\n' + msg)
                error = (actual - wanted.float()).norm()
                assert error <= 0.02 * wanted.float().norm() + 1e-5, (name, error)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            optimizer.step()
            ref_optimizer.step()
            print(f'motion={motion} ikv={ikv} step={step}: '
                  f'loss={loss.detach().item():.6f}, gradients match reference', flush=True)
        del model, reference, optimizer, ref_optimizer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
