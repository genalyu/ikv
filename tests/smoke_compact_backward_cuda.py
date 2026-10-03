"""Compare bounded direct Flash backward with native autograd on packed attention."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from n0_twam.models.compact_ikv_attention import (
    packed_compact_ikv_attention, streamed_packed_ikv_attention,
)


def main():
    torch.manual_seed(42)
    n = 2048
    idx = torch.arange(n - 8, device="cuda")
    groups = [(idx[j:j+128], idx[max(0, j-256):j+128])
              for j in range(0, len(idx), 128)]
    q = torch.randn(1, n, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    upstream = torch.randn_like(q)
    reference = packed_compact_ikv_attention(q, k, v, groups)
    expected = torch.autograd.grad(reference, (q, k, v), upstream)
    errors = {}
    for budget in (512, 2048, 65536):
        actual = streamed_packed_ikv_attention(q, k, v, groups, max_keys=budget)
        gradients = torch.autograd.grad(actual, (q, k, v), upstream)
        torch.testing.assert_close(actual, reference, rtol=.002, atol=.002)
        row = []
        for got, want in zip(gradients, expected):
            assert torch.isfinite(got).all()
            assert not got[:, -8:].count_nonzero(), "padding must have zero gradient"
            relative = float((got.float()-want.float()).norm()/want.float().norm())
            # Native gather backward accumulates into BF16; bounded attention
            # intentionally accumulates duplicated K/V gradients into FP32.
            assert relative < .01, (budget, relative)
            row.append(relative)
        errors[budget] = row
    print(json.dumps(dict(status="passed", gradient_relative_l2=errors)), flush=True)


if __name__ == "__main__":
    main()
