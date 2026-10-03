"""Metrics retain distinct global means/maxima and leave local losses untouched."""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from n0_twam.distributed.util import dist_mean_and_max


def _worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        local = torch.tensor([1., 8., 3.]) if rank == 0 else torch.tensor([5., 2., 3.])
        original = local.clone()
        mean, maximum = dist_mean_and_max(local)
        torch.testing.assert_close(mean, torch.tensor([3., 5., 3.]))
        torch.testing.assert_close(maximum, torch.tensor([5., 8., 3.]))
        assert torch.equal(local, original)
    finally:
        dist.destroy_process_group()


def test_distinct_mean_max_and_input_preservation(tmp_path):
    mp.spawn(_worker, args=("file://" + str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_single_rank_metrics_are_independent():
    local = torch.tensor([2., 3.])
    mean, maximum = dist_mean_and_max(local)
    mean.zero_()
    torch.testing.assert_close(maximum, local)
