import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from n0_twam.task_pipeline.config import accumulation, training_config
from n0_twam.task_pipeline.data import (
    causal_indices,
    normalize_grip,
    quality_exclusion_reason,
    validate_pose,
)
from n0_twam.task_pipeline.neosim import quaternion_to_rot6d
from n0_twam.task_pipeline.features import build_payloads


def task(tmp_path):
    t = json.loads(
        (
            Path(__file__).parents[1] / "examples/tasks/ur5e_phone_weight.json"
        ).read_text()
    )
    t["source"] = str(tmp_path / "raw")
    t["runtime"] = {
        "work_root": str(tmp_path),
        "base_checkpoint": str(tmp_path / "base"),
        "dino_model": str(tmp_path / "dino"),
    }
    return t


@pytest.mark.parametrize("world,acc", [(1, 32), (2, 16), (4, 8), (8, 4)])
def test_official_effective_batch_and_four_modes(tmp_path, world, acc):
    assert accumulation(world) == acc
    t = task(tmp_path)
    for mode, bits in [
        ("baseline", (False, False)),
        ("motion", (True, False)),
        ("ikv", (False, True)),
        ("motion_ikv", (True, True)),
    ]:
        c = training_config(t, mode, world, require_ready=False)
        assert (c.use_rgb_motion_tokens, c.use_ikv_training) == bits
        assert c.batch_size * c.gradient_accumulation_steps * world == 32
        assert c.num_steps == 2000 and c.learning_rate == 1e-4 and c.warmup_steps == 20
        assert (c.beta1, c.beta2, c.weight_decay) == (0.9, 0.95, 0.1)
        assert c.kv_cache_policy == ("global" if bits[1] else "fifo")
        if bits[1]:
            assert c.ikv_train_execution == "masked"
            assert c.ikv_train_sample_capacity is True
            assert c.ikv_train_min_capacity == 2048
            assert all(c.kv_retention[name] == 0 for name in (
                "query_weight", "action_query_weight", "tactile_query_weight"))
        assert c.max_latent_frames == c.max_tactile_frames == 0
        assert c.used_action_channel_ids == list(range(10))


def test_accumulated_microbatches_sync_sharded_gradients_before_update():
    from test_rgb_motion_model_integration import _load_trainer_class

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.sync_calls = []

        def set_requires_gradient_sync(self, enabled):
            self.sync_calls.append(enabled)

        def forward(self, batch, train_mode):
            assert train_mode
            return self.weight * batch

    trainer = _load_trainer_class().__new__(_load_trainer_class())
    trainer.gradient_accumulation_steps = 2
    trainer.transformer = TinyTransformer()
    trainer.optimizer = torch.optim.SGD(trainer.transformer.parameters(), lr=0.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _: 1.0
    )
    trainer.convert_input_format = lambda batch: batch
    trainer._prepare_input_dict = lambda batch: batch
    trainer.compute_loss = lambda batch, output: {
        "total_loss": (output - 1).square() / 2
    }

    first = trainer._train_step(torch.tensor(2.0), 0)
    assert not first["should_log"]
    assert trainer.transformer.weight.item() == pytest.approx(0.5)
    second = trainer._train_step(torch.tensor(3.0), 1)
    assert second["should_log"]
    assert trainer.transformer.sync_calls == [True, True]
    assert trainer.transformer.weight.item() == pytest.approx(0.35)
    assert trainer.lr_scheduler.last_epoch == 1


def test_causal_alignment_duplicate_timestamps_and_stale_rejection():
    ids, age = causal_indices([0, 0.03, 0.03, 0.06], [0, 0.033, 0.066])
    assert ids.tolist() == [0, 2, 3]
    assert np.all(age >= 0)
    with pytest.raises(ValueError):
        causal_indices([0.1, 0.2], [0])
    with pytest.raises(ValueError):
        causal_indices([0, 0.02], [0.3])
    with pytest.raises(ValueError):
        causal_indices([0, 0.03, 0.02], [0.03])


def test_units_rotation_and_gripper():
    pose = np.tile([0.5, 0.2, 0.1, 1, 0, 0, 0, 1, 0, 0.085], (3, 1))
    converted = normalize_grip(pose, "width_m", 0.085)
    assert np.allclose(converted[:, -1], 1) and np.allclose(pose[:, -1], 0.085)
    validate_pose(converted, "test")
    bad = converted.copy()
    bad[:, 3] = 2
    with pytest.raises(ValueError):
        validate_pose(bad, "bad")
    assert np.allclose(quaternion_to_rot6d([[1, 0, 0, 0]]), [[1, 0, 0, 0, 1, 0]])
    # +90 degrees about z: first column +y, second column -x.
    assert np.allclose(
        quaternion_to_rot6d([[2**-0.5, 0, 0, 2**-0.5]]),
        [[0, 1, 0, -1, 0, 0]],
        atol=1e-6,
    )


class Encoder:
    def __call__(self, rgb):
        x = rgb.float()
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        if x.max() > 1:
            x = x / 255
        return SimpleNamespace(
            tokens=torch.nn.functional.adaptive_avg_pool2d(x, (16, 16)).permute(
                0, 2, 3, 1
            )
        )


def test_chunked_motion_matches_online_and_dense_order():
    from n0_twam.preprocessing.rgb_frame_difference import (
        RGBFrameDifferencePreprocessor,
    )

    frames = np.zeros((9, 64, 64, 3), np.uint8)
    frames[1:5, :32, :32, 0] = 255
    frames[5:, :32, 32:, 1] = 255
    cams = {"top": frames, "wrist": np.flip(frames, axis=2).copy()}
    motion, dense = build_payloads(
        {k: iter(v) for k, v in cams.items()},
        list(range(0, 27, 3)),
        [0, 4, 8],
        Encoder(),
    )
    online = RGBFrameDifferencePreprocessor(
        camera_keys=list(cams), height=256, width=256, dino_encoder=Encoder()
    )(
        {
            k: SimpleNamespace(rgb=torch.from_numpy(v), num_frames=len(v))
            for k, v in cams.items()
        },
        anchor_indices=[0, 4, 8],
        world_time_ids=[0, 1, 2],
    )
    for key in (
        "motion_indices",
        "motion_valid_mask",
        "motion_scores",
        "world_time_id",
        "dino_features",
    ):
        torch.testing.assert_close(motion[key], online[key])
    assert dense["dino_features"].shape == (3, 128, 3)
    assert motion["motion_valid_mask"][0].all()
    assert dense["neoforce_features"].shape[-1] == 0


def test_future_rgb_cannot_change_past_sidecars():
    a = np.zeros((9, 64, 64, 3), np.uint8)
    b = a.copy()
    b[5:] = 255
    first = build_payloads({"top": iter(a)}, list(range(9)), [0, 4, 8], Encoder())[0]
    second = build_payloads({"top": iter(b)}, list(range(9)), [0, 4, 8], Encoder())[0]
    for key in ("motion_indices", "motion_scores", "dino_features"):
        torch.testing.assert_close(first[key][:2], second[key][:2])


def test_dense_loader_accepts_generated_index(tmp_path):
    from n0_twam.dataset.ikv_index import load_dense_index

    frames = np.zeros((5, 64, 64, 3), np.uint8)
    _, dense = build_payloads(
        {"top": iter(frames)}, [0, 3, 6, 9, 12], [0, 4], Encoder()
    )
    p = tmp_path / "index.pth"
    torch.save(dense, p)
    loaded = load_dense_index(
        p,
        camera_keys=["top"],
        patch_size=[1, 2, 2],
        grid_shape=(8, 8),
        latent_frame_ids=[0, 3, 6, 9, 12],
        full_frames=2,
    )
    assert loaded["dino_features"].shape == (2, 64, 3)


def test_accumulation_matches_one_batch_update():
    torch.manual_seed(7)
    initial = torch.nn.Linear(3, 2).state_dict()
    x = torch.randn(32, 3)
    y = torch.randn(32, 2)
    results = []
    for accum in (1, 4, 16, 32):
        m = torch.nn.Linear(3, 2)
        m.load_state_dict(initial)
        opt = torch.optim.AdamW(
            m.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1)
        for a, b in zip(x.chunk(accum), y.chunk(accum)):
            ((m(a) - b).square().mean() / accum).backward()
        opt.step()
        scheduler.step()
        assert scheduler.last_epoch == 1
        results.append(copy.deepcopy(m.state_dict()))
    for out in results[1:]:
        for k in out:
            torch.testing.assert_close(out[k], results[0][k])


def test_full_checkpoint_restores_optimizer_scheduler_rng(tmp_path):
    from n0_twam.task_pipeline.checkpoint import (
        save_training_state,
        load_training_state,
    )
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

    def trainer():
        m = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(m.parameters(), lr=0.001)
        ds = TensorDataset(torch.arange(8))
        loader = DataLoader(
            ds,
            sampler=DistributedSampler(ds, num_replicas=1, rank=0, seed=42),
            generator=torch.Generator().manual_seed(42),
        )
        return SimpleNamespace(
            transformer=m,
            optimizer=opt,
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.9**s),
            config=SimpleNamespace(
                rank=0, world_size=1, task_fingerprint="test", task_mode="baseline"
            ),
            step=1,
            data_epoch=0,
            data_offset=2,
            train_loader=loader,
        )

    a = trainer()
    x = torch.ones(3, 2)
    a.transformer(x).sum().backward()
    a.optimizer.step()
    a.optimizer.zero_grad()
    a.lr_scheduler.step()
    save_training_state(a, tmp_path / "state")
    expected_random = torch.rand(4)
    a.transformer(x).sum().backward()
    a.optimizer.step()
    a.lr_scheduler.step()
    b = trainer()
    load_training_state(b, tmp_path / "state")
    torch.testing.assert_close(torch.rand(4), expected_random)
    assert b.step == 1 and b.data_offset == 2
    b.transformer(x).sum().backward()
    b.optimizer.step()
    b.lr_scheduler.step()
    for p, q in zip(a.transformer.parameters(), b.transformer.parameters()):
        torch.testing.assert_close(p, q)
    assert b.lr_scheduler.state_dict() == a.lr_scheduler.state_dict()
    b.config.world_size = 2
    with pytest.raises(ValueError, match="world_size"):
        load_training_state(b, tmp_path / "state")
    b.config.world_size = 1
    b.config.ikv_train_sample_capacity = True
    with pytest.raises(ValueError, match="ikv_train_sample_capacity"):
        load_training_state(b, tmp_path / "state")


def test_conversion_real_video_and_lerobot_metadata(tmp_path, monkeypatch):
    import av
    import pandas as pd
    from n0_twam.task_pipeline.data import Episode
    from n0_twam.task_pipeline import convert as conversion
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    t = task(tmp_path)
    t["robot"]["cameras"] = {"camera": "observation.images.top"}
    t["robot"]["tactile"] = {"tactile": "observation.images.tactile_a"}
    source = tmp_path / "source.mp4"
    t["source"] = str(source)
    with av.open(str(source), "w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(13):
            rgb = np.zeros((64, 64, 3), np.uint8)
            rgb[:, :, 0] = 180 + i
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(rgb, format="rgb24")
            ):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    a = np.tile([0.5, 0.2, 0.1, 1, 0, 0, 0, 1, 0, 0.7], (13, 1))
    s = a.copy()
    s[:, 0] -= 0.01

    def eps(task):
        for sid in ("episode_0004", "episode_0106"):
            keys = ["observation.images.top", "observation.images.tactile_a"]
            yield Episode(
                sid,
                np.arange(13) / 30,
                a,
                s,
                {"source_id": sid},
                {k: source for k in keys},
                {k: np.arange(13) / 30 for k in keys},
            )

    monkeypatch.setattr(conversion, "episodes", eps)
    conversion.convert(t)
    root = tmp_path / t["name"] / "dataset"
    m = LeRobotDatasetMetadata(repo_id="local/test", root=root)
    assert m.info["total_episodes"] == 2 and m.info["total_frames"] == 26
    assert m.episodes[0]["action_config"] == [{"start_frame": 0, "end_frame": 13}]
    df = pd.read_parquet(root / "data/chunk-000/episode_000001.parquet")
    assert np.stack(df.action).shape == (13, 20)
    assert np.allclose(np.stack(df.action)[:, 0], 0.5)
    assert np.allclose(np.stack(df["observation.state"])[:, 0], 0.49)
    with av.open(
        str(root / "videos/chunk-000/observation.images.top/episode_000000.mp4")
    ) as vid:
        frames = list(vid.decode(video=0))
        assert len(frames) == 13
        rgb = frames[0].to_ndarray(format="rgb24")
        assert rgb[:, :, 0].mean() > 150 and rgb[:, :, 2].mean() < 15
    provenance = [
        json.loads(x)
        for x in (root / "meta/source_provenance.jsonl").read_text().splitlines()
    ]
    assert [x["source_id"] for x in provenance] == ["episode_0004", "episode_0106"]
    marker = root / "conversion_episodes/episode_000000.json"
    assert set(json.loads(marker.read_text())["video_stats"]) == set(t["robot"]["cameras"].values()) | set(t["robot"]["tactile"].values())
    from n0_twam.task_pipeline.config import conversion_identity, fingerprint

    (root / "conversion.json").unlink()
    (root / "conversion_pending.json").write_text(
        json.dumps({"fingerprint": fingerprint(conversion_identity(t))})
    )
    def fail_if_redecoded(_):
        raise AssertionError("Completed episode video statistics were recalculated")
    monkeypatch.setattr(conversion, "image_stats_video", fail_if_redecoded)
    conversion.convert(t)  # interrupted conversion reuses episode statistics
    conversion.convert(t)  # complete configuration is safely reusable
    t["prompt"] = "changed"
    with pytest.raises(ValueError, match="different conversion"):
        conversion.convert(t)


def test_common_start_is_causal_and_large_sensor_gap_rejected():
    from n0_twam.task_pipeline.data import Episode, aligned_timeline

    t = np.arange(31) / 30
    pose = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5], (31, 1))
    ep = Episode("test", t, pose, pose, {}, camera_times={"cam": t + 0.008})
    timeline, ids, ci, audit = aligned_timeline(
        ep, {"fps": 30, "max_alignment_age_s": 0.1}
    )
    assert timeline[0] == pytest.approx(0.008)
    assert ci["cam"][0] == 0 and ids[0] == 0
    assert audit["common_start_offset_s"] == pytest.approx(0.008)
    ep.camera_times = {"cam": np.r_[0, np.arange(0.55, 1.01, 1 / 30)]}
    with pytest.raises(ValueError, match="Alignment age"):
        aligned_timeline(ep, {"fps": 30, "max_alignment_age_s": 0.1})


def test_neosim_observations_do_not_expose_oracle(tmp_path):
    import h5py
    from n0_twam.task_pipeline.neosim import iter_neosim

    root = tmp_path / "raw"
    (root / "hdf5").mkdir(parents=True)
    (root / "metadata").mkdir()
    metadata = {
        "target_slot_idx": 2,
        "selected_slot_idx": 2,
        "query_start_step": 8,
        "cue_start_step": 2,
    }
    (root / "metadata/7.json").write_text(json.dumps(metadata))
    with h5py.File(root / "hdf5/7.hdf5", "w") as h:
        h["step"] = np.arange(0, 10, 2)
        ee = np.tile([0.5, 0.2, 0.3, 1, 0, 0, 0], (5, 1))
        ee[:, 0] += np.arange(5) * 0.01
        h["embodiment/ee"] = ee
        joint = np.zeros((5, 9))
        joint[:, 7:] = 0.0195
        h["embodiment/joint"] = joint
        h["observation/head/rgb"] = np.array([b"placeholder"] * 5)
        h["tactile/left_tactile/rgb"] = np.array([b"placeholder"] * 5)
        h["actor/slot_2"] = np.ones((5, 7)) * 12345
        h["atom/tag"] = np.array([b"hidden target 2"] * 5)
    t = task(tmp_path)
    t["format"] = "neosim_hdf5"
    t["source"] = str(root)
    t["action_labels"] = "next_observation"
    t["robot"].update(
        fps=60,
        simulation_hz=120,
        quaternion_order="wxyz",
        gripper_joint_indices=[7, 8],
        gripper_stroke_m=0.078,
        cameras={"observation/head/rgb": "observation.images.top"},
        tactile={"tactile/left_tactile/rgb": "observation.images.tactile_a"},
    )
    ep = next(iter_neosim(t))
    assert len(ep.timestamps) == 4
    np.testing.assert_allclose(ep.action[:, 0], ee[1:, 0])
    np.testing.assert_allclose(ep.state[:, 0], ee[:-1, 0])
    np.testing.assert_allclose(ep.action[:, -1], 0.5)
    assert ep.metadata["target_slot_idx"] == 2  # audit only
    assert ep.action.shape == (4, 10) and ep.state.shape == (4, 10)
    t["action_labels"] = "unavailable"
    ep = next(iter_neosim(t))
    assert np.isnan(ep.action).all() and len(ep.state) == 5


def test_two_collector_tars_are_one_task_with_distinct_episode_ids(tmp_path):
    import io
    import tarfile
    from n0_twam.task_pipeline.config import load_task, source_inventory
    from n0_twam.task_pipeline.data import iter_collector

    t = task(tmp_path)
    episode = "yanqiang/episode_0000"
    actions = (
        "timestamp_ms,tcp.x,tcp.y,tcp.z,tcp.r1,tcp.r2,tcp.r3,tcp.r4,tcp.r5,tcp.r6,gripper.pos\n"
        + "".join(f"{ms},0.5,0.2,0.3,1,0,0,0,1,0,0.5\n" for ms in (0, 33, 66))
    )
    states = (
        "timestamp_ms,x,y,z,r1,r2,r3,r4,r5,r6,gripper\n"
        + "".join(f"{ms},0.5,0.2,0.3,1,0,0,0,1,0,0.0425\n" for ms in (0, 33, 66))
    )
    stamps = "timestamp_ms,is_new\n0,1\n33,1\n66,1\n"
    for number in range(2):
        archive = tmp_path / f"day_{number}.tar"
        with tarfile.open(archive, "w") as tar:
            entries = {
                f"{episode}/metadata.json": json.dumps(
                    {"version": "v0.6", "fps_config": 30}
                ),
                f"{episode}/actions.eef_pose/data.csv": actions,
                f"{episode}/observation.state.eef_pose/data.csv": states,
            }
            for camera in [*t["robot"]["cameras"], *t["robot"]["tactile"]]:
                entries[f"{episode}/{camera}/timestamps.csv"] = stamps
            for name, contents in entries.items():
                data = contents.encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
    t["source"] = ["day_0.tar", "day_1.tar"]
    path = tmp_path / "task.json"
    path.write_text(json.dumps(t))
    merged = load_task(path)
    assert len(source_inventory(merged["source"])) == 2
    eps = list(iter_collector(merged))
    assert len(eps) == 2
    assert eps[0].source_id != eps[1].source_id
    assert all(ep.action.shape == (3, 10) for ep in eps)
    assert all(np.allclose(ep.state[:, -1], 0.5) for ep in eps)



def test_quality_filter_keeps_only_explicitly_allowed_demonstrations(tmp_path):
    from n0_twam.task_pipeline.data import Episode

    t = task(tmp_path)
    t["allowed_quality_labels"] = ["完全正常"]
    pose = np.tile([0.5, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.5], (3, 1))
    episode = Episode("example", np.arange(3) / 30, pose, pose, {})
    for labels, excluded in [
        (["完全正常"], False),
        (["测试任务"], True),
        (["完全正常", "机械臂放置物体失败"], True),
        ([], True),
    ]:
        episode.metadata = {"quality": {"labels": labels}}
        assert (quality_exclusion_reason(episode, t) is not None) == excluded



def test_image_stats_rgb_matches_full_precision_reference():
    from n0_twam.task_pipeline.convert import image_stats_rgb

    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 256, size=(31, 37, 3), dtype=np.uint8) for _ in range(4)]
    result = image_stats_rgb(iter(frames))
    pixels = np.concatenate([x.reshape(-1, 3) for x in frames]).astype(np.float64) / 255
    for key, expected in {
        "min": pixels.min(axis=0),
        "max": pixels.max(axis=0),
        "mean": pixels.mean(axis=0),
        "std": pixels.std(axis=0),
    }.items():
        np.testing.assert_allclose(
            np.asarray(result[key]).reshape(3), expected, rtol=1e-8, atol=1e-8
        )
    assert result["count"] == [4]


def test_relocated_dino_preserves_task_identity_only_for_identical_weights(tmp_path):
    import hashlib
    from n0_twam.task_pipeline.config import fingerprint

    original = task(tmp_path)
    original["runtime"]["dino_model"] = str(tmp_path / "old-dino")
    expected = fingerprint(original)
    replacement = tmp_path / "new-dino"
    replacement.mkdir()
    weights = replacement / "model.safetensors"
    weights.write_bytes(b"frozen DINO weights")
    relocated = copy.deepcopy(original)
    relocated["runtime"]["dino_model"] = str(replacement)
    relocated["runtime"]["dino_model_fingerprint_compat"] = {
        "original_path": original["runtime"]["dino_model"],
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    assert fingerprint(relocated) == expected
    weights.write_bytes(b"different DINO weights")
    with pytest.raises(ValueError, match="differs from feature source"):
        fingerprint(relocated)



def test_compact_attention_runtime_override_preserves_prepared_task_identity(tmp_path, monkeypatch):
    from n0_twam.task_pipeline.config import fingerprint
    current = task(tmp_path)
    before = fingerprint(current)
    monkeypatch.setenv("IKV_COMPACT_ATTENTION", "1")
    monkeypatch.setenv("IKV_COMPACT_MAX_PACKED_KEYS", "32768")
    compact = training_config(current, "ikv", 4, require_ready=False)
    baseline = training_config(current, "baseline", 4, require_ready=False)
    assert compact.ikv_train_compact_attention is True
    assert compact.ikv_train_compact_max_packed_keys == 32768
    assert baseline.ikv_train_compact_attention is False
    assert fingerprint(current) == before
