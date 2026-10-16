"""Semantic patch alignment and relevance integration, without downloads."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import pytest
import torch
from torch import nn
from n0_twam.preprocessing.semantic_patch import FrozenSemanticPatchEncoder
from n0_twam.preprocessing.kv_index import encode_dense_semantic, observed_index, prediction_index
from n0_twam.models.global_kv_retention import RetentionConfig, token_rows
from n0_twam.models.multimodal_kv_retention import make_retention_policy
from test_global_kv_retention import rows, append


class TextTokenizer:
    def tokenize(self, texts):
        return torch.tensor([[1 if texts[0] == "right" else -1]])


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Identity()
        self.calls = 0

    def get_backbone_features(self, pixels):
        self.calls += 1
        b = len(pixels)
        patches = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
                               device=pixels.device).expand(b, -1, -1)
        return patches[:, 0], patches, patches[:, :2] * 0

    def get_class_and_patch_tokens(self, pixels):
        cls, raw, _ = self.get_backbone_features(pixels)
        return cls, raw, raw


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.visual_model = Vision()
        self.text_calls = 0

    def encode_text(self, ids):
        self.text_calls += 1
        # Global half deliberately points in the opposite direction.
        value = ids[:, 0].float()
        return torch.stack((-value, value * 0, value, value * 0), dim=-1)


@pytest.mark.parametrize("backend", ["dinov2_txt", "dinov3_txt"])
def test_dinotxt_uses_local_half_and_one_backbone_pass(backend):
    model = TextModel()
    encoder = FrozenSemanticPatchEncoder(model, TextTokenizer(), backend=backend,
              prompt="right", image_size=(4, 4), patch_size=2, task_temperature=1.)
    output = encoder(torch.zeros(2, 3, 7, 9))
    assert output.tokens.shape == (2, 2, 2, 2)
    assert output.task_relevance.shape == (2, 2, 2)
    assert output.task_relevance[0, 0, 0] > output.task_relevance[0, 0, 1]
    assert model.visual_model.calls == 1
    assert model.text_calls == 1
    encoder.set_prompt("right")
    assert model.text_calls == 1
    old = encoder.provenance()
    encoder.set_prompt("left")
    assert encoder.provenance()["prompt_sha256"] != old["prompt_sha256"]
    changed = encoder(torch.zeros(1, 3, 4, 4))
    assert changed.task_relevance[0, 0, 0] < changed.task_relevance[0, 0, 1]
    encoder.train()
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert not output.tokens.requires_grad


def test_real_transformers_siglip_architecture_patch_head():
    from transformers import SiglipConfig, SiglipModel
    cfg = SiglipConfig(
        text_config=dict(vocab_size=20, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2,
                         max_position_embeddings=4),
        vision_config=dict(hidden_size=8, intermediate_size=16,
                           num_hidden_layers=1, num_attention_heads=2,
                           image_size=8, patch_size=4))
    model = SiglipModel(cfg)
    def tokenizer(texts, **kwargs):
        return dict(input_ids=torch.ones(1, 4, dtype=torch.long))
    encoder = FrozenSemanticPatchEncoder(model, tokenizer, backend="siglip2",
              prompt="right", image_size=(8, 8), patch_size=4)
    calls = []
    hook = model.vision_model.register_forward_hook(lambda *args: calls.append(1))
    result = encoder(torch.rand(2, 3, 8, 8))
    hook.remove()
    assert len(calls) == 1
    assert result.tokens.shape == (2, 2, 2, 8)
    assert result.task_relevance.shape == (2, 2, 2)
    assert torch.isfinite(result.task_relevance).all()
    assert ((result.task_relevance >= 0) & (result.task_relevance <= 1)).all()


def test_dense_camera_token_order_and_prediction_seed():
    encoder = FrozenSemanticPatchEncoder(TextModel(), TextTokenizer(), backend="dinov3_txt",
              prompt="right", image_size=(4, 4), patch_size=2, task_temperature=1.)
    videos = torch.zeros(2, 3, 3, 4, 4)
    index = encode_dense_semantic(videos, [0, 2], (2, 2), encoder)
    assert index["dino"].shape == (16, 2)
    # camera concatenation is along width, not a camera-major flatten.
    assert torch.equal(index["dino"][:8], torch.tensor([
        [1., 0.], [-1., 0.], [1., 0.], [-1., 0.],
        [0., 1.], [0., -1.], [0., 1.], [0., -1.]]))
    real = observed_index(index, 16, "cpu")
    pred = prediction_index(20, "cpu", seed=real)
    assert torch.equal(pred["task_relevance"][:16], real["task_relevance"])
    assert pred["task_relevance"][16:].eq(0).all()


def test_task_term_changes_eviction_and_ignores_predicted_action_touch():
    cfg = RetentionConfig(version=2, video_capacity=2, action_capacity=2,
                          tactile_capacity=2, time_weight=0, visual_weight=0,
                          persistence_weight=0, contact_weight=0, task_weight=2.)
    p = make_retention_policy(6, "cpu", cfg)
    mask = torch.zeros(6, dtype=torch.bool)
    r = rows([0., 1.])
    r["task_relevance"] = torch.tensor([.9, .1])
    slots, _ = append(p, mask, r)
    assert torch.allclose(p.scores(slots), torch.tensor([1.8, .2]))
    incoming = rows([2.])
    incoming["task_relevance"] = torch.tensor([.5])
    _, victims = p.plan(mask, 1, incoming)
    assert victims.tolist() == [slots[1].item()]
    p.data["observation_flag"][slots[0]] = False
    assert p.scores(slots[:1]).item() == 0.
    for kind in (1, 2):
        p.data["kind"][slots[0]] = kind
        p.data["observation_flag"][slots[0]] = True
        assert p.scores(slots[:1]).item() == 0.


def test_token_rows_cfg_relevance_and_validation():
    context = dict(grid_id=torch.zeros(2, 4, 3),
                   index={"task_relevance": torch.tensor([[.1, .8], [.1, .8]])})
    result = token_rows(context, batch_size=2, length=3, main_count=2,
                        action_mode=False, update_cache=2, device="cpu")
    assert torch.allclose(result["task_relevance"], torch.tensor([.1, .8, 0.]))
    context["index"]["task_relevance"][1, 0] = .2
    with pytest.raises(ValueError, match="differs"):
        token_rows(context, batch_size=2, length=3, main_count=2,
                   action_mode=False, update_cache=2, device="cpu")
    with pytest.raises(ValueError, match="task_relevance"):
        observed_index({"task_relevance": torch.tensor([float("nan")])}, 1, "cpu")
    with pytest.raises(ValueError, match="version 2"):
        RetentionConfig(task_weight=1.)


def test_dense_sidecar_stale_backend_prompt_and_missing_relevance(tmp_path):
    # Load this standalone validator without requiring the optional lerobot dependency.
    spec = importlib.util.spec_from_file_location("dense_validator",
        Path(__file__).parents[1] / "n0_twam/dataset/ikv_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    payload = dict(camera_keys=["cam"], patch_size=[1, 2, 2],
                   spatial_grid_shape=[1, 2], frame_ids=[0],
                   dino_features=torch.ones(1, 2, 3), task_relevance=torch.tensor([[.2, .8]]),
                   semantic_provenance={"backend": "dinov3_txt", "prompt": "a"})
    path = tmp_path / "index.pth"
    torch.save(payload, path)
    kwargs = dict(camera_keys=["cam"], patch_size=[1, 2, 2], grid_shape=[1, 2],
                  latent_frame_ids=[0], full_frames=1, require_task=True)
    result = mod.load_dense_index(path, **kwargs, semantic_expected=payload["semantic_provenance"])
    assert result["task_relevance"].shape == (1, 2)
    with pytest.raises(ValueError, match="regenerate"):
        mod.load_dense_index(path, **kwargs, semantic_expected={"backend": "siglip2", "prompt": "a"})
    del payload["task_relevance"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="regenerated"):
        mod.load_dense_index(path, **kwargs)


def test_feature_builder_caches_relevance_on_dense_grid():
    import numpy as np
    from n0_twam.task_pipeline.features import build_payloads
    encoder = FrozenSemanticPatchEncoder(TextModel(), TextTokenizer(), backend="dinov3_txt",
              prompt="right", image_size=(4, 4), patch_size=2, task_temperature=1.)
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    _, dense = build_payloads({"top": iter(frames), "wrist": iter(frames)},
                             list(range(5)), [0, 4], encoder)
    assert dense["task_relevance"].shape == (2, 128)
    assert dense["semantic_provenance"] == encoder.provenance()
    assert encoder.model.visual_model.calls == 4  # two cameras, two anchors


@pytest.mark.parametrize("sparse", [False, True])
def test_training_metadata_gathers_dense_relevance(sparse):
    from n0_twam.models.ikv_training import training_metadata
    indices = torch.tensor([[3, 0]]) if sparse else None
    count = 2 if sparse else 4
    splits = [count, count, 2, 2]
    n = sum(splits)
    latent = dict(task_relevance=torch.tensor([[[.1, .2], [.3, .9]]]),
                  dense_dino_features=torch.ones(1, 2, 2, 3))
    motion = dict(indices=indices, semantic_index=None) if sparse else None
    metadata = training_metadata(
        torch.zeros(1, 4, n), {"kind": torch.zeros(n, dtype=torch.long), "seq": torch.zeros(n, dtype=torch.long)}, splits,
        latent, {"latent": torch.zeros(1, 1, 1, 1, 2)}, motion, version=2)
    expected = torch.tensor([.9, .1]) if sparse else torch.tensor([.1, .2, .3, .9])
    torch.testing.assert_close(metadata["task_relevance"][count:2*count], expected)
    assert metadata["task_relevance"][:count].eq(0).all()


def test_real_training_runs_with_task_term_and_no_future_leakage():
    from copy import deepcopy
    from test_ikv_training import case
    from test_global_kv_retention import tiny_model
    from n0_twam.models.ikv_training import run_ikv_training, build_ikv_support_plan
    args = case()
    h, text, ts, temb, rope, memory = args
    memory["config"].update(capacity=6, retention=dict(version=2,
        video_capacity=2, action_capacity=2, tactile_capacity=2,
        task_weight=1., visual_weight=0., persistence_weight=0.,
        contact_weight=0., query_weight=0., action_query_weight=0.,
        tactile_query_weight=0.))
    memory["rows"]["task_relevance"] = torch.linspace(0, 1, len(memory["rows"]["kind"]))
    model = tiny_model(False).mot
    output = run_ikv_training(model, h, text, ts, temb, rope, memory)
    output.square().mean().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    memory["layout"]["seq"].zero_()
    build_ikv_support_plan(memory, "cpu")
    changed = deepcopy(memory)
    future = changed["layout"]["phase"] >= 2
    changed["rows"]["task_relevance"][future] = 1 - changed["rows"]["task_relevance"][future]
    second = run_ikv_training(model, h.detach(), text, ts, temb, rope, changed)
    torch.testing.assert_close(output[:, ~future], second[:, ~future])


def test_online_server_produces_relevance_and_rejects_mixed_backends():
    from test_dense_kv_index_server import dense_server
    server = dense_server()
    server.job_config.kv_semantic_encoder = {"backend": "dinov3_txt"}
    server.job_config.kv_retention = {"version": 2, "task_weight": 1.}
    encoder = FrozenSemanticPatchEncoder(TextModel(), TextTokenizer(), backend="dinov3_txt",
              prompt="right", image_size=(4, 4), patch_size=2, task_temperature=1.)
    server._kv_dino_encoder = encoder
    videos = torch.zeros(2, 3, 1, 64, 64)
    index = server._prepare_dense_observed_index({}, videos)
    assert index["task_relevance"].shape == index["observation_flag"].shape
    assert index["task_relevance"].gt(0).all()
    with pytest.raises(ValueError, match="both"):
        server._prepare_dense_observed_index({"kv_index": {"dino": index["dino"]}}, videos)
    supplied = {"dino": index["dino"], "task_relevance": index["task_relevance"]}
    with pytest.raises(ValueError, match="provenance"):
        server._prepare_dense_observed_index({"kv_index": supplied}, videos)
    accepted = server._prepare_dense_observed_index({
        "kv_index": supplied, "kv_semantic_provenance": encoder.provenance()}, videos)
    torch.testing.assert_close(accepted["task_relevance"], index["task_relevance"])


def test_training_config_checks_semantic_manifest_and_assets(tmp_path, monkeypatch):
    import json
    from test_task_pipeline import task
    from n0_twam.task_pipeline.config import training_config, paths, fingerprint
    import n0_twam.preprocessing.semantic_patch as semantic
    config = task(tmp_path)
    config["runtime"]["semantic_encoder"] = {"backend": "siglip2", "model": "siglip2-local"}
    root = paths(config)["dataset"]
    root.mkdir(parents=True)
    manifest = root / "features.json"
    manifest.write_text(json.dumps({"task_fingerprint": fingerprint(config),
        "semantic_provenance": {"backend": "siglip2", "assets": {"weights": "a"}}}))
    monkeypatch.setattr(semantic, "_identity", lambda cfg: {"weights": "a"})
    cfg = training_config(config, "ikv", 1, require_ready=False)
    assert cfg.kv_retention["task_weight"] == 1.
    assert cfg.kv_semantic_provenance["backend"] == "siglip2"
    monkeypatch.setattr(semantic, "_identity", lambda cfg: {"weights": "b"})
    with pytest.raises(ValueError, match="assets changed"):
        training_config(config, "ikv", 1, require_ready=False)
