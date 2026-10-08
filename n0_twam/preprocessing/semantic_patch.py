"""Frozen whole-image patch indexing with optional prompt relevance.

DINO text adapters must match their backbone. SigLIP2 singleton-patch pooling
uses its trained image head, but remains a dense relevance heuristic rather
than a calibrated patch probability. No crop-per-patch inference is used.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import importlib
import torch
from torch import nn
from torch.nn import functional as F
from .dinov2 import FrozenDinoV2PatchEncoder


@dataclass
class SemanticPatchOutput:
    tokens: torch.Tensor
    grid_size: tuple
    resized_size: tuple
    task_relevance: torch.Tensor


class FrozenSemanticPatchEncoder(FrozenDinoV2PatchEncoder):
    def __init__(self, model, tokenizer, *, backend, prompt, image_size=(224, 224),
                 patch_size=16, image_mean=(.485, .456, .406),
                 image_std=(.229, .224, .225), task_temperature=.07,
                 task_bias=0., identity=None):
        if backend not in ("dinov2_txt", "dinov3_txt", "siglip2"):
            raise ValueError("unknown semantic patch backend")
        if not math.isfinite(task_temperature) or task_temperature <= 0:
            raise ValueError("task_temperature must be finite and positive")
        if not math.isfinite(task_bias):
            raise ValueError("task_bias must be finite")
        super().__init__(model, image_size=image_size, patch_size=patch_size,
                         image_mean=image_mean, image_std=image_std)
        self.tokenizer, self.backend = tokenizer, backend
        self.task_temperature, self.task_bias = task_temperature, task_bias
        self.identity = identity or {}
        self._prompt = None
        self._text = None
        self.set_prompt(prompt)

    @torch.no_grad()
    def set_prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("semantic patch indexing requires one nonempty episode prompt")
        if prompt == self._prompt:
            return
        device = next(self.model.parameters()).device
        if self.backend == "siglip2":
            inputs = self.tokenizer([prompt], padding="max_length",
                                    truncation=True, return_tensors="pt",
                                    max_length=self.model.config.text_config.max_position_embeddings)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text = self.model.get_text_features(**inputs)
            if not isinstance(text, torch.Tensor):
                text = text.pooler_output
        else:
            tokenize = getattr(self.tokenizer, "tokenize", self.tokenizer)
            text = self.model.encode_text(tokenize([prompt]).to(device))
            # dino.txt concatenates CLS and pooled patch embedding spaces.
            if text.shape[-1] % 2:
                raise ValueError("dino.txt requires concatenated global/local text features")
            text = text[:, text.shape[-1] // 2:]
        if text.ndim != 2 or len(text) != 1 or not torch.isfinite(text).all():
            raise ValueError("invalid prompt embedding")
        if text.float().norm(dim=-1).eq(0).any():
            raise ValueError("zero prompt embedding cannot define task relevance")
        self._text = F.normalize(text.detach().float(), dim=-1)
        self._prompt = prompt

    def provenance(self):
        return dict(schema=1, backend=self.backend, assets=self.identity,
                    image_size=list(self.image_size), patch_size=list(self.patch_size),
                    model_dtype=str(next(self.model.parameters()).dtype),
                    image_mean=list(self.image_mean.flatten().tolist()),
                    image_std=list(self.image_std.flatten().tolist()),
                    prompt_sha256=hashlib.sha256(self._prompt.encode()).hexdigest(),
                    task_temperature=self.task_temperature, task_bias=self.task_bias,
                    task_projection=("singleton_attention_pool" if self.backend == "siglip2"
                                     else "dinotxt_local_half"),
                    resize="full_field_bicubic_antialias",
                    score="sigmoid((cosine-bias)/temperature)")

    @torch.no_grad()
    def forward(self, rgb):
        pixels = self._prepare_pixel_values(rgb)
        if self.backend == "dinov2_txt":
            visual = self.model.visual_model
            cls, raw, registers = visual.get_backbone_features(pixels)
            adapted = visual.head(torch.cat((cls[:, None], registers, raw), dim=1))
            aligned = adapted[:, 1 + registers.shape[1]:]
        elif self.backend == "dinov3_txt":
            _, aligned, raw = self.model.visual_model.get_class_and_patch_tokens(pixels)
        else:
            vision = self.model.vision_model
            raw = vision(pixel_values=pixels, return_dict=True).last_hidden_state
            # The same learned output transform as image/text scoring. Each
            # singleton pool corresponds to exactly one spatial patch.
            head = vision.head
            aligned = torch.cat([head(block[:, None]) for block in
                                 raw.flatten(0, 1).split(256)], dim=0).reshape_as(raw)
        grid = (self.image_size[0] // self.patch_size[0],
                self.image_size[1] // self.patch_size[1])
        if raw.ndim != 3 or raw.shape[:2] != (len(rgb), grid[0] * grid[1]):
            raise ValueError("backend patches do not match the configured spatial grid")
        if aligned.shape[:2] != raw.shape[:2] or aligned.shape[-1] != self._text.shape[-1]:
            raise ValueError("patch/text projection dimensions differ")
        if not torch.isfinite(raw).all() or not torch.isfinite(aligned).all():
            raise ValueError("nonfinite semantic patches")
        text = self._text.to(aligned.device)
        cosine = (F.normalize(aligned.float(), dim=-1) * text[:, None]).sum(-1)
        relevance = torch.sigmoid((cosine - self.task_bias) / self.task_temperature)
        relevance *= (raw.ne(0).any(-1) & aligned.ne(0).any(-1))
        return SemanticPatchOutput(
            raw.detach().float().reshape(len(rgb), *grid, -1),
            grid, self.image_size, relevance.reshape(len(rgb), *grid).detach())


def _file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _identity(config):
    """Hash asset bytes once at load, not once per image."""
    assets = {"torch_version": str(torch.__version__)}
    if config.get("repo"):
        repo = Path(config["repo"]).expanduser().resolve()
        revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True,
                               text=True, check=True).stdout
        if dirty:
            raise ValueError("Use a clean pinned official DINO checkout")
        assets["repo_revision"] = revision
    if config.get("backend") == "siglip2":
        import transformers
        assets["transformers_version"] = transformers.__version__
    for key in ("backbone_weights", "text_weights", "head_weights", "bpe"):
        if config.get(key):
            path = Path(_file(config[key]))
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            assets[key] = digest.hexdigest()
    model_path = config.get("model")
    if model_path:
        root = Path(model_path)
        for path in sorted(root.glob("*")):
            if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".model"):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                assets[path.name] = digest.hexdigest()
    return assets


def load_patch_encoder(config, *, device="cpu", prompt=None):
    """Local assets only. Legacy dinov2 remains available without text scoring.

    Semantic config keys: backend, repo (local official DINO checkout),
    backbone_weights, head_weights, text_weights (v2 only), bpe, model (SigLIP2),
    image_size, task_temperature, task_bias.
    """
    config = dict(config)
    backend = config.get("backend", "dinov2")
    dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if config.get("dtype", "float32") not in dtypes:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    dtype = dtypes[config.get("dtype", "float32")]
    if backend == "dinov2":
        return FrozenDinoV2PatchEncoder.from_pretrained(
            config["model"], local_files_only=True, device=device,
            image_size=tuple(config.get("image_size", (224, 224))), torch_dtype=dtype)
    if backend in ("dinov2_txt", "dinov3_txt"):
        repo = Path(config["repo"]).expanduser().resolve()
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError("repo must be a local official DINO checkout")
        backbone = _file(config["backbone_weights"])
        head = _file(config["head_weights"])
        bpe = _file(config["bpe"])
        if backend == "dinov3_txt":
            # Import the text entrypoint directly; hubconf also imports unrelated
            # segmentation/detection code and its optional dependencies.
            sys.path.insert(0, str(repo))
            try:
                module = importlib.import_module("dinov3.hub.dinotxt")
            finally:
                sys.path.remove(str(repo))
            if not Path(module.__file__).resolve().is_relative_to(repo):
                raise ValueError("DINOv3 module came from a different checkout; use a fresh process")
            model, tokenizer = module.dinov3_vitl16_dinotxt_tet1280d20h24l(
                weights=head, backbone_weights=backbone, bpe_path_or_url=bpe)
            patch = 16
        else:
            # The official v2 hub text entrypoint downloads weights unconditionally.
            # Build the identical architecture and load all three local assets.
            backbone_model = torch.hub.load(str(repo), "dinov2_vitl14_reg",
                                            source="local", pretrained=False)
            from dinov2.hub.text.dinov2_wrapper import DINOv2Wrapper
            from dinov2.hub.text.dinotxt_model import DinoTxt, DinoTxtConfig
            from dinov2.hub.text.text_transformer import TextTransformer
            from dinov2.hub.text.tokenizer import Tokenizer
            backbone_model.load_state_dict(torch.load(backbone, map_location="cpu",
                                                      weights_only=True), strict=True)
            cfg = DinoTxtConfig(embed_dim=2048, vision_model_use_patch_tokens=True,
                                vision_model_num_head_blocks=2,
                                text_model_use_linear_projection=True,
                                text_model_tokens_pooler_type="argmax")
            text = TextTransformer(context_length=77, vocab_size=49408, dim=1280,
                                   num_heads=20, num_layers=24, ffn_ratio=4,
                                   is_causal=True, ls_init_value=None, dropout_prob=0.)
            model = DinoTxt(cfg, DINOv2Wrapper(backbone_model), text)
            model.visual_model.head.load_state_dict(torch.load(
                head, map_location="cpu", weights_only=True), strict=True)
            model.text_model.load_state_dict(torch.load(
                _file(config["text_weights"]), map_location="cpu", weights_only=True), strict=True)
            tokenizer = Tokenizer(vocab_path=bpe)
            patch = 14
        mean, std = (.485, .456, .406), (.229, .224, .225)
        image_size = tuple(config.get("image_size", (224, 224)))
    elif backend == "siglip2":
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
        path = config["model"]
        processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        vc = model.config.vision_config
        if getattr(model.config, "model_type", "") != "siglip" or not hasattr(model.vision_model, "head"):
            raise ValueError("Only fixed-resolution SigLIP2 checkpoints are supported; NaFlex is unsupported")
        if "siglip2" not in str(getattr(model.config, "_name_or_path", path)).lower():
            raise ValueError("model must identify a SigLIP2 checkpoint (not SigLIP1)")
        patch = vc.patch_size
        image_size = tuple(config.get("image_size", (vc.image_size, vc.image_size)))
        if image_size != (vc.image_size, vc.image_size):
            raise ValueError("SigLIP2 fixed-resolution model must use its native image size")
        mean, std = tuple(processor.image_mean), tuple(processor.image_std)
    else:
        raise ValueError(f"unknown patch backend: {backend}")
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return FrozenSemanticPatchEncoder(
        model, tokenizer, backend=backend, prompt=prompt,
        image_size=image_size, patch_size=patch, image_mean=mean, image_std=std,
        task_temperature=float(config.get("task_temperature", .07)),
        task_bias=float(config.get("task_bias", 0.)), identity=_identity(config))
