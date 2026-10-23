"""Existing Qwen SFT JSON contract with exact full-distribution sampling."""
from pathlib import Path
import json
import sys
import torch
from PIL import Image
import numpy as np

SCORER = Path(__file__).resolve().parents[1]


class QwenPolicy:
    def __init__(self, config):
        from transformers import AutoProcessor, AutoConfig, Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        sys.path.insert(0, str(SCORER))
        from vision_patch_linear import enable_linear_patch_embed
        from vision_chunking import enable_chunked_vision
        checkpoint = Path(config["sft_checkpoint"])
        if not (checkpoint / "COMPLETE.json").is_file():
            raise ValueError("A completed SFT checkpoint is required")
        self.device = torch.device(config["qwen_device"])
        self.beta = float(config.get("beta", .01))
        self.clip = float(config.get("clip", .2))
        self.std_floor = float(config.get("reward_std_floor", 1e-4))
        self.max_new_tokens = int(config.get("max_new_tokens", 4096))
        self.max_length = int(config.get("max_length", 524288))
        self.processor = AutoProcessor.from_pretrained(config["qwen_model"], local_files_only=True)
        model_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["qwen_model"], config=model_config, dtype=torch.bfloat16,
            attn_implementation=config.get("qwen_attention", "sdpa"), local_files_only=True)
        self.model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True,
                                               adapter_name="policy")
        # Reference is the actual trained SFT adapter, NOT the raw base model.
        self.model.load_adapter(checkpoint, adapter_name="reference", is_trainable=False)
        self.model.set_adapter("policy")
        visual = self.model.base_model.model.model.visual
        visual.load_state_dict(torch.load(checkpoint / "vision_state.pt",
                               map_location="cpu", weights_only=True), strict=True)
        visual.requires_grad_(False)
        enable_linear_patch_embed(visual)
        enable_chunked_vision(visual, config.get("vision_images_per_chunk", 24),
                              deterministic_attention=True)
        self.model.to(self.device).eval()
        self.model.enable_input_require_grads()
        # RL initially updates the language LoRA only. Frozen vision is shared
        # with reference, so switching adapters reproduces the SFT reference.
        self.model.config.use_cache = False
        self.checkpoint_gradients = config.get("gradient_checkpointing", True)
        if self.checkpoint_gradients:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout) and module.p:
                raise ValueError("GRPO likelihood replay requires dropout=0")

    def prompt(self, record):
        sys.path.insert(0, str(SCORER))
        from common import prompt_text
        images = np.load(record["rgb_npy"], mmap_mode="r")
        count = record["query_frame"] + 1
        if images.dtype != np.uint8 or images.shape[1:] != (3,256,256,3) or len(images) < count:
            raise ValueError("Existing scorer RGB contract mismatch")
        batch = self.processor(
            text=[prompt_text(self.processor, record)],
            images=[Image.fromarray(images[f,c]) for f in range(count) for c in range(3)],
            return_tensors="pt", size={"shortest_edge":65536,"longest_edge":65536})
        if batch["input_ids"].shape[1] + self.max_new_tokens > min(
                self.max_length, self.model.config.text_config.max_position_embeddings):
            raise ValueError("Full-history context exceeds limit; no truncation allowed")
        return {k:v.to(self.device) for k,v in batch.items()}

    def logprobs(self, prompt, completion):
        if torch.is_grad_enabled() and getattr(self, "checkpoint_gradients", False):
            self.model.train()
            self.model.base_model.model.model.visual.eval()
        length = prompt["input_ids"].shape[1]
        batch = dict(prompt)
        completion = completion.to(self.device)
        batch["input_ids"] = torch.cat((prompt["input_ids"], completion[None]), 1)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        if "mm_token_type_ids" in batch:
            batch["mm_token_type_ids"] = torch.cat((prompt["mm_token_type_ids"],
                                                  torch.zeros_like(completion[None])), 1)
        # Cached Qwen multimodal RoPE deltas belong to generation; recompute.
        base = self.model.base_model.model.model
        if hasattr(base, "rope_deltas"):
            base.rope_deltas = None
        positions = torch.arange(length-1, length+len(completion)-1, device=self.device)
        logits = self.model(**batch, use_cache=False, logits_to_keep=positions).logits[0]
        return -torch.nn.functional.cross_entropy(
            logits.float(), completion, reduction="none")

    @torch.no_grad()
    def sample(self, prompt, count=4):
        from transformers import GenerationConfig
        completions, old, reference = [], [], []
        eos = self.processor.tokenizer.eos_token_id
        cfg = GenerationConfig(do_sample=True, temperature=1., top_k=0, top_p=1.,
                               repetition_penalty=1., max_new_tokens=self.max_new_tokens,
                               eos_token_id=eos, pad_token_id=eos, use_cache=True)
        self.model.set_adapter("policy")
        self.model.eval()
        for _ in range(count):
            base = self.model.base_model.model.model
            if hasattr(base, "rope_deltas"):
                base.rope_deltas = None
            ids = self.model.generate(**prompt, generation_config=cfg)
            completion = ids[0, prompt["input_ids"].shape[1]:].detach()
            completions.append(completion)
            old.append(self.logprobs(prompt, completion).detach().cpu())
        self.model.set_adapter("reference")
        try:
            for completion in completions:
                reference.append(self.logprobs(prompt, completion).detach().cpu())
        finally:
            self.model.set_adapter("policy")
        return completions, old, reference

    def decode(self, completion, record):
        sys.path.insert(0, str(SCORER))
        from common import decode_scores
        if int(completion[-1]) != self.processor.tokenizer.eos_token_id:
            raise ValueError("Truncated completion")
        text = self.processor.tokenizer.decode(completion, skip_special_tokens=True)
        return decode_scores(json.loads(text), record["query_frame"] + 1)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(path, selected_adapters=["policy"], safe_serialization=True)
        # PEFT nests a non-default adapter; flatten it so existing infer.py
        # accepts this checkpoint directory without a special RL code path.
        for child in (path/"policy").iterdir():
            child.rename(path/child.name)
        (path/"policy").rmdir()
        self.processor.save_pretrained(path)
        self.model.config.save_pretrained(path)
        torch.save(self.model.base_model.model.model.visual.state_dict(), path/"vision_state.pt")
