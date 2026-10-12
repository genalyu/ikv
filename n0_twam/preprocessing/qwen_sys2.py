"""Qwen3.5 native multimodal scorer with immutable CPU/disk vision features.

No decoder state survives requests. Cached features are the unmodified native
vision merger outputs; Sys1 patch maps live in text metadata, not hidden states.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import torch
from .sys2 import validate_candidates, validate_scores, unwrap_json_fence


class QwenSys2:
    def __init__(self, model_path, feature_directory, *, device="cuda:0",
                 max_context=32768, max_new_tokens=16384, max_feature_bytes=8*1024**3):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to(device).eval()
        self.device = torch.device(device)
        self.directory = Path(feature_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not 0 < max_new_tokens < max_context <= self.model.config.text_config.max_position_embeddings:
            raise ValueError("invalid Sys2 context/output limits")
        self.max_context, self.max_new_tokens = max_context, max_new_tokens
        self.max_feature_bytes = max_feature_bytes
        self.prompt = Path(__file__).with_name("sys2_prompt.txt").read_text()
        self.last_request = self.last_result = None
        self.feature_bytes = sum(p.stat().st_size for p in self.directory.glob("*.pt"))
        manifest = Path(model_path) / "download-manifest.json"
        revision = manifest.read_bytes() if manifest.exists() else b""
        if not revision:
            raise ValueError("Qwen model directory requires a verified download-manifest.json")
        self.provenance = hashlib.sha256(revision +
            self.model.config.to_json_string().encode()).hexdigest()

    @torch.inference_mode()
    def encode(self, encoded_image):
        from PIL import Image
        raw = base64.b64decode(encoded_image, validate=True)
        digest = hashlib.sha256(self.provenance.encode()+raw).hexdigest()
        frame = int(digest[:15],16)
        destination = self.directory / (str(frame)+".pt")
        if destination.exists():
            packet = torch.load(destination, map_location="cpu", weights_only=True)
            if packet["digest"] != digest:
                raise ValueError("vision frame hash collision")
            return dict(frame=frame)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # Bound per-frame native image resolution explicitly, never alter Sys1 RGB.
        inputs = self.processor.image_processor(images=[image], return_tensors="pt",
                                                size={"shortest_edge":256*32*32,"longest_edge":256*32*32})
        pixels = inputs["pixel_values"].to(self.device)
        grid = inputs["image_grid_thw"].to(self.device)
        output = self.model.model.get_image_features(pixels, grid, return_dict=True)
        features = output.pooler_output
        if isinstance(features, (tuple,list)):
            features = torch.cat(features, dim=0)
        packet = dict(digest=digest, provenance=self.provenance,
                      features=features.detach().cpu(),grid=grid.cpu())
        buffer=io.BytesIO(); torch.save(packet,buffer)
        content=buffer.getvalue()
        if self.feature_bytes+len(content)>self.max_feature_bytes:
            raise MemoryError("Qwen vision index storage limit exceeded; increase the limit or start a new store")
        temporary=destination.with_suffix(".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        self.feature_bytes+=len(content)
        return dict(frame=frame)

    @torch.inference_mode()
    def score(self, request):
        candidates=validate_candidates(request["candidates"])
        if not isinstance(request.get("task"),str) or not request["task"].strip():
            raise ValueError("task must be nonempty text")
        canonical=json.dumps(request,sort_keys=True,allow_nan=False)
        # PP ranks ask the identical question. Reuse only the last validated
        # answer, not language-model state or a conversation prefix.
        if canonical==self.last_request:
            return self.last_result
        if not candidates:
            return dict(visual_scores=[],tactile_scores=[])
        frames=list(dict.fromkeys(c["frame"] for c in candidates))
        features,grids,content=[],[],[]
        for frame in frames:
            packet=torch.load(self.directory/(str(frame)+".pt"),map_location="cpu",weights_only=True)
            if packet["provenance"]!=self.provenance:
                raise ValueError("Qwen feature/model provenance mismatch")
            features.append(packet["features"])
            grids.append(packet["grid"])
            content.append(dict(type="text",text=f"Frame {frame}:"))
            # Native placeholder expansion, exactly one token per merger output.
            content.append(dict(type="text",text="<|vision_start|>"+
                "<|image_pad|>"*len(features[-1])+"<|vision_end|>"))
        metadata=dict(task=request["task"],current_time=request["current_time"],
            visual_candidates=[c for c in candidates if c["modality"]=="visual"],
            tactile_candidates=[c for c in candidates if c["modality"]=="tactile"])
        content.append(dict(type="text",text=json.dumps(metadata,separators=(",",":"),allow_nan=False)))
        messages=[dict(role="system",content=self.prompt),dict(role="user",content=content)]
        text=self.processor.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        inputs=self.processor.tokenizer(text,return_tensors="pt",add_special_tokens=False).to(self.device)
        ids=inputs["input_ids"]
        image_mask=ids==self.model.config.image_token_id
        if int(image_mask.sum()) != sum(len(f) for f in features):
            raise ValueError("native Qwen image placeholders do not match vision features")
        # Preflight full context; never truncate history or accept partial JSON.
        if ids.shape[1]+self.max_new_tokens>self.max_context:
            raise ValueError(f"complete Sys2 request exceeds context: {ids.shape[1]} input + "
                             f"{self.max_new_tokens} output reserve > {self.max_context}")
        features=[f.to(self.device) for f in features]
        inputs["mm_token_type_ids"]=image_mask.long()
        inputs["image_grid_thw"]=torch.cat(grids).to(self.device)
        inputs["pixel_values"]=torch.empty(0,device=self.device)
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        def cached_features(*args,**kwargs):
            return BaseModelOutputWithPooling(pooler_output=tuple(features))
        self.model.model.rope_deltas=None
        with patch.object(self.model.model,"get_image_features",side_effect=cached_features):
            generated=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,
                do_sample=False,use_cache=True,past_key_values=None)
        decoded=self.processor.tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()
        decoded = unwrap_json_fence(decoded)
        try:
            validate_scores(decoded,candidates)
        except (ValueError, TypeError) as exc:
            (self.directory / "last-invalid-response.json").write_text(json.dumps(
                dict(response=decoded,request=request),indent=2))
            raise ValueError("Qwen returned invalid score JSON; see last-invalid-response.json") from exc
        result=json.loads(decoded)
        with (self.directory / "scores.jsonl").open("a") as audit:
            audit.write(json.dumps(dict(request_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                current_time=request["current_time"], candidates=len(candidates),
                input_tokens=int(ids.shape[1]), output_tokens=int(generated.shape[1]-ids.shape[1]),
                scores=result),allow_nan=False)+"\n")
        self.last_request,self.last_result=canonical,result
        return result
