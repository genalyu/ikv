"""Immutable context-independent features, shared by all frozen scorer sizes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import av
import torch
import torch.nn.functional as F
from PIL import Image
from .sidecar import sha256

CANONICAL_IMAGE_BATCH = 1

def encoder_environment():
    from importlib.metadata import version
    return dict(packages={name:version(name) for name in ["torch","transformers","numpy","pillow","av"]},
        cuda=str(torch.version.cuda),capability=list(torch.cuda.get_device_capability()))


def extract(video, anchors):
    wanted=set(map(int,anchors));images={}
    with av.open(str(video)) as reader:
        reader.streams.video[0].thread_count=2
        for i,frame in enumerate(reader.decode(video=0)):
            if i in wanted:
                x=torch.from_numpy(frame.to_ndarray(format="rgb24").copy()).permute(2,0,1)[None].float()
                x=F.interpolate(x,size=(256,256),mode="bilinear",align_corners=False)
                images[i]=Image.fromarray(x[0].permute(1,2,0).round().clamp(0,255).byte().numpy())
            if len(images)==len(wanted):break
    if set(images)!=wanted:
        raise ValueError("Video does not contain all actual VAE frame anchors")
    return [images[int(i)] for i in anchors]



def episode_features(root,episode,length,cameras,prompt,revision,encoder_hashes,get_encoder,cache_root=None):
    root=Path(root);name=f"episode_{episode:06d}_0_{length}"
    payloads=[];sources={};videos={}
    for camera in cameras:
        files=list((root/"latents").glob(f"chunk-*/{camera}/{name}.pth"))
        if len(files)!=1:raise ValueError("Missing or ambiguous corrected latent")
        payload=torch.load(files[0],map_location="cpu",weights_only=True)
        if payload.get("frame_sampling_contract")!="explicit-source-stride-v1" or payload.get("rgb_resize_mode")!="torch_bilinear":
            raise ValueError("Legacy image/time latents forbidden")
        if payload.get("text")!=prompt:raise ValueError("Latent/scorer prompt mismatch")
        anchors=torch.as_tensor(payload["frame_ids"],dtype=torch.long)[::4]
        if payloads and not torch.equal(anchors,payloads[0]):raise ValueError("Camera clocks disagree")
        payloads.append(anchors);sources[camera]=sha256(files[0])
        files=list((root/"videos").glob(f"chunk-*/{camera}/episode_{episode:06d}.mp4"))
        if len(files)!=1:raise ValueError("Missing corrected video")
        videos[camera]=files[0]
    from training.parallel_scorer import encoders
    from training.qwen_scorer import vision_patch_linear
    fps=float(json.loads((root/"meta/info.json").read_text())["fps"])
    identity=dict(version="frozen-qwen-native-anchor-features-v1",encoder_revision=revision,encoder_environment=encoder_environment(),
        encoder_sha256=encoder_hashes,feature_code_sha256=sha256(__file__),encoder_code_sha256=sha256(encoders.__file__),
        patch_code_sha256=sha256(vision_patch_linear.__file__),dtype="bfloat16",image_batch=CANONICAL_IMAGE_BATCH,
        cameras=list(cameras),frame_ids=anchors.tolist(),prompt=prompt,source_fps=fps,
        latent_sha256=sources,video_sha256={k:sha256(v) for k,v in videos.items()},
        rgb_contract="canonical-RGB-torch-bilinear-u8-round-v1")
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    target=None if cache_root is None else Path(cache_root)/f"{key}.pt"
    if target is not None and target.exists():
        data=torch.load(target,map_location="cpu",weights_only=True)
        if data["identity"]!=identity:raise ValueError("Feature cache identity mismatch")
    else:
        encoder=get_encoder()
        if encoder.image_batch!=CANONICAL_IMAGE_BATCH:raise ValueError("Native Memory features require fixed one-image execution")
        features=[]
        for camera in cameras:
            with torch.inference_mode():
                values=encoder.images(extract(videos[camera],anchors)).detach().cpu()
            features.append(values.reshape(len(anchors),8,8,-1))
        data=dict(identity=identity,features=torch.stack(features,dim=1).contiguous(),
            text=encoder.task(prompt).detach().cpu(),times=anchors.float()/fps)
        if data["features"].dtype!=torch.bfloat16:raise ValueError("Frozen cache requires CUDA BF16 encoders")
        if target is not None:
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent,suffix=".tmp",delete=False) as handle:
                temporary=Path(handle.name)
            try:torch.save(data,temporary);os.replace(temporary,target)
            finally:temporary.unlink(missing_ok=True)
    if not torch.isfinite(data["features"]).all() or not torch.isfinite(data["text"]).all():
        raise ValueError("Nonfinite cached encoder features")
    return data["features"],data["text"],data["times"],anchors,sources,key
