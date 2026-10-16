#!/usr/bin/env python3
"""Real-weight native-feature parity and independent scoring smoke test."""
import argparse
import base64
import io
import json
import time
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image, ImageDraw
from transformers.modeling_outputs import BaseModelOutputWithPooling
from n0_twam.preprocessing.qwen_sys2 import QwenSys2


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True)
    p.add_argument("--feature-directory",required=True);p.add_argument("--report",required=True)
    args=p.parse_args()
    engine=QwenSys2(args.model,args.feature_directory,max_new_tokens=1024)
    image=Image.new("RGB",(512,512),"gray");draw=ImageDraw.Draw(image)
    draw.rectangle((60,150,180,360),fill="red");draw.rectangle((300,150,420,360),fill="black")
    b=io.BytesIO();image.save(b,format="PNG")
    frame=engine.encode(base64.b64encode(b.getvalue()).decode())["frame"]
    packet=torch.load(Path(args.feature_directory)/(str(frame)+".pt"),weights_only=True)
    features=packet["features"].to(engine.device);grid=packet["grid"].to(engine.device)
    content="<|vision_start|>"+"<|image_pad|>"*len(features)+"<|vision_end|>Describe the image."
    text=engine.processor.apply_chat_template([dict(role="user",content=content)],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=engine.processor.tokenizer(text,return_tensors="pt",add_special_tokens=False).to(engine.device)
    inputs["mm_token_type_ids"]=(inputs["input_ids"]==engine.model.config.image_token_id).long()
    inputs["image_grid_thw"]=grid
    pixels=engine.processor.image_processor(images=[image],return_tensors="pt",
        size={"shortest_edge":256*32*32,"longest_edge":256*32*32})["pixel_values"].to(engine.device)
    with torch.inference_mode():
        engine.model.model.rope_deltas=None
        baseline=engine.model(**inputs,pixel_values=pixels,use_cache=False,logits_to_keep=1).logits
        engine.model.model.rope_deltas=None
        with patch.object(engine.model.model,"get_image_features",return_value=BaseModelOutputWithPooling(pooler_output=(features,))):
            reused=engine.model(**inputs,pixel_values=torch.empty(0,device=engine.device),use_cache=False,logits_to_keep=1).logits
        torch.testing.assert_close(reused,baseline,rtol=0,atol=0)
    candidates=[dict(id="v0",time=0.,modality="visual",frame=frame,patch=0,grid=[1,2]),
                dict(id="v1",time=0.,modality="visual",frame=frame,patch=1,grid=[1,2]),
                dict(id="t0",time=0.,modality="tactile",frame=frame,patch=0,grid=[1,1],contact=True)]
    start=time.monotonic()
    result=engine.score(dict(task="Remember the red marked slot and insert the plug into it.",current_time=1.,candidates=candidates))
    report=dict(native_feature_logits_exact=True,scores=result,score_seconds=time.monotonic()-start,
                feature_tokens=len(features),model=args.model,transformers="5.3.0",
                disclaimer="Synthetic image smoke test, not tactile discrimination or task success validation")
    Path(args.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
