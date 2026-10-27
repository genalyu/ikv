"""Strict native-RGB client; failures abort the policy instead of falling back."""
import base64
import io
import json
import urllib.request
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from .selection import VERSION


class ScorerClient:
    def __init__(self,endpoint,scorer_sha256,timeout=300):
        self.endpoint=endpoint;self.sha=scorer_sha256;self.timeout=timeout
        self.session=str(uuid.uuid4())
        identity=self.call("status")
        if identity["version"]!=VERSION or identity["chunk_size"]!=2 or identity.get("encoder_image_batch")!=1:
            raise ValueError("Online scorer protocol differs from training")

    def call(self,op,**kwargs):
        packet=dict(op=op,session=self.session,scorer_sha256=self.sha,**kwargs)
        req=urllib.request.Request(self.endpoint,data=json.dumps(packet).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=self.timeout) as response:
            return json.load(response)

    def reset(self,prompt):
        self.call("reset",prompt=prompt);self.next_frame=0

    def append(self,observations,cameras,anchors):
        images=[]
        for anchor in anchors:
            triplet=[]
            for camera in cameras:
                raw=np.asarray(observations[anchor][camera])
                if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[-1]!=3:
                    raise ValueError("Scorer observations must be decoded native RGB")
                x=torch.from_numpy(raw.copy()).permute(2,0,1)[None].float()
                x=F.interpolate(x,size=(256,256),mode="bilinear",align_corners=False)
                triplet.append(x[0].permute(1,2,0).round().clamp(0,255).byte().numpy())
            images.append(np.stack(triplet))
        stream=io.BytesIO();np.save(stream,np.stack(images),allow_pickle=False)
        self.call("append",start=self.next_frame,images=base64.b64encode(stream.getvalue()).decode())
        self.next_frame+=len(images)

    def score(self,prefix):
        return torch.tensor(self.call("score",prefix=prefix)["scores"],dtype=torch.float32)

    def score_history(self,history_prefix,observed_prefix):
        """Explicit real-observation conditioning; legacy score remains causal-prefix only."""
        if not 0 <= history_prefix < observed_prefix <= self.next_frame:
            raise ValueError("History must precede available observed context")
        return torch.tensor(self.call("score_history",history_prefix=history_prefix,
            observed_prefix=observed_prefix)["scores"],dtype=torch.float32)
