"""Frozen Qwen scorer service; independent environment from policy inference."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .selection import VERSION, scorer_to_native
from .sidecar import verify_binding, sha256
from .feature_cache import CANONICAL_IMAGE_BATCH


class ObservedPrefix:
    def __init__(self,encoder,scorer,prompt):
        if getattr(encoder,"image_batch",CANONICAL_IMAGE_BATCH)!=CANONICAL_IMAGE_BATCH:raise ValueError("Streaming encoder batch differs from native plans")
        self.encoder,self.scorer=encoder,scorer
        self.text=encoder.task(prompt).detach().cpu()
        self.frames=[];self.digests=[];self.last_scores=None

    def append(self,start,images):
        if images.dtype!=np.uint8 or images.ndim!=5 or images.shape[1:]!=(3,256,256,3):
            raise ValueError("Expected native RGB [F,3,256,256,3] uint8")
        if len(images) not in (1,2) or start<0 or start>len(self.frames):
            raise ValueError("Missing/reordered observation prefix")
        for offset,triplet in enumerate(images):
            frame=start+offset
            import hashlib
            digest=hashlib.sha256(triplet.tobytes()).hexdigest()
            if frame<len(self.frames):
                if self.digests[frame]!=digest:raise ValueError("Observation changed during retry")
                continue
            with torch.inference_mode():
                values=self.encoder.images([Image.fromarray(x) for x in triplet]).detach().cpu()
            self.frames.append(values.reshape(3,8,8,-1))
            self.digests.append(digest)

    def score_history(self,history_prefix,observed_prefix):
        """Condition on real observations; return only strictly historical candidates.

        Prefixes are exclusive frame counts, not original RGB frame numbers.
        Never append imagined/teacher-forced future targets as observations.
        """
        if not 0 <= history_prefix < observed_prefix <= len(self.frames):
            raise ValueError("History must precede available observed context")
        return self.score(observed_prefix)[:history_prefix*192]

    def score(self,n):
        if n<1 or n>len(self.frames):
            raise ValueError("Scorer cannot read unavailable observations")
        if self.last_scores is not None and self.last_scores[0]==n:
            return self.last_scores[1]
        features=torch.stack(self.frames[:n])
        times=torch.arange(n,dtype=torch.float32)*12/30
        xy=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing="ij"),-1).reshape(-1,2)/7.
        packet=dict(features=features.reshape(n*192,-1),text=self.text,times=times.repeat_interleave(192),
            xy=xy.repeat(n*3,1),modalities=torch.zeros(n*192,dtype=torch.long),
            sensors=torch.arange(3).repeat_interleave(64).repeat(n),query_time=float(times[-1]),
            ids=list(range(n*192)),encoder_revision=self.scorer.info["encoder_revision"])
        scores=self.scorer.score(packet,{0:0,1:0})["scores"]
        scores=scorer_to_native(scores.reshape(n,3,8,8)).flatten()
        self.last_scores=(n,scores)
        return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binding",required=True);p.add_argument("--port",type=int,default=29841)
    p.add_argument("--host",default="127.0.0.1");p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    binding=verify_binding(dict(binding=args.binding))
    if binding.get("encoder_image_batch")!=CANONICAL_IMAGE_BATCH:raise ValueError("Online scorer requires canonical one-image encoder plans")
    if binding["chunk_size"]!=2:raise ValueError("Only native two-frame Memory serving is supported")
    for name,digest in binding["encoder_sha256"].items():
        if sha256(Path(binding["encoder"])/name)!=digest:raise ValueError("Encoder changed after binding")
    from training.parallel_scorer.encoders import QwenEncoders
    from training.parallel_scorer.runtime import RuntimeScorer
    scorer=RuntimeScorer(binding["checkpoint"],args.device)
    encoder=QwenEncoders(binding["encoder"],binding["encoder_revision"],args.device,image_batch=CANONICAL_IMAGE_BATCH).eval()
    sessions={}
    identity=dict(version=VERSION,encoder_image_batch=CANONICAL_IMAGE_BATCH,scorer_sha256=binding["scorer_sha256"],chunk_size=2)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length=int(self.headers["Content-Length"])
                if length<1 or length>16*1024**2:raise ValueError("Invalid request size")
                req=json.loads(self.rfile.read(length))
                if req["scorer_sha256"]!=identity["scorer_sha256"]:raise ValueError("Scorer binding mismatch")
                op=req["op"];sid=req.get("session")
                if op=="status":out=identity
                elif op=="reset":
                    if not req["prompt"].strip():raise ValueError("Missing task prompt")
                    if sid not in sessions and len(sessions)>=8:raise ValueError("Release an old scorer session first")
                    sessions[sid]=ObservedPrefix(encoder,scorer,req["prompt"]);out=identity
                elif op=="release":sessions.pop(sid,None);out=dict(released=True)
                elif op=="append":
                    data=base64.b64decode(req["images"],validate=True)
                    images=np.load(io.BytesIO(data),allow_pickle=False)
                    sessions[sid].append(int(req["start"]),images);out=dict(frames=len(sessions[sid].frames))
                elif op=="score_history":
                    out=dict(scores=sessions[sid].score_history(
                        int(req["history_prefix"]),int(req["observed_prefix"])).tolist())
                elif op=="score":out=dict(scores=sessions[sid].score(int(req["prefix"])).tolist())
                else:raise ValueError("Unknown operation")
                payload=json.dumps(out,allow_nan=False).encode();code=200
            except Exception as exc:
                payload=json.dumps(dict(error=str(exc))).encode();code=400
            self.send_response(code);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload)
    HTTPServer((args.host,args.port),Handler).serve_forever()

if __name__=="__main__":main()
