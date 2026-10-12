#!/usr/bin/env python3
"""Single-worker Sys2 service; use an SSH tunnel for remote clients."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from n0_twam.preprocessing.qwen_sys2 import QwenSys2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--feature-directory",required=True)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8097)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--max-context",type=int,default=32768)
    parser.add_argument("--max-new-tokens",type=int,default=16384)
    parser.add_argument("--max-feature-bytes",type=int,default=8*1024**3)
    args=parser.parse_args()
    engine=QwenSys2(args.model,args.feature_directory,device=args.device,
        max_context=args.max_context,max_new_tokens=args.max_new_tokens,
        max_feature_bytes=args.max_feature_bytes)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length=int(self.headers.get("Content-Length",0))
                if not 0<length<=64*1024**2:
                    raise ValueError("request body size invalid")
                data=json.loads(self.rfile.read(length))
                if self.path=="/encode": result=engine.encode(data["image"])
                elif self.path=="/score": result=engine.score(data)
                else: raise ValueError("unknown route")
                status=200
            except Exception as exc:
                result=dict(error=type(exc).__name__,message=str(exc));status=400
            body=json.dumps(result,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
    HTTPServer((args.host,args.port),Handler).serve_forever()

if __name__=="__main__":main()
