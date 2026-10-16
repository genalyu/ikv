"""Download official Qwen3.5-4B ModelScope files and verify every SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with urlopen("https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B/repo/files?Revision=master&Recursive=true",timeout=60) as response:
        manifest=json.load(response)
    files=manifest["Data"]["Files"]
    def digest(path):
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
        return h.hexdigest()
    def download(item):
        relative=Path(item["Path"])
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("invalid model path")
        path=args.output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==item["Size"] and digest(path)==item["Sha256"]:return
        temporary=path.with_name(path.name+".partial")
        revision=quote(item["Revision"],safe="")
        url="https://modelscope.cn/models/Qwen/Qwen3.5-4B/resolve/"+revision+"/"+quote(item["Path"])
        with urlopen(url,timeout=120) as response,temporary.open("wb") as out:
            for block in iter(lambda:response.read(8*1024**2),b""):out.write(block)
        if temporary.stat().st_size!=item["Size"] or digest(temporary)!=item["Sha256"]:
            raise ValueError("model checksum mismatch: "+item["Path"])
        temporary.replace(path);print(item["Path"],"verified",flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(download,files))
    (args.output/"download-manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
