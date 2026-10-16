"""Download large public assets with byte ranges and verify the recorded SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
import time


def fetch(item, root, workers, original_root):
    path = root / Path(item["path"]).relative_to(original_root)
    if path.is_file() and path.stat().st_size == item["bytes"]:
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4 * 1024**2), b""): digest.update(block)
        if digest.hexdigest() == item["sha256"]: return
    size = item["bytes"]
    part = path.with_suffix(path.suffix + ".range-part")
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    def chunk(start):
        end = min(size - 1, start + 4 * 1024**2 - 1)
        for attempt in range(5):
            try:
                with urlopen(Request(item["source"], headers={"Range": f"bytes={start}-{end}"}), timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Server did not honor byte range")
                    data = response.read()
                if len(data) != end - start + 1: raise ValueError("Short range")
                offset = 0
                while offset < len(data):
                    offset += os.pwrite(fd, data[offset:], start + offset)
                return
            except Exception:
                if attempt == 4: raise
                time.sleep(2**attempt)
    try:
        with ThreadPoolExecutor(workers) as pool:
            for number, _ in enumerate(pool.map(chunk, range(0, size, 4 * 1024**2)), 1):
                if number % 32 == 0: print(path.name, number * 4, "MiB", flush=True)
    finally: os.close(fd)
    digest = hashlib.sha256()
    with part.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""): digest.update(block)
    if digest.hexdigest() != item["sha256"]: raise ValueError("SHA256 mismatch")
    part.replace(path)
    print("verified", path.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--source", choices=["official", "modelscope", "all"], default="official")
    args = parser.parse_args()
    manifest = json.loads((args.root / "assets-manifest.json").read_text())
    original_root = Path(os.path.commonpath([item["path"] for item in manifest]))
    for item in manifest:
        official = "dl.fbaipublicfiles.com" in item["source"]
        selected = args.source == "all" or (official if args.source == "official" else not official)
        if selected and item["bytes"] > 512 * 1024**2:
            fetch(item, args.root, args.workers, original_root)


if __name__ == "__main__": main()
