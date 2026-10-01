#!/usr/bin/env python3
"""Download public NeoSim-Mem metadata and explicit HDF5 episodes; verify SHA256."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

API = "https://www.modelscope.cn/api/v1/datasets/genalyu/neosim-mem/repo"


def listing(root):
    result = []
    seen = set()
    for page in range(1, 10000):
        query = urllib.parse.urlencode(
            dict(
                Revision="master",
                Root=root,
                Recursive="false",
                PageNumber=page,
                PageSize=100,
            )
        )
        with urllib.request.urlopen(API + "/tree?" + query, timeout=60) as response:
            body = json.load(response)
        if body.get("Code") != 200:
            raise RuntimeError(body.get("Message", "Listing failed"))
        files = body["Data"]["Files"]
        if not files:
            break
        fresh = [f for f in files if f["Path"] not in seen]
        if not fresh:
            raise RuntimeError("Server repeated a page; refusing incomplete manifest")
        result.extend(fresh)
        seen.update(f["Path"] for f in fresh)
        if len(files) < 100:
            break
    return result


def download(item, root, prefix):
    relative = PurePosixPath(item["Path"]).relative_to(prefix)
    if ".." in relative.parts or relative.is_absolute():
        raise ValueError("Unsafe dataset path")
    path = root / str(relative)
    path.parent.mkdir(parents=True, exist_ok=True)

    def digest(p):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    expected = item.get("Sha256")
    if (
        path.exists()
        and path.stat().st_size == item["Size"]
        and (not expected or digest(path) == expected)
    ):
        return
    query = urllib.parse.urlencode(dict(Revision="master", FilePath=item["Path"]))
    part = path.with_suffix(path.suffix + ".part")
    h = hashlib.sha256()
    with (
        urllib.request.urlopen(API + "?" + query, timeout=90) as response,
        part.open("wb") as out,
    ):
        for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
            out.write(block)
            h.update(block)
    if part.stat().st_size != item["Size"] or (expected and h.hexdigest() != expected):
        raise ValueError(f"Download verification failed: {path}")
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", default="neosim_mem_hidden_usb_slot")
    p.add_argument(
        "--episodes",
        nargs="*",
        default=["0"],
        help="HDF5 episode IDs; 'all' downloads the entire task, empty = metadata only",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = listing(args.task + "/metadata")
    hdf5 = listing(args.task + "/hdf5")
    blob = [f for f in metadata + hdf5 if f["Type"] == "blob"]
    (args.output / "remote_manifest.json").write_text(json.dumps(blob, indent=2))
    selected = [
        f
        for f in blob
        if "/metadata/" in f["Path"]
        or "all" in args.episodes
        or PurePosixPath(f["Path"]).stem in args.episodes
    ]
    print(
        f"Remote: {len(hdf5)} HDF5 entries; selected download: {sum(f['Size'] for f in selected) / 1e9:.3f} GB",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: download(f, args.output, args.task), selected))
    print(f"Verified {len(selected)} files in {args.output}")


if __name__ == "__main__":
    main()
