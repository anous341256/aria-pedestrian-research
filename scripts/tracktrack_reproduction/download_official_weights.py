"""Download only the two author-published full MOT17 checkpoints for Aria use.

These are author-trained weights, not a download of the MOT17 dataset.
No credentials, uploads, or private data are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import gdown


FILES = [
    ("1. YOLOX/weights/mot17.pth.tar", "1MAb-Bhikx-fWe0VlJON_VMrYIyyyrt-F"),
    ("2. FastReID/weights/mot17_sbs_S50.pth", "1rUYqWIj0nsQ23rDSv8NVx0Rrp3Lco1KP"),
]


def fetch(repo, item):
    name, file_id = item
    destination = repo / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_name(destination.name + ".part")
        result = gdown.download(id=file_id, output=str(partial), quiet=False, resume=True)
        if result is None:
            raise RuntimeError(f"Download failed: {name}")
        if partial.stat().st_size < 1_000_000:
            raise RuntimeError(f"Unexpectedly small checkpoint: {partial}")
        partial.rename(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(destination), "bytes": destination.stat().st_size,
            "sha256": digest.hexdigest(),
            "source": f"https://drive.google.com/file/d/{file_id}/view"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda item: fetch(args.repo, item), FILES))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2), flush=True)
