#!/usr/bin/env python3
"""Fetch model weights. All URLs are GitHub release assets, so they are stable
and do not need a Google Drive cookie dance like the upstream RIFE repo."""

import argparse
import os
import sys
import urllib.request

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

RIFE = "https://github.com/HolyWu/vs-rife/releases/download/model"
ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download"

MODELS = {
    # --- frame interpolation ---
    # 4.25 is the author's recommended default; 5 flow blocks + feature encoder
    "flownet_v4.25.pkl": f"{RIFE}/flownet_v4.25.pkl",
    # newest release (2024.09). Same architecture as 4.25
    "flownet_v4.26.pkl": f"{RIFE}/flownet_v4.26.pkl",
    # the old conservative default: 4 blocks, no encoder, fewest surprises
    "flownet_v4.6.pkl": f"{RIFE}/flownet_v4.6.pkl",
    # wider feature encoder (16ch vs 4ch), slower
    "flownet_v4.26.heavy.pkl": f"{RIFE}/flownet_v4.26.heavy.pkl",

    # --- upscaling ---
    "realesr-general-x4v3.pth": f"{ESRGAN}/v0.2.5.0/realesr-general-x4v3.pth",
    "RealESRGAN_x4plus.pth": f"{ESRGAN}/v0.1.0/RealESRGAN_x4plus.pth",
}

DEFAULT = [
    "flownet_v4.25.pkl",
    "realesr-general-x4v3.pth",
    "RealESRGAN_x4plus.pth",
]


def fetch(name, url, dest_dir):
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have {name}")
        return
    print(f"  get  {name}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "video-enhance"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="also fetch the alternative RIFE versions")
    ap.add_argument("--dir", default=MODEL_DIR)
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    wanted = MODELS.keys() if args.all else DEFAULT
    print(f"models -> {args.dir}")
    for name in wanted:
        try:
            fetch(name, MODELS[name], args.dir)
        except Exception as e:
            print(f"  FAILED {name}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
