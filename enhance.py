#!/usr/bin/env python3
"""
Frame interpolation + upscaling for video, streamed through ffmpeg pipes.

Nothing is written to disk as an image sequence: frames go
ffmpeg(decode) -> stdout pipe -> torch -> stdin pipe -> ffmpeg(encode).
A ten second 480p clip going to 4x/60fps would otherwise be several GB of PNGs.

Order of operations is interpolate-then-upscale. Interpolating at the source
resolution is 16x cheaper than doing it after a 4x upscale, and the quality
difference is small because RIFE's flow estimation does not gain much from
detail that was hallucinated rather than measured.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

from esrgan import Upscaler
from rife import RifeInterpolator

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))


# ---------------------------------------------------------------- probing
def probe(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,pix_fmt",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    info = json.loads(subprocess.check_output(cmd))
    st = info["streams"][0]
    fps = Fraction(st["r_frame_rate"])
    nb = st.get("nb_frames")
    if nb and nb != "N/A":
        n_frames = int(nb)
    else:
        dur = float(info.get("format", {}).get("duration", 0) or 0)
        n_frames = int(dur * float(fps)) if dur else 0
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": fps,
        "n_frames": n_frames,
    }


def encoder_works(name):
    """Being *built* with an encoder is not the same as being able to *open*
    it. A static ffmpeg built against a newer NVENC SDK than the host driver
    supports will happily list hevc_nvenc and then fail at open time with
    "Driver does not support the required nvenc API version". The only
    reliable check is to encode a frame and see what happens."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "nullsrc=s=256x256", "-frames:v", "1",
             "-c:v", name, "-f", "null", "-"],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------- io pipes
def decoder(path, width, height):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8,
    )
    frame_bytes = width * height * 3
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        yield np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
    proc.stdout.close()
    proc.wait()


def encoder(out_path, width, height, fps, src_path, args):
    vcodec, extra = args.codec, []
    if vcodec == "auto":
        for cand in ("hevc_nvenc", "h264_nvenc"):
            if encoder_works(cand):
                vcodec = cand
                break
        else:
            vcodec = "libx264"
            print("[encode] no usable NVENC encoder (driver too old for this "
                  "ffmpeg build?), falling back to libx264", file=sys.stderr)

    if "nvenc" in vcodec:
        extra = ["-preset", "p5", "-tune", "hq", "-rc", "vbr",
                 "-cq", str(args.quality), "-b:v", "0"]
    else:
        extra = ["-preset", args.preset, "-crf", str(args.quality)]

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-i", src_path,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", vcodec, *extra,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        out_path,
    ]
    print(f"[encode] {vcodec} -> {out_path}", file=sys.stderr)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10 ** 8)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Interpolate and upscale video")
    p.add_argument("input")
    p.add_argument("output")

    p.add_argument("--fps", default=None,
                   help="exact target frame rate, e.g. 60 or 60000/1001. "
                        "Overrides --interp-factor")
    p.add_argument("--interp-factor", type=float, default=2.0,
                   help="frame rate multiplier, default 2 (doubles it). "
                        "Overridden by --fps; disable with --no-interp")
    p.add_argument("--no-interp", action="store_true")

    p.add_argument("--rife-model", default="flownet_v4.25",
                   help="flownet_v4.25 (default) | flownet_v4.26 | flownet_v4.6 | "
                        "flownet_v4.26.heavy")
    p.add_argument("--upscale-model", default="realesr-general-x4v3",
                   help="file stem in MODEL_DIR: realesr-general-x4v3 | RealESRGAN_x4plus")
    p.add_argument("--no-upscale", action="store_true")
    p.add_argument("--tile", type=int, default=0,
                   help="tile size for the upscaler; use 256 or 512 if you hit OOM")
    p.add_argument("--upscale-factor", type=float, default=None,
                   help="output scale relative to the source. Default is the "
                        "model's native scale (4x) with no resampling; set e.g. "
                        "2 to resample the 4x result down")
    p.add_argument("--out-height", type=int, default=None,
                   help="exact output height, overriding --upscale-factor")

    p.add_argument("--rife-scale", type=float, default=1.0,
                   help="0.5 for 4K input, 2.0 for very fast motion at low resolution")
    p.add_argument("--fp32", action="store_true", help="disable half precision")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--codec", default="auto")
    p.add_argument("--quality", type=int, default=19, help="CRF for x264/x265, CQ for NVENC")
    p.add_argument("--preset", default="slow")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")

    meta = probe(args.input)
    w, h, src_fps = meta["width"], meta["height"], meta["fps"]
    print(f"[input ] {w}x{h} @ {float(src_fps):.3f} fps, ~{meta['n_frames']} frames",
          file=sys.stderr)

    # ---- resolve target frame rate
    if args.no_interp:
        out_fps = src_fps
    elif args.fps:
        out_fps = Fraction(args.fps)
    elif args.interp_factor:
        out_fps = src_fps * Fraction(args.interp_factor).limit_denominator(1000)
    else:
        out_fps = src_fps

    fp16 = not args.fp32
    interp = None
    if out_fps != src_fps:
        wpath = os.path.join(MODEL_DIR, args.rife_model + ".pkl")
        interp = RifeInterpolator(wpath, device=args.device, fp16=fp16,
                                  scale=args.rife_scale)
        print(f"[interp] {args.rife_model} ({interp.arch}) "
              f"-> {float(out_fps):.3f} fps", file=sys.stderr)

    up = None
    scale = 1
    if not args.no_upscale:
        wpath = os.path.join(MODEL_DIR, args.upscale_model + ".pth")
        up = Upscaler(wpath, device=args.device, fp16=fp16, tile=args.tile)
        scale = up.scale
        print(f"[upscl ] {args.upscale_model} x{scale} -> {w*scale}x{h*scale}",
              file=sys.stderr)

    # Resolve the final size. By default there is no resampling at all - the
    # model's native 4x output is what gets encoded. When a smaller size *is*
    # requested, the resample happens on the GPU rather than in an ffmpeg
    # filter, so the pipe only carries the final resolution: asking for 2x
    # moves a quarter of the bytes of the raw 4x output.
    model_w, model_h = w * scale, h * scale
    if args.out_height:
        final_h = args.out_height
        final_w = round(w * final_h / h)
    elif args.upscale_factor:
        final_h = round(h * args.upscale_factor)
        final_w = round(w * args.upscale_factor)
    else:
        final_h, final_w = model_h, model_w
    final_w -= final_w % 2   # yuv420p needs even dimensions
    final_h -= final_h % 2

    resize_to = (final_h, final_w) if (final_h, final_w) != (model_h, model_w) else None
    if resize_to:
        print(f"[resize] {model_w}x{model_h} -> {final_w}x{final_h}", file=sys.stderr)

    dev = torch.device(args.device)
    enc = encoder(args.output, final_w, final_h, out_fps, args.input, args)

    def to_tensor(arr):
        t = torch.from_numpy(arr).to(dev)
        return t.permute(2, 0, 1)[None].float().div_(255.0)

    written = 0

    def emit(t):
        nonlocal written
        if up is not None:
            t = up.upscale(t)
        if resize_to is not None:
            t = F.interpolate(t, size=resize_to, mode="bicubic",
                              antialias=True, align_corners=False).clamp(0, 1)
        arr = (t[0].permute(1, 2, 0) * 255.0).round().clamp(0, 255).to(torch.uint8)
        enc.stdin.write(arr.cpu().numpy().tobytes())
        written += 1
        if written % 50 == 0:
            print(f"\r[write ] {written} frames", end="", file=sys.stderr, flush=True)

    # Output frame k samples the source timeline at position k * src_fps / out_fps.
    # This handles non-integer ratios such as 24 -> 60 in one pass, which is why
    # RIFE v4.x (arbitrary timestep) is used rather than recursive halving.
    ratio = Fraction(src_fps, out_fps)
    out_idx = 0
    prev = None

    try:
        for i, frame in enumerate(decoder(args.input, w, h)):
            cur = to_tensor(frame)
            if prev is None:
                prev = cur
                continue

            if interp is None:
                emit(prev)
            else:
                timesteps, exacts = [], []
                while out_idx * ratio < i:
                    frac = float(out_idx * ratio) - (i - 1)
                    if frac < 1e-6:
                        exacts.append((len(timesteps), None))
                    else:
                        exacts.append((len(timesteps), frac))
                        timesteps.append(frac)
                    out_idx += 1

                made = interp.interpolate(prev, cur, timesteps) if timesteps else []
                k = 0
                for _, frac in exacts:
                    if frac is None:
                        emit(prev)
                    else:
                        emit(made[k])
                        k += 1
            prev = cur

        if prev is not None:
            emit(prev)
    except BrokenPipeError:
        # The encoder died; its own error is already on stderr above, and is
        # far more useful than a traceback from this end of the pipe.
        pass
    finally:
        try:
            enc.stdin.close()
        except BrokenPipeError:
            pass
        rc = enc.wait()

    if rc != 0 or written == 0:
        sys.exit(
            f"\n[error ] encoder exited with status {rc} after {written} frames. "
            f"The ffmpeg message above says why.\n"
            f"         If it mentions the nvenc API version or driver version, "
            f"this ffmpeg build is newer than the GPU driver; rerun with "
            f"--codec libx264."
        )

    print(f"\r[done  ] wrote {written} frames to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
