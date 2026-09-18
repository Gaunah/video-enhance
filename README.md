# video-enhance

Frame interpolation (RIFE v4.25) plus upscaling (Real-ESRGAN) for low-resolution,
low-frame-rate clips. Single container, runs anywhere with an NVIDIA GPU.

## Quick start

```bash
docker build -t video-enhance .

docker run --rm --gpus all -v "$PWD/clips:/data" video-enhance \
    /data/in.mp4 /data/out.mp4 --fps 60
```

That reads `in.mp4`, interpolates to 60 fps, upscales 4x, and writes `out.mp4`
with the original audio.

## Common invocations

```bash
# 480p/12fps source -> 1080p/60fps (4x model, then resample down to 1080p)
enhance.py in.mp4 out.mp4 --fps 60 --out-height 1080

# Upscale only, leave timing alone
enhance.py in.mp4 out.mp4 --no-interp

# Smooth motion only, keep the resolution
enhance.py in.mp4 out.mp4 --interp-factor 2 --no-upscale

# Clean source, want maximum detail (slower)
enhance.py in.mp4 out.mp4 --fps 60 --upscale-model RealESRGAN_x4plus

# Try a different interpolation model
enhance.py in.mp4 out.mp4 --fps 60 --rife-model flownet_v4.6

# Out of VRAM
enhance.py in.mp4 out.mp4 --fps 60 --tile 512
```

## Choosing an upscale model

| Model | Arch | Best for |
|---|---|---|
| `realesr-general-x4v3` (default) | compact, 1.2M | noisy or compressed sources; fast |
| `RealESRGAN_x4plus` | RRDBNet-23, 16.7M | most detail on clean footage, roughly 10x slower |

Both are 4x and both ship in the image. To land on a 2x result, upscale 4x and
come back down with `--out-height`; that consistently looks better than a
native 2x model.

## Choosing a RIFE version

| Model | Arch | Notes |
|---|---|---|
| `flownet_v4.25` (default) | 5 blocks + encoder | the author's recommended default for most scenes |
| `flownet_v4.26` | same as 4.25 | newest release (2024.09) |
| `flownet_v4.6` | 4 blocks, no encoder | the old conservative default; fewest surprises |
| `flownet_v4.26.heavy` | 16ch encoder | wider encoder, slower |

Only 4.25 is baked into the image. Get the rest with
`python download_models.py --all`.

Versions 4.7 through 4.24 are deliberately not supported: each changed the block
layout, and carrying five more architectures for models nobody recommends is not
worth it. Loading one fails with a clear message rather than silently producing
garbage.

A caution on picking between these. RIFE's author notes that "improving the PSNR
index is not consistent with subjective perception", and that is easy to confirm:
on a synthetic pan-with-occlusion sequence, v4.6 scored 26.2 dB against v4.25's
21.0 dB, which contradicts every subjective report and the author's own
recommendation. Reconstruction error rewards the conservative, blurry answer.
Judge these on your own footage by eye; do not trust a PSNR number, including
that one.

## Deploying to RunPod

**As a Pod.** Push the image somewhere RunPod can pull it:

```bash
docker build -t ghcr.io/you/video-enhance:latest .
docker push ghcr.io/you/video-enhance:latest
```

Create a Pod with that image, attach a network volume at `/workspace`, and set
the container start command to `sleep infinity` so the pod stays up. Then from
the web terminal:

```bash
cd /app && python enhance.py /workspace/in.mp4 /workspace/out.mp4 --fps 60
```

Upload and download clips over the pod's SSH/SCP endpoint, or via the volume.

**Without building an image.** Start any RunPod PyTorch template, then:

```bash
git clone <this repo> /workspace/video-enhance && cd /workspace/video-enhance
pip install -r requirements.txt
python download_models.py
python enhance.py /workspace/in.mp4 /workspace/out.mp4 --fps 60
```

The bundled ffmpeg is the only thing you lose that way; install one with NVENC
or pass `--codec libx264`.

**GPU choice.** This workload is compute-bound, not memory-bound, so consumer
cards are the value pick: a 4090 or L40S will beat an A100 per dollar here.
VRAM needed is modest — a 480p source upscaled 4x fits comfortably in 12 GB
without tiling.

**Serverless.** Possible, but you have to solve file transfer yourself
(presigned S3 URLs in and out). For a handful of clips a Pod is less work.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--fps` | source | Target rate. Accepts `60` or `60000/1001` |
| `--interp-factor` | – | Alternative to `--fps`, e.g. `2` |
| `--no-interp` / `--no-upscale` | – | Skip either stage |
| `--rife-model` | `flownet_v4.25` | See table above |
| `--upscale-model` | `realesr-general-x4v3` | See table above |
| `--out-height` | – | Lanczos resample after upscaling |
| `--tile` | `0` | Tile size for the upscaler; try `512` or `256` on OOM |
| `--rife-scale` | `1.0` | `0.5` for 4K input, `2.0` for very fast motion at low res |
| `--codec` | `auto` | Prefers `hevc_nvenc`, falls back to `libx264` |
| `--quality` | `19` | CRF for x264/x265, CQ for NVENC. Lower is better |
| `--fp32` | off | Disable half precision if you see artefacts |

## How it works

Frames are streamed `ffmpeg -> stdout pipe -> torch -> stdin pipe -> ffmpeg`.
Nothing is written as an image sequence, which matters: a ten second 480p clip
going to 4x/60fps would otherwise be several GB of PNGs.

Interpolation runs **before** upscaling. Doing it at source resolution is ~16x
cheaper, and the quality cost is small because RIFE's optical flow does not gain
much from detail that was hallucinated rather than measured.

Output frame *k* samples the source timeline at `k * src_fps / out_fps`. RIFE
v4.x accepts an arbitrary timestep, so awkward ratios like 24 to 60 fps are a
single pass rather than recursive doubling followed by frame dropping.

`rife.py` and `esrgan.py` are self-contained reimplementations of the inference
path. In particular `basicsr` is deliberately not a dependency: it imports
`torchvision.transforms.functional_tensor`, which no longer exists, and pulls in
a training stack this does not need. Checkpoints are key-matched exactly, so a
mismatched file fails loudly instead of producing garbage.

Both RIFE architectures were verified bit-exact (max abs difference 0.0) against
the reference implementations in [vs-rife](https://github.com/HolyWu/vs-rife),
which is where the weights come from. The architecture follows Practical-RIFE
(hzwer) and vs-rife (HolyWu), both MIT licensed. The `teacher` (distillation) and
`caltime` (timestep estimator) submodules in the checkpoints are training-only
and are dropped on load.

## Limitations and what to reach for next

RIFE struggles with occlusion boundaries, motion blur, and scene cuts. There is
no shot-change detection here, so a hard cut will produce one smeared frame; if
your clips have cuts, split on them first (`ffmpeg -f segment` with a scene
filter) and process each shot separately.

Real-ESRGAN upscales each frame independently, so fine texture can shimmer
slightly between frames. If that bothers you, the next steps up are:

- **TensorRT**, for roughly 3-5x the throughput. `styler00dollar/VSGAN-tensorrt-docker`
  packages VapourSynth, RIFE and Real-ESRGAN with TRT engines already wired up.
- **Temporally aware restoration** — SeedVR2, STAR, or Upscale-A-Video — which
  use neighbouring frames and largely remove the shimmer. Much heavier, and they
  invent detail more aggressively, which is a problem for archival footage.

Model weights are downloaded from GitHub release assets, so no Google Drive
cookie handling is needed.
