# video-enhance

Frame interpolation (RIFE v4.25) plus upscaling (Real-ESRGAN) for low-resolution,
low-frame-rate clips. Single container, runs anywhere with an NVIDIA GPU.

## Quick start

```bash
docker build -t video-enhance .

docker run --rm --gpus all -v "$PWD/clips:/data" video-enhance \
    /data/in.mp4 /data/out.mp4
```

With no flags that doubles the frame rate and upscales 4x, the model's native
scale, keeping the original audio. Both defaults are overridable per run.

## Common invocations

```bash
# Pin an exact frame rate instead of a multiplier
enhance.py in.mp4 out.mp4 --fps 60

# Pin an exact height instead of taking the model's native 4x
enhance.py in.mp4 out.mp4 --out-height 1080

# 2x instead of 4x (runs the 4x model, resamples down)
enhance.py in.mp4 out.mp4 --upscale-factor 2

# Upscale only, leave timing alone
enhance.py in.mp4 out.mp4 --no-interp

# Smooth motion only, keep the resolution
enhance.py in.mp4 out.mp4 --no-upscale

# Clean source, want maximum detail (slower)
enhance.py in.mp4 out.mp4 --upscale-model RealESRGAN_x4plus

# Try a different interpolation model
enhance.py in.mp4 out.mp4 --rife-model flownet_v4.6

# Out of VRAM
enhance.py in.mp4 out.mp4 --tile 512
```

## Choosing an upscale model

| Model | Arch | Best for |
|---|---|---|
| `realesr-general-x4v3` (default) | compact, 1.2M | noisy or compressed sources; fast |
| `RealESRGAN_x4plus` | RRDBNet-23, 16.7M | most detail on clean footage, roughly 10x slower |

Both are 4x and both ship in the image, and 4x is what you get by default with
no resampling. If you want less, `--upscale-factor 2` or `--out-height` runs the
4x model and resamples down, which still looks better than a native 2x model
would.

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
cd /app && python enhance.py /workspace/in.mp4 /workspace/out.mp4
```

Upload and download clips over the pod's SSH/SCP endpoint, or via the volume.

**Without building an image.** Start any RunPod PyTorch template, then:

```bash
git clone <this repo> /workspace/video-enhance && cd /workspace/video-enhance
pip install -r requirements.txt
python download_models.py
python enhance.py /workspace/in.mp4 /workspace/out.mp4
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
| `--interp-factor` | `2` | Frame rate multiplier |
| `--fps` | – | Exact target rate, overrides the multiplier. `60` or `60000/1001` |
| `--no-interp` / `--no-upscale` | – | Skip either stage |
| `--rife-model` | `flownet_v4.25` | See table above |
| `--upscale-model` | `realesr-general-x4v3` | See table above |
| `--upscale-factor` | model native (4x) | Output scale relative to source |
| `--out-height` | – | Exact output height, overrides the factor |
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

By default nothing is resampled: the model's native 4x output is what gets
encoded. When you do ask for something smaller, the resample happens on the GPU
rather than in an ffmpeg filter, so the pipe only carries the final resolution.
Asking for 2x from a 1080p source moves ~25 MB per frame instead of ~99 MB. The
cost is that torch has no Lanczos kernel, so it uses antialiased bicubic, which
measures marginally softer than `flags=lanczos` (about 7% less edge energy)
while landing closer to a reference Lanczos in PSNR. For a 4x pipe saving on the
downscale path that is a good trade.

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

## Troubleshooting

**`Driver does not support the required nvenc API version. Required: 13.1 Found: 13.0`**

The ffmpeg build is newer than the GPU driver, not broken. The static build in
this image tracks ffmpeg master and is compiled against whatever NVENC SDK was
current; an older host driver exposes an older API and refuses to open the
encoder. Either:

- rerun with `--codec libx264` (CPU encode, slow at 4K and above), or
- pick a RunPod template with a newer driver, or
- swap the static ffmpeg in the Dockerfile for the distro package
  (`apt-get install -y ffmpeg`), which is built against older NVENC headers and
  works with older drivers.

`--codec auto` now test-encodes a frame before committing to NVENC, so it falls
back to libx264 on its own. Only an explicitly requested `--codec hevc_nvenc`
will still fail, which is deliberate: silently overriding what you asked for is
worse than stopping.

**CUDA out of memory.** Add `--tile 512`, or `--tile 256` if that is still too
big. Tiling is verified to match whole-frame output, so it costs nothing but
time.

**Frame rate goes down, not up.** `--fps` is an absolute target, not a
multiplier, and it overrides `--interp-factor`. Passing `--fps 30` to a 60 fps
source *halves* the rate by dropping frames; RIFE is not invoked at all when the
ratio is an exact integer. Leave `--fps` off to get the default 2x.

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
