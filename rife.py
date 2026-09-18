"""
RIFE frame interpolation - self-contained inference.

Two architecture families are supported and auto-detected from the checkpoint:

  v4.6            4 IFBlocks, no feature encoder. The long-standing default;
                  conservative and rarely produces gross artefacts.
  v4.25 / v4.26   5 IFBlocks plus a feature encoder, with features passed
                  between blocks. Better on large motion. v4.26.heavy widens
                  the encoder from 4 to 16 channels.

Architecture follows Practical-RIFE (hzwer) and the vs-rife port (HolyWu),
both MIT licensed. Only the inference path is reimplemented here so that
neither repository is a runtime dependency.

Training-only submodules in the checkpoints (`teacher`, the distillation
branch, and `caltime`, a timestep estimator) are dropped on load.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_GRID_CACHE = {}


def warp(tensor, flow):
    """Backward-warp `tensor` by `flow`, which is in pixels."""
    key = (str(flow.device), str(flow.dtype), str(flow.shape))
    if key not in _GRID_CACHE:
        h, w = flow.shape[2], flow.shape[3]
        xs = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype)
        ys = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype)
        gx = xs.view(1, 1, 1, w).expand(flow.shape[0], -1, h, -1)
        gy = ys.view(1, 1, h, 1).expand(flow.shape[0], -1, -1, w)
        _GRID_CACHE[key] = torch.cat([gx, gy], 1)
    grid = _GRID_CACHE[key]

    norm = torch.cat(
        [
            flow[:, 0:1] / ((tensor.shape[3] - 1.0) / 2.0),
            flow[:, 1:2] / ((tensor.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    g = (grid + norm).permute(0, 2, 3, 1)
    return F.grid_sample(
        tensor, g, mode="bilinear", padding_mode="border", align_corners=True
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """out_ch is 6 for v4.6 (flow 4, mask 1, spare 1) and 13 for v4.25+
    (flow 4, mask 1, feature 8 carried into the next block)."""

    def __init__(self, in_planes, c=64, out_ch=6):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * out_ch, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class Head(nn.Module):
    """Feature encoder introduced in the v4.7+ line."""

    def __init__(self, c=16, out_ch=4):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, c, 3, 2, 1)
        self.cnn1 = nn.Conv2d(c, c, 3, 1, 1)
        self.cnn2 = nn.Conv2d(c, c, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(c, out_ch, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class IFNetV46(nn.Module):
    DEFAULT_SCALES = (8, 4, 2, 1)

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7, c=192, out_ch=6)
        self.block1 = IFBlock(8 + 4, c=128, out_ch=6)
        self.block2 = IFBlock(8 + 4, c=96, out_ch=6)
        self.block3 = IFBlock(8 + 4, c=64, out_ch=6)

    def forward(self, img0, img1, timestep, scale_list):
        if not torch.is_tensor(timestep):
            timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        flow = mask = None
        w0, w1 = img0, img1
        for i, block in enumerate([self.block0, self.block1, self.block2, self.block3]):
            if flow is None:
                flow, mask, _ = block(
                    torch.cat((img0, img1, timestep), 1), None, scale=scale_list[i]
                )
            else:
                fd, m0, _ = block(
                    torch.cat((w0, w1, timestep, mask), 1), flow, scale=scale_list[i]
                )
                flow = flow + fd
                mask = mask + m0  # v4.6 accumulates the mask
            w0 = warp(img0, flow[:, :2])
            w1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return w0 * mask + w1 * (1 - mask)


class IFNetV425(nn.Module):
    """Covers v4.25, v4.26 (encode_ch=4) and v4.26.heavy (encode_ch=16)."""

    DEFAULT_SCALES = (16, 8, 4, 2, 1)

    def __init__(self, encode_ch=4):
        super().__init__()
        e = encode_ch
        self.block0 = IFBlock(7 + 2 * e, c=192, out_ch=13)
        self.block1 = IFBlock(8 + 4 + 8 + 2 * e, c=128, out_ch=13)
        self.block2 = IFBlock(8 + 4 + 8 + 2 * e, c=96, out_ch=13)
        self.block3 = IFBlock(8 + 4 + 8 + 2 * e, c=64, out_ch=13)
        self.block4 = IFBlock(8 + 4 + 8 + 2 * e, c=32, out_ch=13)
        self.encode = Head(16, e)

    def forward(self, img0, img1, timestep, scale_list):
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        if not torch.is_tensor(timestep):
            timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        f0 = self.encode(img0)
        f1 = self.encode(img1)

        flow = mask = feat = None
        w0, w1 = img0, img1
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i, block in enumerate(blocks):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=scale_list[i]
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block(
                    torch.cat((w0, w1, wf0, wf1, timestep, mask, feat), 1),
                    flow,
                    scale=scale_list[i],
                )
                flow = flow + fd
                mask = m0  # v4.25+ replaces the mask rather than accumulating
            w0 = warp(img0, flow[:, :2])
            w1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return w0 * mask + w1 * (1 - mask)


def build_from_state_dict(state):
    """Pick the architecture from the checkpoint's own shape signature."""
    if "block4.conv0.0.0.weight" in state:
        encode_ch = state["encode.cnn3.weight"].shape[1]
        return IFNetV425(encode_ch=encode_ch)
    if "encode.cnn0.weight" in state:
        raise ValueError(
            "This looks like RIFE v4.7-v4.24, which uses a different block "
            "layout. Supported: v4.6, v4.25, v4.26, v4.26.heavy."
        )
    return IFNetV46()


class RifeInterpolator:
    ALIGN = 64

    def __init__(self, weights_path, device="cuda", fp16=True, scale=1.0):
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.dtype = torch.half if self.fp16 else torch.float32

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        state = {
            k: v for k, v in state.items() if not k.startswith(("teacher.", "caltime."))
        }

        self.net = build_from_state_dict(state)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match architecture; "
                f"missing={missing[:4]} unexpected={unexpected[:4]}"
            )

        self.net.eval().to(self.device, dtype=self.dtype)
        for p in self.net.parameters():
            p.requires_grad_(False)

        # scale < 1 helps with very fast motion, scale > 1 is cheaper on 4K
        self.scale_list = [s / scale for s in self.net.DEFAULT_SCALES]
        self.arch = type(self.net).__name__

    def _pad(self, x):
        h, w = x.shape[2], x.shape[3]
        ph = ((h - 1) // self.ALIGN + 1) * self.ALIGN
        pw = ((w - 1) // self.ALIGN + 1) * self.ALIGN
        return F.pad(x, (0, pw - w, 0, ph - h), mode="replicate"), h, w

    @torch.inference_mode()
    def interpolate(self, frame0, frame1, timesteps):
        """frame0/frame1: float tensors [1,3,H,W] in [0,1] on self.device.
        Returns one frame per t in `timesteps`."""
        a = frame0.to(self.dtype)
        b = frame1.to(self.dtype)
        a, h, w = self._pad(a)
        b, _, _ = self._pad(b)

        out = []
        for t in timesteps:
            res = self.net(a, b, float(t), self.scale_list)
            out.append(res[:, :, :h, :w].float().clamp(0, 1))
        return out
