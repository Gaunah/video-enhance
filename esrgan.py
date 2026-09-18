"""
Real-ESRGAN inference, vendored.

basicsr is deliberately not a dependency: it pins old torchvision internals
(`torchvision.transforms.functional_tensor`) that were removed, and it drags in
a training stack we do not need. The two architectures below are the only ones
the official Real-ESRGAN weights use, and the constructor arguments are
inferred from the checkpoint so you do not have to pass them per model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# SRVGGNetCompact - used by realesr-general-x4v3
# --------------------------------------------------------------------------
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        out = out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out


# --------------------------------------------------------------------------
# RRDBNet - used by RealESRGAN_x4plus
# --------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# --------------------------------------------------------------------------
def build_from_state_dict(state):
    """Infer the architecture and its hyper-parameters from a checkpoint."""
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]

    if "body.0.weight" in state:  # SRVGGNetCompact
        num_feat = state["body.0.weight"].shape[0]
        # body layout: conv, prelu, (conv, prelu) * num_conv, conv
        last_idx = max(
            int(k.split(".")[1]) for k in state if k.startswith("body.") and "weight" in k
        )
        num_conv = (last_idx - 2) // 2
        out_ch_sq = state[f"body.{last_idx}.weight"].shape[0]
        upscale = int(round((out_ch_sq / 3) ** 0.5))
        model = SRVGGNetCompact(
            num_feat=num_feat, num_conv=num_conv, upscale=upscale
        )
        return model, state, upscale

    if "conv_first.weight" in state:  # RRDBNet
        num_feat = state["conv_first.weight"].shape[0]
        in_ch = state["conv_first.weight"].shape[1]
        scale = {3: 4, 12: 2, 48: 1}.get(in_ch, 4)
        num_block = (
            max(int(k.split(".")[1]) for k in state if k.startswith("body.")) + 1
        )
        num_grow_ch = state["body.0.rdb1.conv1.weight"].shape[0]
        model = RRDBNet(
            scale=scale, num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch
        )
        return model, state, scale

    raise ValueError("Unrecognised Real-ESRGAN checkpoint layout")


class Upscaler:
    """Tiled Real-ESRGAN inference so large frames fit in VRAM."""

    def __init__(self, weights_path, device="cuda", fp16=True, tile=0, tile_pad=16):
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        self.dtype = torch.half if self.fp16 else torch.float32
        self.tile = tile
        self.tile_pad = tile_pad

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        model, state, scale = build_from_state_dict(ckpt)
        model.load_state_dict(state, strict=True)
        self.scale = scale
        self.model = model.eval().to(self.device, dtype=self.dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def _forward_whole(self, x):
        return self.model(x)

    @torch.inference_mode()
    def _forward_tiled(self, x):
        b, c, h, w = x.shape
        out = x.new_zeros((b, c, h * self.scale, w * self.scale))
        n_h = (h + self.tile - 1) // self.tile
        n_w = (w + self.tile - 1) // self.tile
        for i in range(n_h):
            for j in range(n_w):
                y0, y1 = i * self.tile, min((i + 1) * self.tile, h)
                x0, x1 = j * self.tile, min((j + 1) * self.tile, w)
                py0, py1 = max(y0 - self.tile_pad, 0), min(y1 + self.tile_pad, h)
                px0, px1 = max(x0 - self.tile_pad, 0), min(x1 + self.tile_pad, w)
                tile_out = self.model(x[:, :, py0:py1, px0:px1])
                cy0 = (y0 - py0) * self.scale
                cx0 = (x0 - px0) * self.scale
                out[:, :, y0 * self.scale : y1 * self.scale, x0 * self.scale : x1 * self.scale] = (
                    tile_out[
                        :,
                        :,
                        cy0 : cy0 + (y1 - y0) * self.scale,
                        cx0 : cx0 + (x1 - x0) * self.scale,
                    ]
                )
        return out

    @torch.inference_mode()
    def upscale(self, frame):
        """frame: float tensor [1,3,H,W] in [0,1]. Returns [1,3,H*s,W*s]."""
        x = frame.to(self.device, dtype=self.dtype)
        # RRDBNet with pixel_unshuffle needs even dimensions
        pad_h = (-x.shape[2]) % 8
        pad_w = (-x.shape[3]) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        out = self._forward_tiled(x) if self.tile else self._forward_whole(x)

        if pad_h or pad_w:
            out = out[
                :, :, : out.shape[2] - pad_h * self.scale, : out.shape[3] - pad_w * self.scale
            ]
        return out.float().clamp(0, 1)
