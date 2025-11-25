"""Data utils for video quality assessment models."""

# Adapted from https://github.com/VQAssessment/DOVER/blob/master/dover/datasets/dover_datasets.py.

from functools import lru_cache
from typing import Any

import decord
import torch
import torchvision
from decord import VideoReader

from utils import tmp_files

# pyright: reportMissingImports=false
# pyright: reportUnboundVariable=false



def prepare_input(
    video_file: bytes,
) -> Any:
    decord.bridge.set_bridge("torch")
    with tmp_files.make_named_temporary_file(suffix=".mp4") as tmp_file:
        tmp_file.write_bytes(video_file)
        return VideoReader(tmp_file.as_posix())


@lru_cache
def get_resize_function(size_h: int, size_w: int, target_ratio: float = 1, random_crop: bool = False) -> Any:
    if random_crop:
        return torchvision.transforms.RandomResizedCrop((size_h, size_w), scale=(0.40, 1.0))
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w), antialias=False)


def get_resized_video(
    video: torch.Tensor,
    size_h: int = 224,
    size_w: int = 224,
    random_crop: bool = False,
    arp: bool = False,
    **kwargs,
) -> torch.Tensor:
    video = video.permute(1, 0, 2, 3)
    resize_opt = get_resize_function(size_h, size_w, video.shape[-2] / video.shape[-1] if arp else 1, random_crop)
    video = resize_opt(video).permute(1, 0, 2, 3)
    return video


def get_spatial_fragments(
    video: torch.Tensor,
    fragments_h: int = 7,
    fragments_w: int = 7,
    fsize_h: int = 32,
    fsize_w: int = 32,
    aligned: int = 32,
    nfrags: int = 1,
    random: bool = False,
    random_upsample: bool = False,
    fallback_type: str = "upsample",
    upsample: int = -1,
    **kwargs,
) -> torch.Tensor:
    if upsample > 0:
        old_h, old_w = video.shape[-2], video.shape[-1]
        if old_h >= old_w:
            w = upsample
            h = int(upsample * old_h / old_w)
        else:
            h = upsample
            w = int(upsample * old_w / old_h)

        video = get_resized_video(video, h, w)
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    # video: [C,T,H,W]
    # situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        ovideo = video
        video = torch.nn.functional.interpolate(video / 255.0, scale_factor=1 / ratio, mode="bilinear")
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    # make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor([min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)])
    wgrids = torch.LongTensor([min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)])
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned))
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[:, t_s:t_e, h_so:h_eo, w_so:w_eo]
    return target_video
