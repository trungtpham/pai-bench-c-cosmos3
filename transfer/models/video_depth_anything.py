import io
import json
from dataclasses import dataclass
from typing import Literal, Optional

# pyright: reportMissingImports=false
# pyright: reportUnboundVariable=false
import decord
import imageio
import numpy as np
import torch

from utils import model_utils, tmp_files

from .video_depth_anything_model import video_depth

WEIGHTS_FOLDER = "video_depth_anything"
WEIGHTS_NAME_SMALL = "video_depth_anything_vits.pth"
WEIGHTS_NAME_LARGE = "video_depth_anything_vitl.pth"


@dataclass
class VideoDepthResult:
    """Result from video depth generation containing normalized videos and depth statistics."""

    per_video_bytes: bytes  # Video with per-video normalization
    per_frame_bytes: bytes  # Video with per-frame normalization
    min_max_bytes: bytes  # JSON bytes containing depth min/max values


def ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def save_video(
    frames: np.ndarray,
    output_video_path: str,
    fps: float,
    normalization_mode: Literal["per_video", "per_frame"] = "per_video",
) -> tuple[float, float]:
    with imageio.get_writer(
        output_video_path,
        fps=fps,
        macro_block_size=1,
        codec="libx264",
        ffmpeg_params=["-crf", "18"],
    ) as writer:
        d_min, d_max = frames.min(), frames.max()
        if normalization_mode == "per_video":
            for i in range(frames.shape[0]):
                depth = frames[i]
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                writer.append_data(depth_norm)  # type: ignore
        elif normalization_mode == "per_frame":
            for i in range(frames.shape[0]):
                depth = frames[i]
                # Normalize per frame
                f_d_min, f_d_max = depth.min(), depth.max()
                depth_norm = ((depth - f_d_min) / (f_d_max - f_d_min) * 255).astype(
                    np.uint8
                )
                writer.append_data(depth_norm)  # type: ignore
        else:
            raise ValueError(f"Invalid normalization mode: {normalization_mode}")
    return (d_min, d_max)


class VideoDepthAnything(model_utils.ModelInterface):
    model_configs = {  # noqa: RUF012
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    def __init__(self, encoder: str = "vits") -> None:
        self.encoder = encoder
        print(f"Initializing {self.encoder} encoder")

    @property
    def weights_names(self) -> list[str]:
        return [WEIGHTS_FOLDER]

    @property
    def conda_env_name(self) -> str:
        return "paibench-transfer"

    def setup(self) -> None:
        self.download_weights()
        print(f"Loading weights for {self.encoder} encoder")
        weight_name = {
            "vits": WEIGHTS_NAME_SMALL,
            "vitl": WEIGHTS_NAME_LARGE,
        }[self.encoder]
        local_path = (
            model_utils.get_local_dir_for_weights_name(WEIGHTS_FOLDER) / weight_name
        )
        self.model = video_depth.VideoDepthAnything(**self.model_configs[self.encoder])
        self.model.load_state_dict(torch.load(local_path, map_location="cpu"), strict=True)  # type: ignore
        self.model = self.model.to("cuda").eval()  # type: ignore

    @staticmethod
    def _extract_frames(
        video_bytes: bytes,
        max_res: Optional[int] = None,
        target_fps: Optional[float] = None,
        max_len: Optional[int] = None,
    ) -> tuple[np.ndarray, float]:
        buffer = io.BytesIO(video_bytes)
        vid = decord.VideoReader(buffer)
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        buffer.seek(0)
        vid = decord.VideoReader(buffer, width=width, height=height)

        fps = vid.get_avg_fps() if target_fps is None else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if max_len and max_len < len(frames_idx):
            frames_idx = frames_idx[:max_len]
        return vid.get_batch(frames_idx).asnumpy(), fps

    def generate(
        self,
        video: np.ndarray,
    ) -> np.ndarray:
        assert video.ndim == 4, "Video tensor should have shape (T, H, W, 3)"
        assert video.dtype == np.uint8, "Video tensor should be uint8"
        depths, _ = self.model.infer_video_depth(video, 30, device="cuda")  # type: ignore
        return depths

    def generate_video(
        self,
        video_bytes: bytes,
        max_res: Optional[int] = None,
        target_fps: Optional[float] = None,
        max_len: Optional[int] = None,
    ) -> VideoDepthResult:
        frames, target_fps = self._extract_frames(
            video_bytes,
            max_res=max_res,
            target_fps=target_fps,
            max_len=max_len,
        )
        depths = self.generate(frames)

        per_video_bytes = None
        with tmp_files.make_named_temporary_file(suffix=".mp4") as tmp_file_per_video:
            depth_minmix = save_video(
                depths,
                tmp_file_per_video.as_posix(),
                fps=target_fps,
                normalization_mode="per_video",
            )
            per_video_bytes = tmp_file_per_video.read_bytes()

        per_frame_bytes = None
        with tmp_files.make_named_temporary_file(suffix=".mp4") as tmp_file_per_frame:
            _ = save_video(
                depths,
                tmp_file_per_frame.as_posix(),
                fps=target_fps,
                normalization_mode="per_frame",
            )
            per_frame_bytes = tmp_file_per_frame.read_bytes()

        min_max_bytes = None
        with tmp_files.make_named_temporary_file(suffix=".json") as tmp_file_min_max:
            depth_minmax = json.dumps(
                {
                    "depth_min": float(depth_minmix[0]),
                    "depth_max": float(depth_minmix[1]),
                }
            )
            tmp_file_min_max.write_text(depth_minmax)
            min_max_bytes = tmp_file_min_max.read_bytes()

        return VideoDepthResult(
            per_video_bytes=per_video_bytes,
            per_frame_bytes=per_frame_bytes,
            min_max_bytes=min_max_bytes,
        )
