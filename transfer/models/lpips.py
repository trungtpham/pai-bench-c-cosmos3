"""LPIPS model from https://arxiv.org/abs/1801.03924

The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, 2018, Zhang et al.
"""

from typing import Final, Union

# pyright: reportMissingImports=false
# pyright: reportUnboundVariable=false
import lpips
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from utils import model_utils

_DEFAULT_WEIGHTS_NAME: Final = "lpips"


class LPIPS(model_utils.ModelInterface):
    def __init__(self, net: str) -> None:
        valid = ("alex", "vgg", "vgg16", "squeeze")
        if net not in valid:
            raise ValueError(f"net {net} not valid. Available options are {valid}.")
        self.net = net

    @property
    def conda_env_name(self) -> str:
        return "paibench-transfer"

    def setup(self) -> None:
        self._model = lpips.LPIPS(net=self.net)
        if torch.cuda.is_available():
            self.device = "cuda"
            self._model.to(self.device)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.device = "cpu"

    @property
    def weights_names(self) -> list[str]:
        return [_DEFAULT_WEIGHTS_NAME]

    @torch.inference_mode()
    def __call__(
        self,
        images_1: Union[torch.Tensor, npt.NDArray[np.uint8]],
        images_2: Union[torch.Tensor, npt.NDArray[np.uint8]],
        batch_size: int = 10,
    ) -> torch.Tensor:
        n = len(images_1)
        results = []
        for i in range(0, n, batch_size):
            batch_1 = images_1[i : i + batch_size]
            batch_2 = images_2[i : i + batch_size]
            res = self._model(
                self.preprocess(batch_1),
                self.preprocess(batch_2),
            )
            results.append(res)
        return torch.cat(results)

    def preprocess(
        self, images: Union[torch.Tensor, npt.NDArray[np.uint8]]
    ) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            if images.dtype != np.uint8:
                raise ValueError(f"Expected dtype uint8 but got {images.dtype}")
            images_pt = torch.from_numpy(images)
        else:
            if images.dtype != torch.uint8:
                raise ValueError(f"Expected dtype uint8 but got {images.dtype}")
            images_pt = images

        if images_pt.ndim != 4:
            raise ValueError(f"Expected rank 4 tensor, got {images.ndim}")
        if images_pt.shape[-1] != 3:
            raise ValueError(
                f"Expected shape NHWC, where C is 3. Got {images_pt.shape}."
            )
        # move to device
        images_pt = images_pt.to(device=self.device, dtype=torch.float32)

        # normalize between -1 and 1
        images_norm = (images_pt / 127.5) - 1.0

        # permute from NHWC to NCHW
        return images_norm.permute((0, 3, 1, 2))
