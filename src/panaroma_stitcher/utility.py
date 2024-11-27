"""Utility functions/classes"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any, Generator, Optional, Tuple

import logging
import torch
import cv2
import kornia as krn
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class ImageLoader:
    """Load/Save images from/to directories"""

    image_dir: Path
    resize_shape: Optional[Tuple[int, int]] = field(default=None)
    device: str = field(default="cpu")
    images: List[Any] = field(init=False)

    def __post_init__(self) -> None:
        """Check the cuda availability and other post-processing requirements"""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.info("%s is not available.", self.device)
            self.device = "cpu"

    def _list_images(self) -> Generator[Path, None, None]:
        """List images in directory"""
        return self.image_dir.glob("*.jpg")

    def opencv_load_images(self) -> None:
        """Load images for opencv stitcher from a directory"""
        files = self._list_images()
        if not self.resize_shape:
            self.images = [cv2.imread(str(filename)) for filename in files]
        else:
            self.images = [
                cv2.resize(cv2.imread(str(filename)), self.resize_shape)
                for filename in files
            ]
        logger.info(
            "Number of loaded images from %s is: %s",
            str(self.image_dir),
            len(self.images),
        )

    def kornia_load_images(self) -> None:
        """Load images for kornia stitcher from a directory"""
        files = list(self._list_images())
        if not self.resize_shape:
            self.images = [
                krn.io.load_image(
                    str(filename),
                    desired_type=krn.io.ImageLoadType.RGB32,
                    device=self.device,
                )[None, ...]
                for filename in files
            ]
        else:
            self.images = [
                krn.geometry.resize(
                    krn.io.load_image(
                        str(filename),
                        desired_type=krn.io.ImageLoadType.RGB32,
                        device=self.device,
                    )[None, ...],
                    self.resize_shape,
                )
                for filename in files
            ]
        logger.info(
            "Number of loaded images from %s is: %s",
            str(self.image_dir),
            len(self.images),
        )

    @staticmethod
    def save_result(img: Any, save_path: str) -> None:
        """Save the final stitching result"""
        if isinstance(img, torch.Tensor):
            plt.imsave(save_path, krn.tensor_to_image(img))  # type: ignore
        else:
            plt.imsave(save_path, img)
