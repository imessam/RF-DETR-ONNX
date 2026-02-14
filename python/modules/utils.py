# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2026 PierreMarieCurie
# ------------------------------------------------------------------------

import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import io
import os
import requests
import numpy as np
from PIL import Image


def open_image(path: str) -> Image.Image:
    """Open an image from a local path or a URL."""
    # Check if the path is a URL (starts with 'http://' or 'https://')
    if path.startswith('http://') or path.startswith('https://'):
        img = Image.open(io.BytesIO(requests.get(path).content))
    # If it's a local file path, open the image directly
    else:
        if os.path.exists(path):
            img = Image.open(path)
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")
    return img

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxyn(x: np.ndarray) -> np.ndarray:
    """Convert boxes from center x, y, width, height (cxcywh) to min/max format (xyxyn)."""
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)
