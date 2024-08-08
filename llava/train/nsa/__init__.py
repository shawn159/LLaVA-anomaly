import torch
import numpy as np
import cv2
import os
from PIL import Image

from .self_sup_data.mvtec import CLASS_NAMES, TEXTURES, OBJECTS
from .settings import (
    SETTINGS,
    WIDTH_BOUNDS_PCT,
    MIN_OVERLAP_PCT,
    MIN_OBJECT_PCT,
    NUM_PATCHES,
    INTENSITY_LOGISTIC_PARAMS,
    UNALIGNED_OBJECTS,
    BACKGROUND,
)
from .self_sup_data.self_sup_tasks import patch_ex


class NSA_transform:
  def __init__(self, data_dir, class_name):
    """
    written by s.choi

    This is the transform function that synthesis anomalies by nsa.
    Normally, nsa is implemented with various configs, but here, using anomalygpt's configs as default.
    
    SOME DIFFERENCES from nsa/anomalygpt
    - Basically, the nsa and anomalygpt resize the images around 256, here, the process has skipped.
    - After resize, normally for training, nsa/anomalygpt transform one more with augmentation techniques for training. (e.g., random_crop)
      Here, we skipped it for the consistency with reference images.
    - NSA and anomalygpt provide the dataset as class, which load the batch images with
      label(0 or 1), mask, images, but here, this is the function style synthesizer.
    """

    self.data_dir = data_dir
    self.class_name = class_name
    self.xs = []

    self.self_sup_args = {
        "width_bounds_pct": WIDTH_BOUNDS_PCT.get(class_name),
        "intensity_logistic_params": INTENSITY_LOGISTIC_PARAMS.get(class_name),
        "num_patches": 2,  # if single_patch else NUM_PATCHES.get(class_name),
        "min_object_pct": 0,
        "min_overlap_pct": 0.25,
        "gamma_params": (2, 0.05, 0.03),
        "resize": True,
        "shift": True,
        "same": False,
        "mode": cv2.NORMAL_CLONE,
        "label_mode": "logistic-intensity",
        "skip_background": BACKGROUND.get(class_name),
    }
    if class_name in TEXTURES:
        self.self_sup_args["resize_bounds"] = (0.5, 2)
    
    x_paths = []
    img_dir = os.path.join(data_dir, self.class_name, "train", "good")
    img_fpath_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    x_paths.extend(img_fpath_list)

    for path in x_paths:
      self.xs.append(Image.open(path).convert('RGB')) 


  def transform(self, image):
    image = np.asarray(image)
    prev_idx = np.random.randint(len(self.xs))
    get_previous_image = self.xs[prev_idx]

    previous_image = np.asarray(get_previous_image)
    x, mask = patch_ex(image, previous_image, **self.self_sup_args)

    """
    below just changing shape
    torch.Size([1, 256, 256]) => (256, 256, 1)
    """
    mask = torch.tensor(mask[None, ..., 0]).float()
    
    x = Image.fromarray(x)
    return x, mask