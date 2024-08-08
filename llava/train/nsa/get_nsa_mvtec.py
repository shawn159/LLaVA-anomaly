"""
this code is based on original nsa
- anomaly gpt has different dataset class customized to the anomaly gpt.
- here if the arugment "use_anomaly_gpt_configs" set to True, the configs of nsa used from anomalygpt is adapted.
"""

import os
import torch
from torchvision import transforms as T
import argparse
from tqdm import tqdm
import cv2

from self_sup_data.mvtec import SelfSupMVTecDataset, CLASS_NAMES, TEXTURES, OBJECTS
from settings import (
    SETTINGS,
    WIDTH_BOUNDS_PCT,
    MIN_OVERLAP_PCT,
    MIN_OBJECT_PCT,
    NUM_PATCHES,
    INTENSITY_LOGISTIC_PARAMS,
    UNALIGNED_OBJECTS,
    BACKGROUND,
)

def get_nsa_mvtec_dataset(
    class_name,
    root_path,
    setting,
    is_train,
    use_anomaly_gpt_configs=True,
    ellipse=False,
    use_mask=True,
    single_patch=False,
    cutpaste_patch_gen=False,
):
    if is_train:
        if use_anomaly_gpt_configs:
            res = 224

            train_transform = T.Resize(
                (224, 224), interpolation=T.InterpolationMode.BICUBIC
            )            

            dataset = SelfSupMVTecDataset(
                use_anomaly_gpt_configs=use_anomaly_gpt_configs,
                root_path=root_path,
                class_name=class_name,
                is_train=True,
                low_res=res,
                download=False,
                transform=train_transform,
            )
            """
            dataset: return x, y, mask
            """

            self_sup_args = {
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
                self_sup_args["resize_bounds"] = (0.5, 2)

            dataset.configure_self_sup(self_sup_args)

        else:
            # load data
            if class_name in UNALIGNED_OBJECTS:
                train_transform = T.Compose(
                    [T.RandomRotation(5), T.CenterCrop(230), T.RandomCrop(224)]
                )
                res = 256
            elif class_name in OBJECTS:
                # no rotation for aligned objects
                train_transform = T.Compose([T.CenterCrop(230), T.RandomCrop(224)])
                res = 256
            else:  # texture
                train_transform = T.Compose([T.RandomVerticalFlip(), T.RandomCrop(256)])
                res = 264

            train_dat = SelfSupMVTecDataset(
                use_anomaly_gpt_configs=use_anomaly_gpt_configs,
                root_path=root_path,
                class_name=class_name,
                is_train=True,
                low_res=res,
                download=False,
                transform=train_transform,
            )
            """
            train_data: return x, y, mask
            """

            train_dat.configure_self_sup(self_sup_args=setting.get("self_sup_args"))
            if setting.get("skip_background", False) and use_mask:
                train_dat.configure_self_sup(
                    self_sup_args={"skip_background": BACKGROUND.get(class_name)}
                )
            if class_name in TEXTURES:
                train_dat.configure_self_sup(self_sup_args={"resize_bounds": (0.5, 2)})

            train_dat.configure_self_sup(
                on=True,
                self_sup_args={
                    "width_bounds_pct": WIDTH_BOUNDS_PCT.get(class_name),
                    "intensity_logistic_params": INTENSITY_LOGISTIC_PARAMS.get(class_name),
                    "num_patches": 1 if single_patch else NUM_PATCHES.get(class_name),
                    "min_object_pct": MIN_OBJECT_PCT.get(class_name),
                    "min_overlap_pct": MIN_OVERLAP_PCT.get(class_name),
                },
            )
            if ellipse:
                train_dat.configure_self_sup(self_sup_args={"num_ellipses": 5})

            if cutpaste_patch_gen:
                train_dat.configure_self_sup(
                    self_sup_args={"cutpaste_patch_generation": True}
                )
    else:
        dataset = SelfSupMVTecDataset(
            use_anomaly_gpt_configs=use_anomaly_gpt_configs,
            root_path=root_path,
            class_name=class_name,
            is_train=False,
            low_res=256,
            download=False,
            transform=None,
        )

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", required=True, type=str)
    parser.add_argument("-d", "--data_dir", required=True, type=str)
    parser.add_argument("-s", "--setting", required=True, type=str)
    parser.add_argument("-c", "--categories", required=False, type=str, default="all")
    parser.add_argument("-n", "--class_name", required=False, type=str, default=None)
    parser.add_argument("--single_patch", required=False, action="store_true")
    parser.add_argument("--cutpaste_patch_gen", required=False, action="store_true")
    parser.add_argument("--no_mask", required=False, action="store_true")
    parser.add_argument("--no_pool", required=False, action="store_true")
    parser.add_argument("--preact", required=False, action="store_true")
    parser.add_argument("--ellipse", required=False, action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    setting = SETTINGS.get(args.setting)

    if args.class_name is not None:
        categories = [args.class_name]
    elif args.categories == "texture":
        categories = TEXTURES
    elif args.categories == "object":
        categories = OBJECTS
    else:
        categories = CLASS_NAMES

    for class_name in tqdm(categories):
        train(
            class_name,
            data_dir,
            setting,
            args.ellipse,
            not args.no_mask,
            args.single_patch,
            args.cutpaste_patch_gen,
        )
