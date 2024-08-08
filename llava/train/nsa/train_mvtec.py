import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, sampler
import matplotlib.pyplot as plt
from torchvision import transforms as T
import argparse
from tqdm import tqdm

from self_sup_data.mvtec import SelfSupMVTecDataset, CLASS_NAMES, TEXTURES, OBJECTS
from model.resnet import resnet18_enc_dec
from experiments.training_utils import train_and_save_model
from settings import (
    SETTINGS,
    WIDTH_BOUNDS_PCT,
    MIN_OVERLAP_PCT,
    MIN_OBJECT_PCT,
    NUM_PATCHES,
    INTENSITY_LOGISTIC_PARAMS,
    UNALIGNED_OBJECTS,
    EPOCHS,
    BACKGROUND,
)


def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(
    class_name,
    data_dir,
    out_dir,
    setting,
    device,
    pool,
    preact,
    ellipse,
    use_mask=True,
    single_patch=False,
    cutpaste_patch_gen=False,
    min_lr=1e-6,
    max_lr=1e-3,
    batch_size=64,
    seed=1982342,
):
    set_seed(setting.get("seed", seed))
    num_epochs = EPOCHS.get(class_name)
    if setting.get("batch_size"):
        batch_size = setting.get("batch_size")
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
        root_path=data_dir,
        class_name=class_name,
        is_train=True,
        low_res=res,
        download=False,
        transform=train_transform,
    )

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
        num_epochs += 240
        train_dat.configure_self_sup(self_sup_args={"num_ellipses": 5})
        # if MIN_OBJECT_PCT.get(class_name) is not None:
        #    train_dat.configure_self_sup(self_sup_args={'min_object_pct': MIN_OBJECT_PCT.get(class_name) / 2})
        # if setting.get('self_sup_args').get('gamma_params') is not None:
        #    shape, scale, lower_bound = setting.get('self_sup_args').get('gamma_params')
        #    train_dat.configure_self_sup(self_sup_args={'gamma_params': (2*shape, scale, lower_bound)})

    if cutpaste_patch_gen:
        train_dat.configure_self_sup(self_sup_args={"cutpaste_patch_generation": True})

    loader_train = DataLoader(
        train_dat,
        batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        worker_init_fn=lambda _: np.random.seed(
            torch.utils.data.get_worker_info().seed % 2**32
        ),
    )

    model = resnet18_enc_dec(
        num_classes=1,
        pool=pool,
        preact=preact,
        final_activation=setting.get("final_activation"),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=min_lr
    )
    loss_func = setting.get("loss")()

    out_dir = os.path.join(out_dir, setting.get("out_dir"), class_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_and_save_model(
        model,
        optimizer,
        loss_func,
        loader_train,
        class_name + "_" + setting.get("fname"),
        out_dir,
        num_epochs=num_epochs,
        save_freq=80,
        device=device,
        scheduler=scheduler,
        save_intermediate_model=False,
    )


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
            out_dir,
            setting,
            device,
            not args.no_pool,
            args.preact,
            args.ellipse,
            not args.no_mask,
            args.single_patch,
            args.cutpaste_patch_gen,
        )
