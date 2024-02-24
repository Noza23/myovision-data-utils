from generator_backend.adg.adg import generate_training_data
from generator_backend.utils import read_image, read_masks

import os

import torch
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="data_generator")
def main(cfg: DictConfig) -> None:
    """Main function for generating training data."""
    data_dir = str(cfg["config"]["data_dir"])
    patch_size = cfg["config"]["patch_size"]
    files = os.listdir(data_dir)
    image_fns = [fn for fn in files if fn.endswith(".tif")]
    masks_fns = [fn for fn in files if fn.endswith(".json")]
    torch.manual_seed(cfg["augmentation"]["extra"]["manual_seed"])
    for image_fn in image_fns:
        mask_fn = image_fn[:-4] + "_mask.json"
        if mask_fn in masks_fns:
            image = read_image(os.path.join(data_dir, image_fn))
            masks = read_masks(os.path.join(data_dir, mask_fn))
            print(f"Generating training data for {image_fn}...")
            generate_training_data(
                aug_cfg=cfg["augmentation"],
                image=image,
                rle_masks=masks,
                patch_size=(patch_size, patch_size),
                image_name=image_fn[:-4],
                saving_path=str(cfg["config"]["saving_path"]),
                device=cfg["config"]["device"],
            )
            print(f"Training data successfully generated for {image_fn}.")
        else:
            print(f"Could not find mask for {image_fn}. Skipping...")
            continue


if __name__ == "__main__":
    main()
