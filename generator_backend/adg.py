import numpy as np
from segment_anything.utils.amg import rle_to_mask, mask_to_rle_pytorch
import torch
from typing import Any
import gc
from .utils import (
    split_image_into_patches,
    save_image,
    save_annotations
)

# Augmentation not implemented yet

def generate_training_data(
    image: np.ndarray,
    rle_masks: list[dict],
    patch_size: tuple[int, int],
    original_image_name: str,
    saving_path: str,
    device: str="cpu"
) -> None:
    """Generate training data from original image and rle masks.

    Args:
        image (np.ndarray): Image to generate training data from.
        rle_masks (list[dict]): rle masks corresponding to image.
        patch_size (tuple[int, int]): Size of patches to split image into.
        original_image_name (str): Name of original image.
        saving_path (str): Path to save training data to.
        device (str): Device to use.
    """
    grid, patches = split_image_into_patches(image, patch_size)
    for i, patch in enumerate(patches):
        encoded_masks: list[dict[str, Any]] = []
        masks = get_masks_in_patch(rle_masks, patch_size, grid, i)
        batch_size = 10
        inds = np.arange(len(masks))
        batches = np.array_split(inds, batch_size)
        for batch in batches:
            if len(batch) != 0:
                encoded_batch = mask_to_rle_pytorch(
                    torch.from_numpy(np.stack(masks)[batch]).to(device=device)
                )
                encoded_masks.extend(encoded_batch)
                if device != "cpu":
                    torch.cuda.empty_cache()
                gc.collect()
        sa_data = {
            "patch": {
                "patch_id": i,
                "height": patch.shape[0],
                "width": patch.shape[1],
                "augmentation": "None",
                "file_name_patch": f"{original_image_name}_{i}.png",
                "file_name_image": f"{original_image_name}.tif",
            },
            "annotations": encoded_masks
        }
        # Save patch
        save_image(patch, f"{saving_path}/{original_image_name}_{i}.png")
        # Save annotations
        save_annotations(
            sa_data, f"{saving_path}/{original_image_name}_{i}.json"
        )

def get_masks_in_patch(
    rle_masks: list[dict],
    patch_size: tuple[int, int],
    grid: tuple[int, int],
    n: int
) -> list[np.ndarray]:
    """From all masks in the image, filters out the masks that are in the
    patch with index n.

    Args:
        rle_masks (list[dict]): rle masks corresponding to image.
        patch_size (tuple[int, int]): Size of patches image was split into.
        grid (tuple[int, int]): Grid of patches image was split into.
        n (int): Index of patch.
    """
    H, W = rle_masks[0]["size"]
    masks_decoded: list[np.ndarray] = [rle_to_mask(msk) for msk in rle_masks]
    row = n // grid[0]
    col = n % grid[1]
    y_start, x_start = row * patch_size[0], col * patch_size[1]
    y_end = H if row == grid[0] - 1 else (row + 1) * patch_size[0]
    x_end = W if col == grid[1] - 1 else (col + 1) * patch_size[1]
    res = [
        mask[y_start:y_end, x_start:x_end]
        for mask in masks_decoded
        if mask[y_start:y_end, x_start:x_end].sum() > 0
    ]
    return res


    