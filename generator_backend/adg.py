import numpy as np
from segment_anything.utils.amg import rle_to_mask, mask_to_rle_pytorch
import torch
from torchvision import transforms
from typing import Any, Union, Tuple, List
import gc
from .utils import (
    split_image_into_patches,
    save_image,
    save_annotations
)
from PIL import Image
from functools import partial

def add_color_jitter(img: Union[np.ndarray, Image.Image, torch.Tensor], masks: List[np.ndarray], brightness: float = 1, contrast: float = 1, saturation: float = 1, hue: float = 0.5) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Apply color jitter transformation to the input image.

    Parameters:
        img (Union[np.ndarray, Image.Image, torch.Tensor]): Input image.
        masks (List[np.ndarray]): List of input masks. Masks are not manipulated and returned as inputted.
        brightness (float): Brightness factor for color jitter. Default is 1.
        contrast (float): Contrast factor for color jitter. Default is 1.
        saturation (float): Saturation factor for color jitter. Default is 1.
        hue (float): Hue factor for color jitter. Default is 0.5.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: Transformed image and list of masks.
    """
    cj = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    if isinstance(img, np.ndarray):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return cj(torch.from_numpy(img)).permute(1, 2, 0), masks
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return cj(torch.from_numpy(np.moveaxis(img, -1, 0))).permute(1, 2, 0), masks
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    elif isinstance(img, Image.Image):
        return cj(img), masks
    elif isinstance(img, torch.Tensor):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return cj(img).permute(1, 2, 0), masks
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return cj(img.permute(2, 0, 1)).permute(1, 2, 0), masks
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    else:
        raise TypeError(f"Unsupported input type {type(img)}")

def add_white_noise(img: Union[np.ndarray, Image.Image, torch.Tensor], masks: List[np.ndarray], sigma: float = 10) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Add white noise to the input image.

    Parameters:
        img (Union[np.ndarray, Image.Image, torch.Tensor]): Input image.
        masks (List[np.ndarray]): List of input masks. Masks are not manipulated and returned as inputted.
        sigma (float): Standard deviation of the noise. Default is 10.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: Image with added noise and list of masks.
    """
    def add_noise(image, scale=sigma):
        return np.clip(image + np.random.normal(scale=scale, size=image.shape), 0, 255).astype(np.uint8)

    if isinstance(img, np.ndarray):
        img = add_noise(img)
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return torch.from_numpy(img).permute(1, 2, 0), masks
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return torch.from_numpy(img), masks
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    elif isinstance(img, Image.Image):
        return torch.from_numpy(add_noise(np.array(img))), masks
    elif isinstance(img, torch.Tensor):
        return torch.from_numpy(add_noise(np.array(img))), masks
    else:
        raise TypeError(f"Unsupported input type {type(img)}")

def add_blur(img: Union[np.ndarray, Image.Image, torch.Tensor], masks: List[np.ndarray], kernel_size: int = 21, sigma: float = 7) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Apply Gaussian blur to the input image.

    Parameters:
        img (Union[np.ndarray, Image.Image, torch.Tensor]): Input image.
        masks (List[np.ndarray]): List of input masks. Masks are not manipulated and returned as inputted.
        kernel_size (int): Size of the Gaussian kernel. Default is 21.
        sigma (float): Standard deviation of the Gaussian kernel. Default is 7.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: Blurred image and list of masks.
    """
    trafo = partial(transforms.functional.gaussian_blur, kernel_size=kernel_size, sigma=sigma)

    if isinstance(img, np.ndarray):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return trafo(torch.from_numpy(img)).permute(1, 2, 0), masks
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return trafo(torch.from_numpy(np.moveaxis(img, -1, 0))).permute(1, 2, 0), masks
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    elif isinstance(img, Image.Image):
        return trafo(img), masks
    elif isinstance(img, torch.Tensor):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return trafo(img).permute(1, 2, 0), masks
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return trafo(img.permute(2, 0, 1)).permute(1, 2, 0), masks
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    else:
        raise TypeError(f"Unsupported input type {type(img)}")

def flip_image_vertically(img: Union[np.ndarray, Image.Image, torch.Tensor], masks: List[np.ndarray]) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Flip the input image vertically.

    Parameters:
        img (Union[np.ndarray, Image.Image, torch.Tensor]): Input image.
        masks (List[np.ndarray]): List of input masks.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: Vertically flipped image and list of flipped masks.
    """
    flip = transforms.RandomVerticalFlip(p=1.0)
    masks_transformed = [flip(torch.from_numpy(mask).unsqueeze(dim=0)).squeeze(0) for mask in masks]
    
    if isinstance(img, np.ndarray):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return flip(torch.from_numpy(img)).permute(1, 2, 0), masks_transformed
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return flip(torch.from_numpy(np.moveaxis(img, -1, 0))).permute(1, 2, 0), masks_transformed
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    elif isinstance(img, Image.Image):
        return flip(img), masks_transformed
    elif isinstance(img, torch.Tensor):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return flip(img).permute(1, 2, 0), masks_transformed
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return flip(img.permute(2, 0, 1)).permute(1, 2, 0), masks_transformed
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    else:
        raise TypeError(f"Unsupported input type {type(img)}")

def flip_image_horizontally(img: Union[np.ndarray, Image.Image, torch.Tensor], masks: List[np.ndarray]) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Flip the input image horizontally.

    Parameters:
        img (Union[np.ndarray, Image.Image, torch.Tensor]): Input image.
        masks (List[np.ndarray]): List of input masks.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: Horizontally flipped image and list of flipped masks.
    """
    flip = transforms.RandomHorizontalFlip(p=1.0)
    masks_transformed = [flip(torch.from_numpy(mask).unsqueeze(dim=0)).squeeze(0) for mask in masks]
    
    if isinstance(img, np.ndarray):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return flip(torch.from_numpy(img)).permute(1, 2, 0), masks_transformed
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return flip(torch.from_numpy(np.moveaxis(img, -1, 0))).permute(1, 2, 0), masks_transformed
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    elif isinstance(img, Image.Image):
        return flip(img), masks_transformed
    elif isinstance(img, torch.Tensor):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return flip(img).permute(1, 2, 0), masks_transformed
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return flip(img.permute(2, 0, 1)).permute(1, 2, 0), masks_transformed
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    else:
        raise TypeError(f"Unsupported input type {type(img)}")

def random_rotation(img: Union[np.ndarray, Image.Image, torch.Tensor], masks: List[np.ndarray]) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Apply a random rotation (in the interval [-45°, 45°]) to the input image.

    Parameters:
        img (Union[np.ndarray, Image.Image, torch.Tensor]): Input image.
        masks (List[np.ndarray]): List of input masks.

    Returns:
        Tuple[torch.Tensor, List[np.ndarray]]: Rotated image and list of rotated masks.
    """
    rot_angle = 45
    rot_angle = np.random.uniform(-rot_angle, rot_angle)
    rotation = partial(transforms.functional.rotate, angle=rot_angle)
    masks_transformed = [rotation(torch.from_numpy(mask).unsqueeze(dim=0)).squeeze(0) for mask in masks]
    
    if isinstance(img, np.ndarray):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return rotation(torch.from_numpy(img)).permute(1, 2, 0), masks_transformed
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return rotation(torch.from_numpy(np.moveaxis(img, -1, 0))).permute(1, 2, 0), masks_transformed
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    elif isinstance(img, Image.Image):
        return rotation(img), masks_transformed
    elif isinstance(img, torch.Tensor):
        if (img.shape[0] == 3) or (img.shape[0] == 1):
            return rotation(img).permute(1, 2, 0), masks_transformed
        elif (img.shape[-1] == 3) or (img.shape[-1] == 1):
            return rotation(img.permute(2, 0, 1)).permute(1, 2, 0), masks_transformed
        else:
            raise ValueError(f"Unsupported shape {img.shape}")
    else:
        raise TypeError(f"Unsupported input type {type(img)}")



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


    