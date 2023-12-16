import numpy as np
import torch
from segment_anything.utils.amg import mask_to_rle_pytorch
import cv2
import base64, io, json
from typing import Any, Union

def split_image_into_patches(
    image: np.ndarray,
    patch_size: tuple[int ,int]
) -> tuple[tuple[int, int], list[np.ndarray]]:
    """Split an image into patches of a given size."""
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size

    if patch_height > height or patch_width > width:
        return (0, 0), [image]

    patches = []
    width_reminder = width % patch_width
    height_reminder = height % patch_height
    width_range = [*range(0, width, patch_width)]
    height_range = [*range(0, height, patch_height)]

    # if reminder less than quarter of patch-size, merge it to the last patch
    if width_reminder < patch_size[0] / 4:
        width_range[-1] += width_reminder
    else:
        width_range.append(width)
    # if reminder less than quarter of patch-size, merge it to the last patch
    if height_reminder < patch_size[1] / 4:
        height_range[-1] += height_reminder
    else:
        height_range.append(height)

    # grid is needed for later reconstruction
    grid = (len(height_range) - 1, len(width_range) - 1)

    for i in range(len(height_range) - 1):
        for j in range(len(width_range) - 1):
            left, right = width_range[j], width_range[j + 1]
            lower, upper = height_range[i], height_range[i + 1]
            patch = image[lower:upper, left:right, ]
            patches.append(patch)
    return grid, patches

def reconstruct_mask_from_patches(
    patches: list[np.ndarray],
    grid: tuple[int, int],
) -> np.ndarray:
    """Reconstruct full mask from patches."""
    if grid == (0, 0):
        return patches[0]
    n_row, n_col = grid
    # col bind
    row_patches = []
    for i in range(n_row):
        row_patches.append(
            np.concatenate(patches[i * n_col:(i + 1) * n_col], axis=1)
        )
    # row bind
    return np.concatenate(row_patches, axis=0)

def enhance_contrast_bgr(image: np.ndarray) -> np.ndarray:
    """Enhance contrast of an image in BGR color space."""
    # lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(1, 1))
    clahe_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([clahe_l, a_channel, b_channel])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_Lab2BGR)
    return enhanced_image

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)
    return res

def encode_image(image: np.ndarray) -> str:
    """Encode image to base64 string."""
    _, img_bts = cv2.imencode(".png", image)
    image_url = f"data:image/png;base64,{base64.b64encode(img_bts).decode()}"
    return image_url

def decode_image(image_stream: io.BytesIO) -> np.ndarray:
    file_bytes = np.asarray(
        bytearray(image_stream.read()), dtype=np.uint8
    )
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return img

def convert_to_hsl(image: np.ndarray) -> np.ndarray:
    """Convert image to HSL color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def invert_image(image: np.ndarray) -> np.ndarray:
    """Invert an BGR image."""
    return cv2.cvtColor(
        255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        cv2.COLOR_GRAY2BGR
    )

def preprocess_image(
    image: np.ndarray,
    hsl_space: np.ndarray,
    omit_cluster_values: list[int],
    gamma: float,
    clahe: bool,
) -> np.ndarray:
    """
    Preprocesses image before generating masks.
    Args:
        image: image to preprocess.
        hsl_space: image in HSL color space.
        omit_cluster_values: list of values to omit desired clusters
            according to HSL color space based on lightness intensity.
        gamma: gamma value for gamma correction.
        clahe: whether to use clahe for contrast enhancement.
    
    Returns:
        preprocessed image.
    """
    for value in omit_cluster_values:
        image[hsl_space[:, :, 1] == value] = (0, 0, 0)
    if clahe:
        return enhance_contrast_bgr(gamma_correction(image, gamma))
    else:
        return gamma_correction(image, gamma)

def add_masks(
    patch: np.ndarray,
    masks: list[np.ndarray],
    make_gray:bool=False
) -> None:
    # Convert to Gray to save memory
    if type(masks) != list:
        masks = [masks]
    if len(masks) == 0:
        return patch
    if make_gray:
        masked_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        masked_patch = np.copy(patch)
    if len(masks) == 1:
        # If single mask, make it white for gray-scale, green for RGB.
        if len(masked_patch.shape) == 2:
            col = 255
        elif len(masked_patch.shape) == 3:
            col = (0, 255, 0)
        masked_patch[masks[0] > 0] = col
        return masked_patch

    for mask in masks:
        # If multiple masks, make each mask a random color for rgb,
        # white for gray-scale.
        if len(masked_patch.shape) == 2:
            rand_color = 255
        elif len(masked_patch.shape) == 3:
            rand_color = np.random.randint(0, 255, 3)
        masked_patch[mask > 0] = rand_color
    return masked_patch

def mask_to_rle(
    array: Union[np.ndarray, torch.Tensor],
    batch_size: int, 
    device: str="cpu"
) -> list[dict[str, Any]]:
    """Converts a numpy array to a run-length encoded list of dicts."""
    encoded_masks: list[dict[str, Any]] = []
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    batches = np.array_split(
        np.arange(len(array)), int(len(array) / batch_size) + 1
    )
    for b in batches:
        if len(b) != 0:
            b = torch.from_numpy(b).to(device)
            encoded_batch = mask_to_rle_pytorch(array[b])
            encoded_batch = [m for m in encoded_batch if len(m["counts"]) > 1]
            encoded_masks.extend(encoded_batch)
    return encoded_masks

def save_mask_to_path(mask: np.ndarray, path: str) -> None:
    path = path.split(".")[0]
    cv2.imwrite(path + ".png", mask)

def save_rle_masks(rle_masks: list[dict[str, Any]], path: str) -> None:
    path = path.split(".")[0]
    json.dump(rle_masks, open(path + ".json", "w"))

def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image)

def save_annotations(annotations: list[dict[str, Any]], path: str) -> None:
    json.dump(annotations, open(path, "w"))

def read_image(path: str) -> np.ndarray:
    return cv2.imread(path)

def read_image_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def read_masks(path: str) -> list[dict[str, Any]]:
    return json.load(open(path, "r"))