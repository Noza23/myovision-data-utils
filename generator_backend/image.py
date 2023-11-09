import numpy as np
import torch
import io
from .utils import (
    split_image_into_patches,
    enhance_contrast_bgr,
    gamma_correction,
    decode_image
)
from segment_anything import SamAutomaticMaskGenerator
from . import state

class Image:
    def __init__(
        self,
        image_stream: io.BytesIO,
        image_name: str,
        patch_width: int,
        patch_height: int
    ):
        self.image_name = image_name
        self.current_mask_value = 1
        self.image = decode_image(image_stream)
        self.grid, self.patches = split_image_into_patches(
            self.image,
            (patch_width, patch_height)
        )
        self.patches = [self.preprocess_patch(patch) for patch in self.patches]
        if state.MODEL is None:
            raise Exception("Model not loaded")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=state.MODEL,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            min_mask_region_area=100,
        )
        # Generating masks for each patch
        print("Generating masks...")
        self.masks = [self.generate_masks(patch) for patch in self.patches]
        print("Masks generated successfully")
        self.valid_instances: dict[list[int]] = {
            i: [] for i in range(len(self.patches))
        }

    def preprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        patch_processed = enhance_contrast_bgr(gamma_correction(patch, 0.4))
        return patch_processed

    def generate_masks(self, patch: np.ndarray) -> list[np.ndarray]:
        print("Starting Generating Masks for patch...")
        masks = self.mask_generator.generate(patch)
        masks_fltd = [
            mask["segmentation"].astype(np.uint8) * 255
            for mask in masks
            if mask["area"] > 1000
        ]
        print("Generating Masks for patch completed!")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return masks_fltd

    def generate_masks_fake(self, patch: np.ndarray) -> list[np.ndarray]:
        return [
            np.random.binomial(1, 0.5, patch.shape[:2]).astype(np.uint8) * 255
            for _ in range(15)
        ]

    def add_valid_instance(self, patch_id: int, mask_id: int) -> None:
        if mask_id not in self.valid_instances[patch_id]:
            self.valid_instances[patch_id].append(mask_id)

    def get_valid_mask(self, patch_id: int) -> np.ndarray:
        valid_instances_lst = []

        for mask_id in self.valid_instances[patch_id]:
            valid_instances_lst.append(
                self.masks[patch_id][mask_id] * self.current_mask_value
            )
            self.current_mask_value += 1
        if len(valid_instances_lst) == 0:
            return np.zeros(self.masks[patch_id][0].shape, dtype=np.uint8)
        valid_mask = np.sum(valid_instances_lst, axis=0)
        return valid_mask

def set_image(
    image_stream: io.BytesIO,
    image_name: str,
    patch_width: int,
    patch_height: int
) -> None:
    state.IMAGE = Image(image_stream, image_name, patch_width, patch_height)
