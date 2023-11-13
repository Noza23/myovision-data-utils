import numpy as np
import torch
import io
from typing import Union, Any
from .utils import (
    split_image_into_patches,
    enhance_contrast_bgr,
    gamma_correction,
    decode_image,
    convert_to_hsl,
    preprocess_image
)
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.utils.amg import mask_to_rle_pytorch
# Make batch-size 100

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
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.current_mask_value = 1
        # Default preprocessing parameters
        self.preprocess_params = {
            "gamma": 0.4,
            "enhance_contrast": True,
            "omit_cluster": [False for _ in range(10)]
        }
        self.image_preprocessed: Union[np.ndarray, None] = None
        self.grid: Union[tuple[int, int], None] = None
        self.patches: Union[list[np.ndarray], None] = None
        self.patch_sizes: Union[list[tuple[int, int]], None] = None
        self.valid_instances: Union[dict[list[int]], None] = None

        self.image = decode_image(image_stream)
        self.image_hsl = convert_to_hsl(self.image)
        unique, count = np.unique(self.image_hsl[:, :, 1], return_counts=True)
        top_10 = np.argsort(-count)[:10]
        self.clusters, self.cluster_count = unique[top_10], count[top_10]
        
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

    def compute_all_masks(self) -> None:
        self.grid, self.patches = split_image_into_patches(
            self.image_preprocessed,
            (self.patch_width, self.patch_height)
        )
        self.valid_instances = {i: [] for i in range(len(self.patches))}
        self.patch_sizes = [patch.shape[:2] for patch in self.patches]
        
        print("Number of patches: ", len(self.patches))
        self.masks = [self.generate_masks(patch) for patch in self.patches]   
        print("Masks generated successfully")
         
    def change_params(
        self,
        gamma: float,
        he: bool,
        omit_cluster: list[int]
    ) -> None:
        self.preprocess_params["gamma"] = gamma
        self.preprocess_params["enhance_contrast"] = he
        self.preprocess_params["omit_cluster"] = omit_cluster
    
    def perform_preprocessing(self) -> None:
        self.image_preprocessed = preprocess_image(
            self.image.copy(),
            self.image_hsl,
            self.clusters[self.preprocess_params["omit_cluster"]],
            gamma=self.preprocess_params["gamma"],
            clahe=self.preprocess_params["enhance_contrast"],
        )
    
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
                self.masks[patch_id][mask_id] * (self.current_mask_value / 255)
            )
            self.current_mask_value += 1
        if len(valid_instances_lst) == 0:
            return np.zeros(self.masks[patch_id][0].shape, dtype=np.uint8)
        
        valid_mask = np.sum(valid_instances_lst, axis=0, dtype=np.uint16)
        return valid_mask

    def rle_encode_masks(self, patch_id: int) -> list[dict[str, Any]]:
        # map patch masks back to original image
        
        full_image_masks = np.zeros(
            (len(self.valid_instances[patch_id]), *self.image.shape[:2]),
            dtype=np.uint8
        )
        row = patch_id // self.grid[1] # floor of patch_id / grid_width
        col = patch_id % self.grid[1] # remainder of patch_id / grid_width
        
        # find position of patch in original image
        x = sum(self.patch_sizes[row * self.grid[1] + c][1] for c in range(col))
        y = sum(self.patch_sizes[r * self.grid[1]][0] for r in range(row))

        for i, mask_id in enumerate(self.valid_instances[patch_id]):
            full_image_masks[i][
                y:y + self.patch_sizes[patch_id][0],
                x:x + self.patch_sizes[patch_id][1]
            ] = self.masks[patch_id][mask_id] / 255

        # encode masks
        encoded_masks = mask_to_rle_pytorch(
            torch.from_numpy(full_image_masks).to(device=state.MODEL.device),
        )
        return encoded_masks

def set_image(
    image_stream: io.BytesIO,
    image_name: str,
    patch_width: int,
    patch_height: int
) -> None:
    state.IMAGE = Image(image_stream, image_name, patch_width, patch_height)
