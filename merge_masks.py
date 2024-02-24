import os
import argparse
from dataclasses import dataclass

import read_roi
import numpy as np
import cv2

from generator_backend.utils import (
    mask_to_rle,
    save_rle_masks,
    read_masks,
    rle_to_mask,
)


@dataclass
class Arguments:
    masks: str
    roi: str
    output: str
    mask_suffix: str
    roi_suffix: str

    def __post_init__(self):
        if not os.path.exists(self.masks):
            raise FileNotFoundError(f"Masks file not found: {self.masks}")
        if not os.path.exists(self.roi):
            raise FileNotFoundError(f"Roi file not found: {self.roi}")
        if not os.path.exists(self.output):
            os.makedirs(self.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge ROI masks to existing ones")
    parser.add_argument("-m", "--masks", type=str, help="Path to .json masks")
    parser.add_argument("-r", "--roi", type=str, help="Path to zipped roi")
    parser.add_argument("-o", "--output", type=str, help="Path to output")
    parser.add_argument(
        "--mask_suffix", type=str, default="_mask_filtered.json"
    )
    parser.add_argument("--roi_suffix", type=str, default=".zip")

    args = Arguments(**vars(parser.parse_args()))

    mask_files = os.listdir(args.masks)
    roi_files = os.listdir(args.roi)
    matching_files = [
        f.replace(args.mask_suffix, "")
        for f in mask_files
        if f.replace(args.mask_suffix, "") + args.roi_suffix in roi_files
    ]
    print("Matching files found: ", matching_files)

    for f in matching_files:
        print("> Processing: ", f)
        mask_file = os.path.join(args.masks, f + args.mask_suffix)
        roi_file = os.path.join(args.roi, f + args.roi_suffix)

        masks_json = read_masks(mask_file)
        masks_np = [rle_to_mask(mask) for mask in masks_json]

        rois = read_roi.read_roi_zip(roi_file)
        binary_roi_masks = []
        binary_roi_mask = np.zeros_like(masks_np[0], dtype=np.uint8)

        for key, roi in rois.items():
            x, y = roi["x"], roi["y"]
            coords = np.round(np.stack([x, y], axis=1)).astype(np.int32)
            mask = binary_roi_mask.copy()
            try:
                cv2.fillPoly(mask, [coords], (255))
            except Exception as e:
                print(f"in {f} roi with key {key} failed: ")
                raise e
            binary_roi_masks.append(mask.astype(np.bool_))

        masks_np.extend(binary_roi_masks)
        rle_masks = mask_to_rle(
            np.stack(masks_np, axis=0), batch_size=10, device="cpu"
        )
        print("Total masks: ", len(rle_masks))
        mask_fn = os.path.join(args.output, f + "_mask_merged.json")
        save_rle_masks(rle_masks, mask_fn)
        print("RLE masks saved to: ", mask_fn)

    print("Mergning Masks completed!")
