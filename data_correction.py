# Description: This script is used to remove redundant masks manually from the
# image, it addionally saves the images with masks drawn on it.

import argparse
import os

from generator_backend.utils import (
    read_image,
    read_masks,
    save_image,
    save_annotations,
    preprocess_image,
    invert_image,
    overlay_masks_on_image,
    convert_to_hsl,
    rle_to_mask,
)

from dataclasses import dataclass


@dataclass
class Arguments:
    image: str
    masks: str
    delete: list[str]
    gamma: float
    omit_cluster_values: list[str]
    clahe: int
    invert: int
    output: str

    def __post_init__(self):
        if not os.path.exists(self.image):
            raise FileNotFoundError(f"Image file not found: {self.image}")
        if not os.path.exists(self.masks):
            raise FileNotFoundError(f"Masks file not found: {self.masks}")
        if not os.path.exists(self.output):
            os.makedirs(self.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize masks")
    parser.add_argument("-i", "--image", type=str, help="Path to the image")
    parser.add_argument("-m", "--masks", type=str, help="Path to the masks")
    parser.add_argument("--delete", nargs="*", default=[], help="del by ids")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--omit_cluster_values", nargs="*", default=[])
    parser.add_argument("--clahe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--invert", type=int, choices=[0, 1], default=0)
    parser.add_argument("-o", "--output", type=str, help="Path to the output")

    args = Arguments(**vars(parser.parse_args()))
    img = read_image(args.image)
    # PreProcessing image
    img = preprocess_image(
        img,
        convert_to_hsl(img),
        clahe=bool(args.clahe),
        omit_cluster_values=[*map(int, args.omit_cluster_values)],
        gamma=args.gamma,
    )
    if args.invert:
        print("Inverting image...")
        img = invert_image(img)
    masks_json = read_masks(args.masks)
    idxs = [*map(int, args.delete)]
    if args.delete:
        print("Deleting masks with ids: ", idxs)
        masks_json = [
            mask for i, mask in enumerate(masks_json) if i not in idxs
        ]
    print(f"Drawing total of {len(masks_json)} masks...")
    masks = [rle_to_mask(mask) for mask in masks_json]
    img = overlay_masks_on_image(img, masks, patch_size=(1500, 1500))
    image_name: str = os.path.basename(args.image)
    masks_name: str = os.path.basename(args.masks)

    save_image(img, os.path.join(args.output, image_name))
    save_annotations(masks_json, os.path.join(args.output, masks_name))
    print("Overlaid image and masks saved to: ", args.output)
