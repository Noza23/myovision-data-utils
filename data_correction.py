from generator_backend.correction.visual import overlay_masks_on_image

from generator_backend.utils import (
    read_image, read_masks, save_image, save_annotations,
    preprocess_image, invert_image, convert_to_hsl
)
from segment_anything.utils.amg import rle_to_mask


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser("Visualize masks")
    parser.add_argument("-i", "--image", type=str, help="Path to the image")
    parser.add_argument("-m", "--masks", type=str, help="Path to the masks")
    parser.add_argument("--delete", nargs="*", default=[], help="del by ids")
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--omit_cluster_values", nargs="*", default=[])
    parser.add_argument("--clahe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--invert", type=int, choices=[0, 1], default=0)
    parser.add_argument("-o", "--output", type=str, help="Path to the output")
    
    args = parser.parse_args()
    img = read_image(args.image)
    # PreProcessing image
    params = {
        "omit_cluster_values": [*map(int, args.omit_cluster_values)],
        "clahe": bool(args.clahe),
        "gamma": args.gamma,
    }
    print("Preprocessing image with params: ", params)
    img = preprocess_image(img, convert_to_hsl(img), **params)
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