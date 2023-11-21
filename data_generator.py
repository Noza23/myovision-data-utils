from generator_backend.adg import generate_training_data
from generator_backend.utils import read_image, read_masks

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Generate training data.")
    parser.add_argument(
        "--data_directory",
        type=str,
        help="Directory containing images and rle masks"
    )
    parser.add_argument(
        "--patch_size",
        help="Single int. Default: 1500",
        type=int,
        default=1500
    )
    parser.add_argument(
        "--device", help="Default: 'cpu'", type=str, default="cpu"
    )
    parser.add_argument(
        "--saving_path", help="Directory to save to", type=str
    )
    args = parser.parse_args()
    
    data_dir: str = args.data_directory
    patch_size: tuple[int, int] = (args.patch_size, args.patch_size)
    device: str = args.device
    saving_path: str = args.saving_path

    image_fns = [fn for fn in os.listdir(data_dir) if fn.endswith(".tif")]
    masks_fns = [fn for fn in os.listdir(data_dir) if fn.endswith(".json")]
    for image_fn in image_fns:
        mask_fn = image_fn[:-4] + "_mask.json"
        if mask_fn in masks_fns:
            image = read_image(os.path.join(data_dir, image_fn))
            rle_masks = read_masks(os.path.join(data_dir, mask_fn))
            print(f"Generating training data for {image_fn}...")
            generate_training_data(
                image=image,
                rle_masks=rle_masks,
                patch_size=patch_size,
                original_image_name=image_fn[:-4],
                saving_path=saving_path,
                device=device
            )
            print(f"Training data successfully generated for {image_fn}.")
        else:
            print(f"Could not find mask for {image_fn}. Skipping...")
            continue

