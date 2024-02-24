import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        mean = torch.tensor(self.mean, device=device)
        std = torch.tensor(self.std, device=device)
        return tensor + torch.randn(tensor.shape).to(device) * std + mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class Augmenter:
    def __init__(self, config: dict):
        """
        Augmenter class to apply augmentation to an image and masks.
        """
        self.jitter = transforms.ColorJitter(**config["ColorJitter"])
        self.add_noise = AddGaussianNoise(**config["GaussianNoise"])
        self.blur = transforms.GaussianBlur(
            config["GaussianBlur"]["kernel_size"],
            tuple(config["GaussianBlur"]["sigma"]),
        )
        self.rotate = transforms.RandomRotation(**config["RandomRotation"])
        self.invert = transforms.RandomInvert(**config["Invert"])
        self.flip_v = transforms.RandomVerticalFlip(**config["VerticalFlip"])
        self.flip_h = transforms.RandomHorizontalFlip(
            **config["HorizontalFlip"]
        )

    def __call__(
        self, image: torch.Tensor, masks: torch.Tensor, extra: dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Apply augmentation to the image and corresponding masks.

        Args:
            image (torch.Tensor): Image to be augmented.(1xCxHxW).
            masks (torch.Tensor): Masks to be augmented.(Nx1xHxW).
            extra (dict[str, int]): Dictionary containing the aug
                methods and corresponding number of times to reapply.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, list[str]]:
                Augmented image and corresponding masks.
                (AxCxHxW), (AxNx1xHxW) respectively and a list
                of strings containing the augmentation methods
        """
        aug, masks_aug, tech = [image], [masks], [""]
        for key, value in extra.items():
            print(key, value)
            if key == "n_ColorJitter":
                aug.extend([self.jitter(image) for _ in range(value)])
                masks_aug.extend([masks for _ in range(value)])
                tech.extend([repr(self.jitter) for _ in range(value)])
            elif key == "n_RandomRotation":
                for _ in range(value):
                    angle = self.rotate.get_params(self.rotate.degrees)
                    aug.append(TF.rotate(image, angle))
                    masks_aug.append(TF.rotate(masks, angle))
                    tech.append(f"TF.rotate(angle={angle})")
            elif key == "n_GaussianNoise":
                aug.extend([self.add_noise(image) for _ in range(value)])
                masks_aug.extend([masks for _ in range(value)])
                tech.extend([repr(self.add_noise) for _ in range(value)])
            elif key == "n_GaussianBlur":
                aug.extend([self.blur(image) for _ in range(value)])
                masks_aug.extend([masks for _ in range(value)])
                tech.extend([repr(self.blur) for _ in range(value)])
        if self.invert.p:
            aug.append(self.invert(image))
            masks_aug.append(masks)
            tech.append(repr(self.invert))
        if self.flip_v.p:
            aug.append(self.flip_v(image))
            masks_aug.append(self.flip_v(masks))
            tech.append(repr(self.flip_v))
        if self.flip_h.p:
            aug.append(self.flip_h(image))
            masks_aug.append(self.flip_h(masks))
            tech.append(repr(self.flip_h))
        aug = torch.vstack(aug).permute(0, 2, 3, 1)
        masks_aug = torch.stack(masks_aug).squeeze(2)
        return aug, masks_aug, tech
