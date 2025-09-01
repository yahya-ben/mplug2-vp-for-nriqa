import torch
from torchvision import transforms


# Expose normalization parameters and a batch normalization utility
_MPLUG_MEAN = [0.485, 0.456, 0.406]
_MPLUG_STD = [0.229, 0.224, 0.225]


def normalize_batch_after_prompt(images: torch.Tensor) -> torch.Tensor:
    """Apply mPLUG normalization to a batch of images (B, C, H, W).

    This should be called AFTER the visual prompt is applied, so that the prompt
    pixels are normalized consistently with the original image.
    """
    if images.dim() != 4:
        raise ValueError("Expected a 4D tensor (B, C, H, W) for normalization")
    mean = torch.tensor(_MPLUG_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_MPLUG_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


class MPLUGPreprocess:
    """mPLUG-Owl2 preprocessing without normalization.

    Resize -> CenterCrop -> ToTensor
    """
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
        ])
    
    def __call__(self, image):
        return self.transform(image)


def get_train_transform():
    """Get training transform for mPLUG with basic augmentation (no normalization)."""
    return transforms.Compose([
        MPLUGPreprocess(),
        transforms.RandomHorizontalFlip(p=0.5)
    ])


def get_val_test_transform():
    """Get validation/test transform for mPLUG (no augmentation, no normalization)."""
    return MPLUGPreprocess() 