import torch

class IQACollator:
    """Simple collator that assembles a dict for Trainer.

    Each dataset sample is assumed to be ``(image_tensor, score_float)``.
    The collator stacks the images into a single tensor and returns labels as a tensor.
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        # Unpack list of tuples -> two tuples (images, scores)
        images, scores = zip(*batch)

        # (B, 3, H, W)
        images_tensor = torch.stack(list(images), dim=0)
        labels_tensor = torch.tensor(scores, dtype=images_tensor.dtype)

        return {
            "pixel_values": images_tensor,
            "labels": labels_tensor,
        } 