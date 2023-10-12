import seam_carving
import numpy as np
import torch

class SeamCarving:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512},
                ),
                "height": (
                    "INT",
                    {"default": 512},
                ),
            },
            "oprional": {
                "keep_mask": ("MASK",),
                "drop_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, pixels, width, height, keep_mask=None, drop_mask=None):
        results = []

        if keep_mask is not None:
            while keep_mask.dim() < 4:
                keep_mask = keep_mask[None]
            keep_mask = resize(keep_mask, pixels.size()[1:3])

        if drop_mask is not None:
            while drop_mask.dim() < 4:
                drop_mask = drop_mask[None]
            drop_mask = resize(drop_mask, pixels.size()[1:3])

        for i in range(pixels.size()[0]):
            image = pixels[i]
            if keep_mask is not None:
                k_mask = keep_mask[np.clip(i, 0, keep_mask.size()[0] - 1)][0]
            else:
                k_mask = None
            if drop_mask is not None:
                d_mask = drop_mask[np.clip(i, 0, drop_mask.size()[0] - 1)][0]
                if d_mask.any(0).all() or d_mask.any(1).all():
                    print("SeamCarving: Drop mask would delete entire image, ignoring")
                    d_mask = None
            else:
                d_mask = None
            src = (255 * image.cpu().clamp(min=0, max=1).numpy()).astype(np.uint8)
            dst = seam_carving.resize(
                image,
                size=(width, height),
                energy_mode="forward",  # choose from {backward, forward}
                order="width-first",  # choose from {width-first, height-first}
                keep_mask=k_mask,  # object mask to protect from removal
                drop_mask=d_mask
            )
            results.append(torch.from_numpy(dst))
        return (torch.stack(results),)

def resize(image, size):
    if image.size()[-2:] == size:
        return image
    return torch.nn.functional.interpolate(image, size)
