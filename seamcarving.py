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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, pixels, width, height):
        results = []
        for image in pixels:
            src = np.clip(255 * image.cpu().numpy(), 0, 255).astype(np.uint8)
            dst = seam_carving.resize(
                image,
                size=(width, height),
                energy_mode="forward",  # choose from {backward, forward}
                order="width-first",  # choose from {width-first, height-first}
                keep_mask=None,  # object mask to protect from removal
            )
            results.append(torch.from_numpy(dst))
        return (results,)
