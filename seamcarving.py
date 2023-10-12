import seam_carving
import numpy as np

class SeamCarving:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096, "step": 64},
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
            results.append(torch.from_numpy(dst.astype(np.float32) / 255))
        return (results,)
