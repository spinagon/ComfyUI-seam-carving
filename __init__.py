from .SeamCarving import SeamCarving

NODE_CLASS_MAPPINGS = {
    "SeamCarving": SeamCarving,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamCarving": "Image Resize (seam carving)",
}

__all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
