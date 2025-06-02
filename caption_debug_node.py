"""
Caption Debug Node – outputs expanded caption string for debugging
"""
import importlib
from .caption_around_image import expand_placeholders

class CaptionDebugNode:
    CATEGORY = "Debug"
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "caption": ("STRING", {"multiline": True, "default": "ImageOG: %LoadImage.image%"}),
            "prompt": ("PROMPT", {}),
            "extra_pnginfo": ("EXTRA_PNGINFO", {}),
            "debug": ("BOOLEAN", {"default": True}),
        }
        return {"required": req}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(self, caption, prompt=None, extra_pnginfo=None, debug=True, **_):
        prompt = prompt or {}
        nodes = extra_pnginfo.get("workflow", {}).get("nodes", []) if extra_pnginfo else []
        expanded = expand_placeholders(caption, prompt, nodes, debug)
        return (expanded,)

NODE_CLASS_MAPPINGS = {"CaptionDebugNode": CaptionDebugNode}
NODE_DISPLAY_NAME_MAPPINGS = {"CaptionDebugNode": "Caption Debug Node"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
