# ─────────────────────────────────────────────────────────────────────────
# debug_prompt_values.py  (optional helper node)
# ------------------------------------------------------------------------
"""
PrintPromptValues – console helper
──────────────────────────────────
Drop this file alongside your other custom nodes if you need to quickly
inspect what keys / nested structures are inside the PROMPT dict that
ComfyUI passes at runtime. Enable *debug* to pretty‑print the first
10 000 characters.
"""
import json, pprint

class PrintPromptValues:
    CATEGORY = "Debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("PROMPT", {}),
                "show_full": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "run"

    def run(self, prompt, show_full=False, debug=True, **_):
        if debug:
            txt = json.dumps(prompt, indent=2)
            if not show_full and len(txt) > 10000:
                txt = txt[:10000] + "... (truncated)"
            print("[PrintPromptValues] PROMPT →\n" + txt)
        # node returns nothing, acts purely as a side‑effect helper
        return ()

NODE_CLASS_MAPPINGS.update({"PrintPromptValues": PrintPromptValues})
NODE_DISPLAY_NAME_MAPPINGS.update({"PrintPromptValues": "Print Prompt Values"})
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]