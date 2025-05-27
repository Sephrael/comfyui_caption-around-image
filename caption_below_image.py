"""
CaptionBelowImage – mini-parser edition
────────────────────────────────────────────────────────────────────────
• Supports   %date:…%     →   formatted datetime
           %3.image%      →   widget value from node #3
           %Load Image.image% → widget from the highest-ID “LoadImage” node
• Tick “debug” to dump raw/parsed captions and unresolved tokens.

Font-scaling, RGBA composition, etc. unchanged from earlier builds.
"""

import os, re, textwrap, traceback, numpy as np, torch
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import importlib.resources as pkgres


# ───────────────────────── tensor ⇄ PIL (silence Non-writable warning)
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0)


# ───────────────────────── font helper
def _builtin_ttf() -> str | None:
    try:
        return str(pkgres.files("PIL").joinpath("fonts/DejaVuSans.ttf"))
    except Exception:
        return None


def _load_font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size)
        builtin = _builtin_ttf()
        if builtin:
            return ImageFont.truetype(builtin, size)
    except Exception:
        pass
    return ImageFont.load_default()


# ───────────────────────── placeholder mini-parser
_DATE_REPL = {
    "yyyy": "%Y",
    "yy": "%y",
    "MM": "%m",
    "dd": "%d",
    "hh": "%H",
    "mm": "%M",
    "ss": "%S",
}
PH_RE = re.compile(r"%([^%]+)%")


def _fmt_date(spec: str) -> str:
    fmt = spec
    # longest first so 'yyyy' doesn't also hit 'yy'
    for k in sorted(_DATE_REPL, key=len, reverse=True):
        fmt = fmt.replace(k, _DATE_REPL[k])
    return datetime.now().strftime(fmt)


def _canonical(name: str) -> str:
    return name.replace(" ", "").lower()


def _lookup_widget(prompt: dict, node_spec: str, widget: str):
    """
    prompt     – full prompt JSON (dict keyed by node-ID str)
    node_spec  – '3'  or  'Load Image'
    widget     – e.g. 'image', 'width', etc.
    """
    # numeric ID first
    if node_spec.isdigit() and node_spec in prompt:
        return prompt[node_spec]["inputs"].get(widget, None)

    wanted = _canonical(node_spec)
    best_id = None
    # highest-ID match wins
    for nid, ndata in prompt.items():
        if _canonical(ndata.get("class_type", "")) == wanted:
            if widget in ndata.get("inputs", {}):
                if best_id is None or int(nid) > int(best_id):
                    best_id = nid
    if best_id:
        return prompt[best_id]["inputs"][widget]
    return None


def parse_caption(template: str, prompt: dict) -> str:
    unresolved = set()

    def repl(m):
        token = m.group(1)
        if token.startswith("date:"):
            return _fmt_date(token[5:])
        if "." in token:
            node, widget = token.split(".", 1)
            val = _lookup_widget(prompt, node.strip(), widget.strip())
            if val is not None:
                return str(val)
        unresolved.add(token)
        return f"%{token}%"

    result = PH_RE.sub(repl, template)
    return result, unresolved


# ───────────────────────── node class
class CaptionBelowImage:
    CATEGORY = "Image/Text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "caption": ("STRING", {"multiline": True, "default": "Your caption…"}),
            },
            "optional": {
                "font_size": (
                    "FLOAT",
                    {
                        "default": 0.07,
                        "min": 0.02,
                        "max": 256.0,
                        "step": 0.01,
                        "precision": 3,
                    },
                ),
                "font_path": ("STRING", {"default": "C:\\Windows\\Fonts\\arial.ttf"}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "background_color": ("STRING", {"default": "#000000"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            # need prompt + node_id for lookup
            "hidden": {"node_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_caption"

    # ─────────── main
    def apply_caption(
        self,
        image,
        caption,
        font_size=0.07,
        font_path="C:\\Windows\\Fonts\\arial.ttf",
        text_color="#FFFFFF",
        background_color="#000000",
        debug=False,
        node_id=None,
        prompt=None,
        **_,
    ):
        raw_caption = caption
        caption, unresolved = parse_caption(raw_caption, prompt or {})

        if debug:
            print(f"[CaptionBelowImage] raw     : {raw_caption}")
            print(f"[CaptionBelowImage] parsed  : {caption}")
            if unresolved:
                print(f"[CaptionBelowImage] unresolved tokens: {sorted(unresolved)}")

        # ── font size
        base = tensor_to_pil(image).convert("RGBA")
        W, H = base.size
        font_px = (
            max(6, int(H * font_size)) if 0 < font_size < 3 else max(6, int(font_size))
        )
        font = _load_font(font_path, font_px)

        # ── wrap
        cpl = max(4, int(W / (font_px * 0.65)))
        lines = []
        for p in caption.splitlines():
            lines.extend(textwrap.wrap(p, cpl, break_long_words=True) or [""])
        tmp = ImageDraw.Draw(base)
        heights = [tmp.textbbox((0, 0), ln, font=font)[3] for ln in lines]
        spacing = int(font_px * 0.2)
        text_h = sum(heights) + spacing * (len(lines) - 1)

        # ── compose
        pad = font_px // 2
        canvas = Image.new("RGBA", (W, H + text_h + pad * 2), background_color)
        canvas.paste(base, (0, 0))

        draw = ImageDraw.Draw(canvas)
        y = H + pad
        for ln, h in zip(lines, heights):
            w = draw.textbbox((0, 0), ln, font=font)[2]
            draw.text(((W - w) // 2, y), ln, font=font, fill=text_color)
            y += h + spacing

        return (pil_to_tensor(canvas),)


# ───────────────────────── register
NODE_CLASS_MAPPINGS = {"CaptionBelowImage": CaptionBelowImage}
NODE_DISPLAY_NAME_MAPPINGS = {"CaptionBelowImage": "Caption Below Image"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
