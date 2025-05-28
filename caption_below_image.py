"""
CaptionBelowImage – auto-font-fit edition
────────────────────────────────────────────────────────────────────────
Adds a caption panel to any side of an image.

 • Placeholder tokens (%date:..., %node.widget%) are expanded first.
 • If every non-blank line is “key: value” the panel becomes a two-column
   table (keys left-aligned, values right-aligned).
 • Font is automatically shrunk—via binary search—until the rendered
   table / text fits these rules:
      1. Initial max-width = image W
      2. If table would exceed image H, rewrap with max-width = 1.5 × W
 • UI controls: position, font-size upper-bound, colours, debug switch.
"""

import os, re, textwrap, numpy as np, torch
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import importlib.resources as pkgres


# ─────────── tensor ⇄ PIL
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0)


# ─────────── font utilities
def _builtin_ttf():
    try:
        return str(pkgres.files("PIL").joinpath("fonts/DejaVuSans.ttf"))
    except Exception:
        return None


def _load_font(path: str, size: int):
    try:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size)
        builtin = _builtin_ttf()
        if builtin:
            return ImageFont.truetype(builtin, size)
    except Exception:
        pass
    return ImageFont.load_default()


# ─────────── placeholder mini-parser
PH_RE = re.compile(r"%([^%]+)%")
_DATE = {
    "yyyy": "%Y",
    "yy": "%y",
    "MM": "%m",
    "dd": "%d",
    "hh": "%H",
    "mm": "%M",
    "ss": "%S",
}


def _fmt_date(spec: str) -> str:
    out = spec
    for k in sorted(_DATE, key=len, reverse=True):
        out = out.replace(k, _DATE[k])
    return datetime.now().strftime(out)


def _canonical(s: str) -> str:
    return s.replace(" ", "").lower()


def _lookup(prompt: dict, node: str, widget: str):
    # numeric id?
    if node.isdigit() and node in prompt:
        return prompt[node]["inputs"].get(widget)
    want = _canonical(node)
    best = None
    for nid, nd in prompt.items():
        if _canonical(nd.get("class_type", "")) == want and widget in nd.get(
            "inputs", {}
        ):
            if best is None or int(nid) > int(best):
                best = nid
    return prompt[best]["inputs"][widget] if best else None


def parse_caption(tpl: str, prompt: dict) -> str:
    def repl(m):
        token = m.group(1)
        if token.startswith("date:"):
            return _fmt_date(token[5:])
        if "." in token:
            n, w = token.split(".", 1)
            val = _lookup(prompt, n.strip(), w.strip())
            if val is not None:
                return str(val)
        return m.group(0)

    return PH_RE.sub(repl, tpl)


# ─────────── measurement helpers
def measure_table(rows, font, gap):
    """Return key-width, value-width, block-height for table rows."""
    if not rows:
        return 0, 0, 0
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    key_w = val_w = 0
    for k, v in rows:
        kw = draw.textbbox((0, 0), k, font=font)[2]
        vw = draw.textbbox((0, 0), v, font=font)[2]
        key_w = max(key_w, kw)
        val_w = max(val_w, vw)
    line_h = draw.textbbox((0, 0), rows[0][0], font=font)[3]
    block_h = len(rows) * line_h + (len(rows) - 1) * int(line_h * 0.2)
    return key_w, val_w, block_h


def measure_lines(lines, font):
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    heights = [draw.textbbox((0, 0), ln, font=font)[3] for ln in lines]
    max_w = max(draw.textbbox((0, 0), ln, font=font)[2] for ln in lines) if lines else 0
    block_h = sum(heights) + int(heights[0] * 0.2) * (len(lines) - 1) if heights else 0
    return max_w, block_h, heights[0] if heights else 0


def best_font_px(
    start_px, min_px, prompt_rows, free_lines, img_W, img_H, font_path, gap, allow_relax
):
    """Binary-search the biggest font that fits."""
    lo, hi, best = min_px, start_px, min_px
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)
        fits = False
        if prompt_rows:  # table mode
            kw, vw, bh = measure_table(prompt_rows, font, gap)
            total_w = kw + gap + vw
            if bh <= img_H and total_w <= img_W:
                fits = True
            elif allow_relax and bh <= img_H and total_w <= int(img_W * 1.5):
                fits = True
        else:  # free-text mode
            max_w0, bh0, lh = measure_lines(free_lines, font)
            if bh0 <= img_H and max_w0 <= img_W:
                fits = True
            elif allow_relax and bh0 <= img_H and max_w0 <= int(img_W * 1.5):
                fits = True
        if fits:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ─────────── node class
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
                "position": (["bottom", "top", "left", "right"], {"default": "bottom"}),
                "font_size": (
                    "FLOAT",
                    {
                        "default": 32.0,
                        "min": 0.02,
                        "max": 256.0,
                        "step": 0.01,
                        "precision": 3,
                    },
                ),
                "font_path": ("STRING", {"default": ""}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "background_color": ("STRING", {"default": "#000000"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_caption"

    # ─────────── main
    def apply_caption(
        self,
        image,
        caption,
        position="bottom",
        font_size=32.0,
        font_path="",
        text_color="#FFFFFF",
        background_color="#000000",
        debug=False,
        prompt=None,
        **_,
    ):
        prompt = prompt or {}
        caption = parse_caption(caption, prompt)

        base = tensor_to_pil(image).convert("RGBA")
        W, H = base.size
        pad = max(2, int(H * 0.02))
        gap = pad

        # Detect table mode
        raw_lines = [l for l in caption.splitlines() if l.strip()]
        table_mode = raw_lines and all(":" in l for l in raw_lines)
        rows = (
            [(k.strip(), v.strip()) for k, v in (ln.split(":", 1) for ln in raw_lines)]
            if table_mode
            else []
        )
        free_lines = raw_lines if not table_mode else []

        # Binary-search best font
        start_px = (
            max(6, int(H * font_size)) if font_size < 3 else max(6, int(font_size))
        )
        best_px = best_font_px(
            start_px, 6, rows, free_lines, W, H, font_path, gap, allow_relax=True
        )
        font = _load_font(font_path, best_px)

        # Measure using final size
        if table_mode:
            kw, vw, block_h = measure_table(rows, font, gap)
            block_w = kw + gap + vw
        else:
            block_w, block_h, line_h = measure_lines(free_lines, font)

        # Re-wrap free text if width relaxed
        if not table_mode and block_h > H and block_w < int(1.5 * W):
            chars = max(4, int((1.5 * W) / (best_px * 0.65)))
            free_lines = []
            for p in caption.splitlines():
                free_lines.extend(
                    textwrap.wrap(p, chars, break_long_words=True) or [""]
                )
            block_w, block_h, line_h = measure_lines(free_lines, font)

        # Panel dimensions
        panel_w = (
            block_w + pad * 2 if position in ("left", "right") else max(block_w, W)
        )
        panel_h = (
            block_h + pad * 2
            if position in ("top", "bottom")
            else max(block_h + pad * 2, H)
        )

        # Canvas size
        if position in ("top", "bottom"):
            canvas_w = panel_w
            canvas_h = H + panel_h
        else:
            canvas_w = W + panel_w
            canvas_h = panel_h

        canvas = Image.new("RGBA", (canvas_w, canvas_h), background_color)

        # Paste original image
        if position == "bottom":
            canvas.paste(base, ((canvas_w - W) // 2, 0))
            txt_x0, txt_y0 = (canvas_w - block_w) // 2, H + pad
        elif position == "top":
            canvas.paste(base, ((canvas_w - W) // 2, panel_h))
            txt_x0, txt_y0 = (canvas_w - block_w) // 2, pad
        elif position == "left":
            canvas.paste(base, (panel_w, (canvas_h - H) // 2))
            txt_x0, txt_y0 = pad, (canvas_h - block_h) // 2
        else:  # right
            canvas.paste(base, (0, (canvas_h - H) // 2))
            txt_x0, txt_y0 = W + pad, (canvas_h - block_h) // 2

        draw = ImageDraw.Draw(canvas)
        y = txt_y0
        spacing = int(best_px * 0.2)
        if table_mode:
            for key, val in rows:
                kw = draw.textbbox((0, 0), key, font=font)[2]
                vw = draw.textbbox((0, 0), val, font=font)[2]
                # key left
                draw.text((txt_x0, y), key, font=font, fill=text_color)
                # value right-aligned
                vx = txt_x0 + (kw + gap + vw) - vw
                draw.text((vx, y), val, font=font, fill=text_color)
                y += best_px + spacing
        else:
            for line in free_lines:
                draw.text((txt_x0, y), line, font=font, fill=text_color)
                y += best_px + spacing

        if debug:
            print(
                f"[CaptionBelowImage] font_px={best_px}, panel={panel_w}×{panel_h}, canvas={canvas_w}×{canvas_h}"
            )

        return (pil_to_tensor(canvas),)


# ─────────── register
NODE_CLASS_MAPPINGS = {"CaptionBelowImage": CaptionBelowImage}
NODE_DISPLAY_NAME_MAPPINGS = {"CaptionBelowImage": "Caption Below Image"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
