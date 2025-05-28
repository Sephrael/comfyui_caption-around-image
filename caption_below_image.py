"""
CaptionBelowImage – bordered table & multi-line values
────────────────────────────────────────────────────────────────────────
▸ Two-column table now has cell borders.
▸ Column widths are fixed for every row (key_w, value_w).
▸ Values in column 2 can contain “\n”; the cell height grows and lines
  are right-aligned within that fixed width.
▸ Font auto-shrinks to fit image-side panel as before.
"""

import os, re, textwrap, numpy as np, torch
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import importlib.resources as pkgres


# ───────── tensor ⇄ PIL
def tensor_to_pil(t):  # torch.Tensor → PIL
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(img):  # PIL → torch.Tensor
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0)


# ───────── font helper
def _builtin_ttf():
    try:
        return str(pkgres.files("PIL").joinpath("fonts/DejaVuSans.ttf"))
    except Exception:
        return None


def _load_font(path, size):
    try:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size)
        builtin = _builtin_ttf()
        if builtin:
            return ImageFont.truetype(builtin, size)
    except Exception:
        pass
    return ImageFont.load_default()


# ───────── placeholder mini-parser
PH = re.compile(r"%([^%]+)%")
_DATE = {
    "yyyy": "%Y",
    "yy": "%y",
    "MM": "%m",
    "dd": "%d",
    "hh": "%H",
    "mm": "%M",
    "ss": "%S",
}


def _fmt_date(s):
    for k in sorted(_DATE, key=len, reverse=True):
        s = s.replace(k, _DATE[k])
    return datetime.now().strftime(s)


def _canon(s):
    return s.replace(" ", "").lower()


def _lookup(prompt, node, widget):
    if node.isdigit() and node in prompt:
        return prompt[node]["inputs"].get(widget)
    want = _canon(node)
    best = None
    for nid, nd in prompt.items():
        if _canon(nd.get("class_type", "")) == want and widget in nd.get("inputs", {}):
            if best is None or int(nid) > int(best):
                best = nid
    return prompt[best]["inputs"][widget] if best else None


def parse_caption(tpl, prompt):
    def repl(m):
        tok = m.group(1)
        if tok.startswith("date:"):
            return _fmt_date(tok[5:])
        if "." in tok:
            n, w = tok.split(".", 1)
            v = _lookup(prompt, n.strip(), w.strip())
            if v is not None:
                return str(v)
        return m.group(0)

    return PH.sub(repl, tpl)


# ───────── measurement helpers
def measure_rows(rows, font, gap, max_width=None):
    """rows=[(k,[vlines])]; returns kw,vw,block_h,line_h."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    key_w = value_w = 0
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
    heights = []
    for k, vl in rows:
        kw = draw.textbbox((0, 0), k, font=font)[2]
        vw = max(draw.textbbox((0, 0), v, font=font)[2] for v in vl)
        if max_width and (kw + gap + vw) > max_width:
            # force wrap value lines
            chars = max(4, int(max_width / (line_h * 0.6)))
            new_vl = []
            for v in vl:
                new_vl.extend(textwrap.wrap(v, chars, break_long_words=True) or [""])
            vl[:] = new_vl
            vw = max(draw.textbbox((0, 0), v, font=font)[2] for v in vl)
        key_w = max(key_w, kw)
        value_w = max(value_w, vw)
        heights.append(len(vl) * line_h + (len(vl) - 1) * int(line_h * 0.2))
    block_h = sum(heights) + int(line_h * 0.2) * (len(heights) - 1)
    return key_w, value_w, block_h, line_h, heights


def best_font(start, min_px, rows, img_W, img_H, font_path, gap):
    lo, hi = min_px, start
    best = min_px
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)
        rows_copy = [(k, vl[:]) for k, vl in rows]
        kw, vw, bh, lh, _ = measure_rows(rows_copy, font, gap)
        fits = (bh <= img_H and kw + gap + vw <= img_W) or (
            bh <= img_H and kw + gap + vw <= int(1.5 * img_W)
        )
        if fits:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


# ───────── node
class CaptionBelowImage:
    CATEGORY = "Image/Text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "caption": ("STRING", {"multiline": True, "default": "key: value"}),
            },
            "optional": {
                "position": (["bottom", "top", "left", "right"], {"default": "bottom"}),
                "font_size": ("FLOAT", {"default": 32.0, "min": 0.02, "max": 256.0}),
                "font_path": ("STRING", {"default": ""}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "background_color": ("STRING", {"default": "#000000"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_caption"

    # main
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

        # parse rows (multi-line value allowed via \n)
        raw = [l for l in caption.splitlines() if l.strip()]
        rows = []
        for ln in raw:
            if ":" in ln:
                k, v = ln.split(":", 1)
                rows.append((k.strip(), [x.strip() for x in v.split("\\n")]))
            else:
                rows.append((ln.strip(), [""]))

        start_px = (
            max(6, int(H * font_size)) if font_size < 3 else max(6, int(font_size))
        )
        best_px = best_font(start_px, 6, rows, W, H, font_path, gap)
        font = _load_font(font_path, best_px)

        kw, vw, bh, lh, row_heights = measure_rows(rows, font, gap, max_width=W)
        if bh > H and kw + gap + vw < int(1.5 * W):
            kw, vw, bh, lh, row_heights = measure_rows(
                rows, font, gap, max_width=int(1.5 * W)
            )

        # Panel dimensions (same as before)
        panel_w = (
            kw + gap + vw + pad * 2
            if position in ("left", "right")
            else max(kw + gap + vw, W)
        )
        panel_h = (
            bh + pad * 2 if position in ("top", "bottom") else max(bh + pad * 2, H)
        )

        # ── canvas size (FIXED) ───────────────────────────────────────
        if position in ("left", "right"):
            canvas_w = panel_w + W
            canvas_h = panel_h
        else:  # top / bottom
            canvas_w = panel_w
            canvas_h = H + panel_h
        # ──────────────────────────────────────────────────────────────

        canvas = Image.new("RGBA", (canvas_w, canvas_h), background_color)
        if position == "bottom":
            canvas.paste(base, ((canvas_w - W) // 2, 0))
            txt_x0, txt_y0 = (canvas_w - (kw + gap + vw)) // 2 + pad, H + pad
        elif position == "top":
            canvas.paste(base, ((canvas_w - W) // 2, panel_h))
            txt_x0, txt_y0 = (canvas_w - (kw + gap + vw)) // 2 + pad, pad
        elif position == "left":
            canvas.paste(base, (panel_w, (canvas_h - H) // 2))
            txt_x0, txt_y0 = pad, (canvas_h - bh) // 2 + pad
        else:  # right
            canvas.paste(base, (0, (canvas_h - H) // 2))
            txt_x0, txt_y0 = W + pad, (canvas_h - bh) // 2 + pad

        draw = ImageDraw.Draw(canvas)
        y = txt_y0
        spacing = int(lh * 0.2)
        # draw rows + borders
        for (key, vals), rh in zip(rows, row_heights):
            # cell borders
            y_bottom = y + rh
            x_key_end = txt_x0 + kw
            x_val_end = x_key_end + gap + vw
            # outer border for row
            draw.rectangle(
                [
                    txt_x0 - pad,
                    y - spacing // 2,
                    x_val_end + pad,
                    y_bottom + spacing // 2,
                ],
                outline=text_color,
                width=1,
            )
            # inner vertical line
            draw.line(
                [
                    (x_key_end + gap // 2, y - spacing // 2),
                    (x_key_end + gap // 2, y_bottom + spacing // 2),
                ],
                fill=text_color,
                width=1,
            )
            # key
            draw.text((txt_x0, y), key, font=font, fill=text_color)
            vy = y
            for v in vals:
                vw_line = draw.textbbox((0, 0), v, font=font)[2]
                vx = x_val_end - vw_line
                draw.text((vx, vy), v, font=font, fill=text_color)
                vy += lh + spacing
            y = y_bottom + spacing

        if debug:
            print(
                f"[CaptionBelowImage] font={best_px}, panel={panel_w}×{panel_h}, canvas={canvas_w}×{canvas_h}"
            )

        return (pil_to_tensor(canvas),)


# ───────── register
NODE_CLASS_MAPPINGS = {"CaptionBelowImage": CaptionBelowImage}
NODE_DISPLAY_NAME_MAPPINGS = {"CaptionBelowImage": "Caption Below Image"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
