"""
CaptionBelowImageSmart  (2025-05-28 • title-aware)
────────────────────────────────────────────────────────────────────────
Smart caption panel for single images *or* batches, now with placeholder lookup by node title, label as well as numeric ID or class type.
▸ Accepts IMAGE [N,H,W,3] (N≥1); returns IMAGE batch with caption panel.
▸ Placeholder lookup by:
      • numeric id      – %52.image%
      • class type      – %Load Image.width%
      • custom title    – %MainSampler.steps%  (Set node name for S&R)
▸ Tick ‘debug’ to log every token → console.
All previous features (auto font-shrink, bordered table, batch, etc.) are unchanged.
"""
import os, re, textwrap, numpy as np, torch
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import importlib.resources as pkgres

# ───────── tensor ⇄ PIL
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    arr = (t.clamp(0,1).cpu().numpy()*255).astype(np.uint8).copy()
    return Image.fromarray(arr, mode="RGB")
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0)

# ───────── font helper
def _builtin_ttf():
    try: return str(pkgres.files("PIL").joinpath("fonts/DejaVuSans.ttf"))
    except Exception: return None
def _load_font(path,size):
    try:
        if path and os.path.exists(path):
            return ImageFont.truetype(path,size)
        builtin=_builtin_ttf()
        if builtin:
            return ImageFont.truetype(builtin,size)
    except Exception: pass
    return ImageFont.load_default()

# ───────── placeholder parser (title-aware)
PH = re.compile(r"%([^%]+)%")
_DATE = {"yyyy":"%Y","yy":"%y","MM":"%m","dd":"%d","hh":"%H","mm":"%M","ss":"%S"}
def _fmt_date(spec):
    out=spec
    for k in sorted(_DATE,key=len,reverse=True):
        out=out.replace(k,_DATE[k])
    return datetime.now().strftime(out)
def _canon(s): return s.replace(" ","").lower()

def build_title_index(workflow_nodes):
    """Return {canonical_title: node_id} map choosing highest id per title."""
    idx={}
    for n in workflow_nodes:
        for key in ("title","label","name"):
            if key in n and n[key]:
                c=_canon(n[key])
                if c not in idx or int(n["id"])>int(idx[c]):
                    idx[c]=n["id"]
    return idx

def _lookup(token_node, widget, prompt, title_map):
    """Find widget value by numeric id, title, or class type."""
    # numeric id?
    if token_node.isdigit() and token_node in prompt:
        return prompt[token_node]["inputs"].get(widget)
    want=_canon(token_node)
    # title map first
    if want in title_map:
        nid=title_map[want]
        return prompt[str(nid)]["inputs"].get(widget)
    # fallback: match class_type
    best_id=None
    for nid,p in prompt.items():
        if _canon(p.get("class_type",""))==want and widget in p.get("inputs",{}):
            if best_id is None or int(nid)>int(best_id):
                best_id=nid
    return prompt[str(best_id)]["inputs"].get(widget) if best_id else None

def expand_placeholders(caption, prompt, workflow_nodes, debug):
    title_idx = build_title_index(workflow_nodes)
    unresolved=set()
    def repl(m):
        tok=m.group(1)
        if tok.startswith("date:"):
            val=_fmt_date(tok[5:])
            if debug: print(f"[CaptionPanel] %date → {val}")
            return val
        if "." in tok:
            n,w=tok.split(".",1)
            val=_lookup(n.strip(),w.strip(),prompt,title_idx)
            if val is not None:
                if debug: print(f"[CaptionPanel] %{tok}% → {val}")
                return str(val)
        unresolved.add(tok)
        if debug: print(f"[CaptionPanel] unresolved %{tok}%")
        return m.group(0)
    out=PH.sub(repl,caption)
    if debug and unresolved:
        print(f"[CaptionPanel] Unresolved tokens: {sorted(unresolved)}")
    return out

# ───────── measurement + font-fit helpers
def measure_rows(rows,font,gap,max_width=None):
    draw=ImageDraw.Draw(Image.new("RGB",(1,1)))
    key_w=val_w=0
    line_h=draw.textbbox((0,0),"Ag",font=font)[3]
    heights=[]
    for k,vl in rows:
        kw=draw.textbbox((0,0),k,font=font)[2]
        vw=max(draw.textbbox((0,0),v,font=font)[2] for v in vl)
        if max_width and kw+gap+vw>max_width:
            chars=max(4,int(max_width/(line_h*0.6)))
            new=[]
            for v in vl: new.extend(textwrap.wrap(v,chars,break_long_words=True) or [""])
            vl[:] = new
            vw=max(draw.textbbox((0,0),v,font=font)[2] for v in vl)
        key_w=max(key_w,kw); val_w=max(val_w,vw)
        heights.append(len(vl)*line_h + (len(vl)-1)*int(line_h*0.2))
    block_h=sum(heights)+int(line_h*0.2)*(len(rows)-1)
    return key_w,val_w,block_h,line_h,heights

def best_font(start,min_px,rows,W,H,font_path,gap):
    lo,hi,best=min_px,start,min_px
    while lo<=hi:
        mid=(lo+hi)//2
        font=_load_font(font_path,mid)
        kw,vw,bh,_,_=measure_rows([(k,vl[:]) for k,vl in rows],font,gap)
        fits=(bh<=H and kw+gap+vw<=W) or (bh<=H and kw+gap+vw<=int(1.5*W))
        if fits: best,lo=mid,mid+1
        else: hi=mid-1
    return best

# ───────── single-frame renderer
def render_panel(img_t, caption, position, font_px,
                 font_path, text_color, bg_color):

    base=tensor_to_pil(img_t).convert("RGBA")
    W,H=base.size
    pad=max(2,int(H*0.02)); gap=pad

    # build rows
    rows=[]
    for ln in [l for l in caption.splitlines() if l.strip()]:
        if ":" in ln:
            k,v=ln.split(":",1)
            rows.append((k.strip(),[x.strip() for x in v.split("\\n")]))
        else:
            rows.append((ln.strip(),[""]))

    font=_load_font(font_path,font_px)
    kw,vw,bh,lh,row_h=measure_rows(rows,font,gap,max_width=W)
    if bh>H and kw+gap+vw<int(1.5*W):
        kw,vw,bh,lh,row_h=measure_rows(rows,font,gap,max_width=int(1.5*W))

    panel_w = kw+gap+vw+pad*2 if position in("left","right") else max(kw+gap+vw, W)
    panel_h = bh+pad*2 if position in("top","bottom") else max(bh+pad*2, H)
    if position in ("left","right"):
        canvas_w,canvas_h = panel_w + W, panel_h
    else:
        canvas_w,canvas_h = panel_w, H + panel_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

    # paste image and compute caption origin
    if position=="bottom":
        canvas.paste(base,((canvas_w-W)//2,0))
        txt_x0,txt_y0 = (canvas_w-(kw+gap+vw))//2 + pad, H + pad
    elif position=="top":
        canvas.paste(base,((canvas_w-W)//2,panel_h))
        txt_x0,txt_y0 = (canvas_w-(kw+gap+vw))//2 + pad, pad
    elif position=="left":
        canvas.paste(base,(panel_w,0))
        txt_x0,txt_y0 = pad, pad
    else:  # right
        canvas.paste(base,(0,0))
        txt_x0,txt_y0 = W + pad, pad

    draw=ImageDraw.Draw(canvas)
    y=txt_y0
    spacing=int(lh*0.2)
    x_key_end = txt_x0 + kw
    x_val_end = x_key_end + gap + vw

    for (key,vlines),rh in zip(rows,row_h):
        y_bottom = y + rh
        draw.rectangle([txt_x0-pad, y, x_val_end+pad, y_bottom],
                       outline=text_color, width=1)
        draw.line([(x_key_end + gap//2, y),
                   (x_key_end + gap//2, y_bottom)],
                  fill=text_color, width=1)
        draw.text((txt_x0, y), key, font=font, fill=text_color)
        vy=y
        for v in vlines:
            vw_line = draw.textbbox((0,0), v, font=font)[2]
            draw.text((x_val_end - vw_line, vy), v, font=font, fill=text_color)
            vy += lh + spacing
        y = y_bottom + spacing

    return pil_to_tensor(canvas)

# ───────── smart batch node (title-aware)
class CaptionBelowImageSmart:
    CATEGORY = "Image/Text"

    @classmethod
    def INPUT_TYPES(cls):
        base = {
            "images": ("IMAGE", {}),
            "caption": ("STRING", {"multiline": True, "default": "ImageOG: %LoadImage.image%"}),
            "position": (["bottom","top","left","right"], {"default": "right"}),
            "font_size": ("FLOAT", {"default":20.0,"min":0.02,"max":256.0}),
            "font_path": ("STRING", {"default":"C:\Windows\Fonts\arial.ttf"}),
            "text_color": ("STRING", {"default":"#FFFFFF"}),
            "background_color": ("STRING", {"default":"#000000"}),
            "debug": ("BOOLEAN", {"default":False}),
        }
        return {"required": base,
                "hidden": {"prompt":"PROMPT","extra_pnginfo":"EXTRA_PNGINFO"}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, images, caption, position="bottom",
            font_size=32.0, font_path="", text_color="#FFFFFF",
            background_color="#000000", debug=False,
            prompt=None, extra_pnginfo=None, **_):

        prompt = prompt or {}
        workflow_nodes = extra_pnginfo.get("workflow", {}).get("nodes", []) if extra_pnginfo else []
        parsed_caption = expand_placeholders(caption, prompt, workflow_nodes, debug)

        N, H = images.shape[0], images.shape[1]
        font_px = max(6, int(H * font_size)) if font_size < 3 else max(6, int(font_size))

        if N == 1:
            out = render_panel(images, parsed_caption, position,
                               font_px, font_path, text_color, background_color)
            return (out,)
        frames = [render_panel(images[i:i+1], parsed_caption, position,
                               font_px, font_path, text_color, background_color)
                  for i in range(N)]
        return (torch.cat(frames, dim=0),)

# ───────── register
NODE_CLASS_MAPPINGS = {"CaptionBelowImageSmart": CaptionBelowImageSmart}
NODE_DISPLAY_NAME_MAPPINGS = {"CaptionBelowImageSmart": "Caption Below Image (Smart)"}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]