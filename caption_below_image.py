"""
CaptionBelowImageSmart  (2025-05-27)
────────────────────────────────────────────────────────────────────────
▸ Accepts either a single image [1,H,W,3] or an image batch [N,H,W,3].
▸ Applies the caption panel to every frame and returns an IMAGE batch.
▸ Keeps auto-font shrink, placeholder parsing, bordered two-column
  table, multi-line values, width relaxation, side positioning.

Drop this as caption_below_image_smart.py inside your custom_nodes
folder and F5 → Reload Custom Nodes.
"""

import os, re, textwrap, numpy as np, torch
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import importlib.resources as pkgres

# ───────────────────────────────────────── tensor ⇄ PIL
def tensor_to_pil(t):
    if t.ndim == 4: t = t[0]
    arr = (t.clamp(0,1).cpu().numpy()*255).astype(np.uint8).copy()
    return Image.fromarray(arr, mode="RGB")
def pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).float().div(255.0).unsqueeze(0)

# ───────────────────────────────────────── font helper
def _builtin_ttf():
    try: return str(pkgres.files("PIL").joinpath("fonts/DejaVuSans.ttf"))
    except Exception: return None
def _load_font(path,size):
    try:
        if path and os.path.exists(path): return ImageFont.truetype(path,size)
        builtin=_builtin_ttf()
        if builtin: return ImageFont.truetype(builtin,size)
    except Exception: pass
    return ImageFont.load_default()

# ───────────────────────────────────────── placeholder parser
PH = re.compile(r"%([^%]+)%")
_DATE = {"yyyy":"%Y","yy":"%y","MM":"%m","dd":"%d","hh":"%H","mm":"%M","ss":"%S"}
def _fmt_date(s):
    for k in sorted(_DATE,key=len,reverse=True): s=s.replace(k,_DATE[k])
    return datetime.now().strftime(s)
def _canon(s): return s.replace(" ","").lower()
def _lookup(prompt,node,widget):
    if node.isdigit() and node in prompt:
        return prompt[node]["inputs"].get(widget)
    want=_canon(node); best=None
    for nid,nd in prompt.items():
        if _canon(nd.get("class_type",""))==want and widget in nd.get("inputs",{}):
            if best is None or int(nid)>int(best): best=nid
    return prompt[best]["inputs"][widget] if best else None
def parse_caption(tpl,prompt):
    def repl(m):
        tok=m.group(1)
        if tok.startswith("date:"): return _fmt_date(tok[5:])
        if "." in tok:
            n,w=tok.split(".",1)
            v=_lookup(prompt,n.strip(),w.strip())
            if v is not None: return str(v)
        return m.group(0)
    return PH.sub(repl,tpl)

# ───────────────────────────────────────── measurement helpers
def measure_rows(rows,font,gap,max_width=None):
    draw=ImageDraw.Draw(Image.new("RGB",(1,1)))
    key_w=value_w=0
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
        key_w,max_key= max(key_w,kw), kw
        value_w=max(value_w,vw)
        heights.append(len(vl)*line_h+(len(vl)-1)*int(line_h*0.2))
    block_h=sum(heights)+int(line_h*0.2)*(len(rows)-1)
    return key_w,value_w,block_h,line_h,heights

def best_font(start,min_px,rows,W,H,font_path,gap):
    lo,hi=min_px,start; best=min_px
    while lo<=hi:
        mid=(lo+hi)//2
        font=_load_font(font_path,mid)
        kw,vw,bh,_,_=measure_rows([(k,vl[:]) for k,vl in rows],font,gap)
        fits = (bh<=H and kw+gap+vw<=W) or (bh<=H and kw+gap+vw<=int(1.5*W))
        if fits: best,lo=mid,mid+1
        else: hi=mid-1
    return best

# ───────────────────────────────────────── core renderer (single image)
def render_panel(img_tensor, caption, position, font_px, font_path,
                 text_color, background_color):
    base=tensor_to_pil(img_tensor).convert("RGBA")
    W,H = base.size
    pad=max(2,int(H*0.02)); gap=pad

    # parse rows
    raw=[l for l in caption.splitlines() if l.strip()]
    rows=[]
    for ln in raw:
        if ":" in ln:
            k,v=ln.split(":",1); rows.append((k.strip(),[x.strip() for x in v.split("\\n")]))
        else:
            rows.append((ln.strip(),[""]))
    best_px=font_px
    font=_load_font(font_path,best_px)
    kw,vw,bh,lh,row_h=measure_rows(rows,font,gap,max_width=W)
    if bh>H and kw+gap+vw<int(1.5*W):
        kw,vw,bh,lh,row_h=measure_rows(rows,font,gap,max_width=int(1.5*W))
    panel_w = kw+gap+vw+pad*2 if position in("left","right") else max(kw+gap+vw,W)
    panel_h = bh+pad*2 if position in("top","bottom") else max(bh+pad*2,H)
    if position in("left","right"):
        canvas_w,canvas_h = panel_w+W, panel_h
    else:
        canvas_w,canvas_h = panel_w, H+panel_h
    canvas=Image.new("RGBA",(canvas_w,canvas_h),background_color)
    if position=="bottom":
        canvas.paste(base,((canvas_w-W)//2,0)); txt_x0,txt_y0=(canvas_w-(kw+gap+vw))//2+pad,H+pad
    elif position=="top":
        canvas.paste(base,((canvas_w-W)//2,panel_h)); txt_x0,txt_y0=(canvas_w-(kw+gap+vw))//2+pad,pad
    elif position=="left":
        canvas.paste(base,(panel_w,(canvas_h-H)//2)); txt_x0,txt_y0=pad,pad
    else:  # right
        canvas.paste(base,(0,(canvas_h-H)//2)); txt_x0,txt_y0=W+pad,pad

    draw=ImageDraw.Draw(canvas)
    y=txt_y0
    spacing=int(lh*0.2)
    x_key_end=txt_x0+kw
    x_val_end=x_key_end+gap+vw
    for (key,vals),rh in zip(rows,row_h):
        y_bottom=y+rh
        draw.rectangle([txt_x0-pad,y,x_val_end+pad,y_bottom],
                       outline=text_color,width=1)
        draw.line([(x_key_end+gap//2,y),(x_key_end+gap//2,y_bottom)],
                  fill=text_color,width=1)
        draw.text((txt_x0,y),key,font=font,fill=text_color)
        vy=y
        for v in vals:
            vw_line=draw.textbbox((0,0),v,font=font)[2]
            vx=x_val_end-vw_line
            draw.text((vx,vy),v,font=font,fill=text_color)
            vy+=lh+spacing
        y=y_bottom+spacing
    return pil_to_tensor(canvas)

# ───────────────────────────────────────── smart batch wrapper node
class CaptionBelowImageSmart:
    CATEGORY="Image/Text"
    @classmethod
    def INPUT_TYPES(cls):
        base = {
            "images": ("IMAGE", {}),
            "caption": ("STRING", {"multiline": True, "default": "key: value"}),
            "position": (["bottom","top","left","right"], {"default":"right"}),
            "font_size": ("FLOAT", {"default":20,"min":0.02,"max":256.0}),
            "font_path": ("STRING", {"default":"C:\Windows\Fonts\arial.ttf"}),
            "text_color": ("STRING", {"default":"#FFFFFF"}),
            "background_color": ("STRING", {"default":"#000000"}),
            "debug": ("BOOLEAN", {"default":False}),
        }
        return {"required": base, "hidden": {"prompt":"PROMPT"}}
    RETURN_TYPES=("IMAGE",)
    FUNCTION="run"

    def run(self, images, caption, position="bottom", font_size=32.0,
            font_path="", text_color="#FFFFFF", background_color="#000000",
            debug=False, prompt=None, **_):

        prompt = prompt or {}
        parsed_caption = parse_caption(caption, prompt)
        N = images.shape[0]
        # initial upper bound font_px (per image) – use first frame height
        first_h = images[0].shape[0]
        font_px = max(6,int(first_h*font_size)) if font_size<3 else max(6,int(font_size))

        if N == 1:
            out = render_panel(images, parsed_caption, position, font_px,
                               font_path, text_color, background_color)
            if debug: print(f"[CaptionPanel] single frame done.")
            return (out,)
        else:
            panels=[]
            for i in range(N):
                panels.append(render_panel(images[i:i+1],
                                           parsed_caption, position, font_px,
                                           font_path, text_color, background_color))
            if debug: print(f"[CaptionPanel] processed {N} frames.")
            return (torch.cat(panels, dim=0),)

# ───────────────────────────────────────── register
NODE_CLASS_MAPPINGS = {"CaptionBelowImageSmart": CaptionBelowImageSmart}
NODE_DISPLAY_NAME_MAPPINGS = {"CaptionBelowImageSmart": "Caption Below Image (Smart)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]