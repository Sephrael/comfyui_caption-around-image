"""
Caption Around Image (smart) – with optional runtime header
────────────────────────────────────────────────────────────
• Two-column bordered panel can be attached to any edge of an image batch.
• Placeholder lookup by numeric id, class type or node title.
• ‘show_runtime’ adds a green header “ExecutionTime: 12.34 s”
  when execution.WORKFLOW_START exists.
"""
import os, re, textwrap, time, importlib, numpy as np, torch
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import importlib.resources as pkgres

# ───────── tensor ↔ PIL
def tensor_to_pil(t: torch.Tensor):
    if t.ndim == 4: t = t[0]
    a = (t.clamp(0,1).cpu().numpy()*255).astype(np.uint8).copy()
    return Image.fromarray(a, mode="RGB")
def pil_to_tensor(img):
    a = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(a).float().div(255.0).unsqueeze(0)

# ───────── fonts
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

# ───────── placeholder parsing (title-aware)
PH = re.compile(r"%([^%]+)%")
_DATE={"yyyy":"%Y","yy":"%y","MM":"%m","dd":"%d","hh":"%H","mm":"%M","ss":"%S"}
def _fmt_date(s):
    for k in sorted(_DATE,key=len,reverse=True): s=s.replace(k,_DATE[k])
    return datetime.now().strftime(s)
def _canon(s): return s.replace(" ","").lower()
def build_title_index(nodes):
    idx={}
    for n in nodes:
        for k in ("title","label","name"):
            if k in n and n[k]:
                c=_canon(n[k]); idx[c]=max(int(idx.get(c,0)),int(n["id"]))
    return idx
def _lookup(node,widget,prompt,titles):
    if node.isdigit() and node in prompt:
        return prompt[node]["inputs"].get(widget)
    want=_canon(node)
    if want in titles:
        nid=str(titles[want]); return prompt[nid]["inputs"].get(widget)
    best=None
    for nid,p in prompt.items():
        if _canon(p.get("class_type",""))==want and widget in p["inputs"]:
            if best is None or int(nid)>int(best): best=nid
    return prompt[str(best)]["inputs"].get(widget) if best else None
def expand_placeholders(txt,prompt,nodes,debug):
    titles=build_title_index(nodes); unresolved=set()
    def repl(m):
        tok=m.group(1)
        if tok.startswith("date:"): val=_fmt_date(tok[5:])
        elif "." in tok:
            n,w=tok.split(".",1); val=_lookup(n.strip(),w.strip(),prompt,titles)
            if val is None: unresolved.add(tok); return m.group(0)
        else: unresolved.add(tok); return m.group(0)
        if debug: print(f"[CaptionPanel] %{tok}% → {val}")
        return str(val)
    out=PH.sub(repl,txt)
    if debug and unresolved: print("[CaptionPanel] Unresolved:",sorted(unresolved))
    return out

# ───────── measurement helpers
def measure_rows(rows,font,gap,max_width=None):
    d=ImageDraw.Draw(Image.new("RGB",(1,1))); key_w=val_w=0
    lh=d.textbbox((0,0),"Ag",font=font)[3]; heights=[]
    for k,vl in rows:
        kw=d.textbbox((0,0),k,font=font)[2]
        vw=max(d.textbbox((0,0),v,font=font)[2] for v in vl)
        if max_width and kw+gap+vw>max_width:
            chars=max(4,int(max_width/(lh*0.6))); new=[]
            for v in vl: new.extend(textwrap.wrap(v,chars,break_long_words=True) or [""])
            vl[:]=new; vw=max(d.textbbox((0,0),v,font=font)[2] for v in vl)
        key_w=max(key_w,kw); val_w=max(val_w,vw)
        heights.append(len(vl)*lh+(len(vl)-1)*int(lh*0.2))
    block_h=sum(heights)+int(lh*0.2)*(len(rows)-1)
    return key_w,val_w,block_h,lh,heights

# ───────── render
def render_panel(img_t,caption,pos,px,font_path,fg,bg,runtime=None):
    base=tensor_to_pil(img_t).convert("RGBA"); W,H=base.size
    pad=max(2,int(H*0.02)); gap=pad
    raw=[l.strip() for l in caption.splitlines() if l.strip()]
    rows=[(k.strip(),[x.strip() for x in v.split("\\n")]) if ":" in ln else (ln,[""])
          for ln in raw
          for k,v in ([ln.split(":",1)] if ":" in ln else [(ln,"")])]
    font=_load_font(font_path,px)
    kw,vw,bh,lh,rh=measure_rows(rows,font,gap,max_width=W)
    if bh>H and kw+gap+vw<int(1.5*W):
        kw,vw,bh,lh,rh=measure_rows(rows,font,gap,max_width=int(1.5*W))
    panel_w=kw+gap+vw+pad*2 if pos in("left","right") else max(kw+gap+vw,W)
    panel_h=bh+pad*2 if pos in("top","bottom") else max(bh+pad*2,H)
    cw,ch=(panel_w+W,panel_h) if pos in("left","right") else (panel_w,H+panel_h)
    canv=Image.new("RGBA",(cw,ch),bg)
    if pos=="bottom":  canv.paste(base,((cw-W)//2,0));             x0,y0=(cw-(kw+gap+vw))//2+pad,H+pad
    elif pos=="top":   canv.paste(base,((cw-W)//2,panel_h));        x0,y0=(cw-(kw+gap+vw))//2+pad,pad
    elif pos=="left":  canv.paste(base,(panel_w,0));               x0,y0=pad,pad
    else:              canv.paste(base,(0,0));                     x0,y0=W+pad,pad
    d=ImageDraw.Draw(canv); y=y0; spacing=int(lh*0.2)
    xk_end=x0+kw; xv_end=xk_end+gap+vw
    if runtime is not None:
        hdr=f"ExecutionTime: {runtime:.2f}s"
        d.rectangle([x0-pad,y,xv_end+pad,y+lh],outline=fg,width=1)
        d.text((x0,y),hdr,font=font,fill="#00FF00",stroke_width=1,stroke_fill="#00FF00")
        y+=lh+spacing
    for (key,vl),row_h in zip(rows,rh):
        yb=y+row_h
        d.rectangle([x0-pad,y,xv_end+pad,yb],outline=fg,width=1)
        d.line([(xk_end+gap//2,y),(xk_end+gap//2,yb)],fill=fg,width=1)
        d.text((x0,y),key,font=font,fill=fg); vy=y
        for v in vl:
            w=d.textbbox((0,0),v,font=font)[2]
            d.text((xv_end-w,vy),v,font=font,fill=fg); vy+=lh+spacing
        y=yb+spacing
    return pil_to_tensor(canv)

# ───────── node
class CaptionAroundImageSmart:
    CATEGORY="Image/Text"
    @classmethod
    def INPUT_TYPES(cls):
        req={
            "images":("IMAGE",{}),
            "caption":("STRING",{"multiline":True,"default":"ImageOG: %LoadImage.image%"}),
            "position":(["bottom","top","left","right"],{"default":"right"}),
            "font_size":("FLOAT",{"default":20.0,"min":0.02,"max":256.0}),
            "font_path":("STRING",{"default":r"C:\Windows\Fonts\arial.ttf"}),
            "text_color":("STRING",{"default":"#FFFFFF"}),
            "background_color":("STRING",{"default":"#000000"}),
            "show_runtime":("BOOLEAN",{"default":True}),
            "debug":("BOOLEAN",{"default":False}),
        }
        return {"required":req,"hidden":{"prompt":"PROMPT","extra_pnginfo":"EXTRA_PNGINFO"}}
    RETURN_TYPES=("IMAGE",)
    FUNCTION="run"

    def run(self,images,caption,position="right",
            font_size=20.0,font_path="",text_color="#FFFFFF",
            background_color="#000000",show_runtime=True,debug=False,
            prompt=None,extra_pnginfo=None,**_):
        prompt=prompt or {}
        nodes=extra_pnginfo.get("workflow",{}).get("nodes",[]) if extra_pnginfo else []
        caption=expand_placeholders(caption,prompt,nodes,debug)

        runtime=None
        if show_runtime:
            try:
                start=importlib.import_module("execution").WORKFLOW_START
                runtime=time.perf_counter()-start
                if debug: print(f"[CaptionPanel] runtime={runtime:.2f}s")
            except AttributeError:
                print("[CaptionPanel] WARNING: WORKFLOW_START not found; runtime header skipped.")

        N,H=images.shape[0],images.shape[1]
        px=max(6,int(H*font_size)) if font_size<3 else max(6,int(font_size))
        frames=[render_panel(images[i:i+1],caption,position,px,font_path,
                             text_color,background_color,runtime) for i in range(N)]
        return (torch.cat(frames,dim=0),)

# ───────── register
NODE_CLASS_MAPPINGS={"CaptionAroundImageSmart":CaptionAroundImageSmart}
NODE_DISPLAY_NAME_MAPPINGS={"CaptionAroundImageSmart":"Caption Around Image (Smart)"}
__all__=["NODE_CLASS_MAPPINGS","NODE_DISPLAY_NAME_MAPPINGS"]