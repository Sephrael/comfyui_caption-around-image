"""
caption_around_image  –  startup checker
────────────────────────────────────────
• On ComfyUI startup, confirm execution.WORKFLOW_START exporter exists.
• If missing, print a clear WARNING telling the user how to add it.
• Then import the node definitions.
"""

import inspect, importlib, os, textwrap

def _check_timer_patch():
    import execution
    path = inspect.getfile(execution)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    needed = "_ex.WORKFLOW_START = time.perf_counter()"
    if needed not in src:
        banner = textwrap.dedent(f"""
        ╭──────────────────────────────────────────────────────────────╮
        │  WARNING: execution.py is missing the global timer export.  │
        │                                                            │
        │  Add the two lines below inside PromptExecutor.execute():  │
        │                                                            │
        │      import execution as _ex                                │
        │      _ex.WORKFLOW_START = time.perf_counter()              │
        │                                                            │
        │  Your caption panel will show 'ExecutionTime' = N/A until  │
        │  this is present.                                           │
        ╰──────────────────────────────────────────────────────────────╯
        """)
        print(banner)

_check_timer_patch()

# ---- normal node import -----------------------------------------------------
from .caption_around_image import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]