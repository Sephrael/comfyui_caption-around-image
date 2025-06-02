import json

# ─────────────────────────────────────────────────────────────────────────────
# PrintPromptValues – console helper (extended)
# ─────────────────────────────────────────────────────────────────────────────
class PrintPromptValues:
    CATEGORY = "Debug"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always force re-run on every Generate
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "node_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Exact numeric ID of the node in the prompt (ignored if 0)"
                }),
                "node_title": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Exact title/label of the node (ignored if node_id>0)"
                }),
                "widget_name": ("STRING", {
                    "multiline": False,
                    "default": "image",
                    "tooltip": "Name of the widget key you want from the chosen node"
                }),
                "return_selected_node_widgets": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, dump the entire selected node’s JSON instead of one widget"
                }),
                "return_all_node_widgets": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, cycle through ALL nodes and return every widget value"
                }),
                "allowed_float_decimals": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Decimal places for floats"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"

    def run(
        self,
        node_id: int,
        node_title: str,
        widget_name: str,
        return_selected_node_widgets: bool,
        return_all_node_widgets: bool,
        allowed_float_decimals: int,
        prompt: dict = None,
        extra_pnginfo: dict = None,
        unique_id: str = None,
        **_
    ):
        """
        1) If return_all_node_widgets is True:
             → Iterate through every node in `prompt` and collect:
                • inputs dict
                • widget_values dict
                • any other top-level metadata
             → Return a single JSON-string summarizing every node.

        2) Otherwise (return_all_node_widgets is False):
           a) Pick one node by node_id (if >0) or node_title (if non-empty).
           b) If return_selected_node_widgets is True:
                → Dump the entire node’s JSON (pretty-printed, truncated).
              Else:
                → Return exactly one widget value (inputs → widget_values → top-level).
                → Format floats to allowed_float_decimals, append unique_id to force refresh.
        """
        prompt = prompt or {}
        nodes_meta = (extra_pnginfo or {}).get("workflow", {}).get("nodes", [])

        # 1) “dump all nodes” mode
        if return_all_node_widgets:
            all_nodes_output = {}
            for nid, node_dict in prompt.items():
                entry = {}
                # a) inputs
                if "inputs" in node_dict and isinstance(node_dict["inputs"], dict):
                    entry["inputs"] = node_dict["inputs"]
                # b) widget_values
                if "widget_values" in node_dict and isinstance(node_dict["widget_values"], dict):
                    entry["widget_values"] = node_dict["widget_values"]
                # c) any other top-level keys
                extra = {}
                for k, v in node_dict.items():
                    if k not in ("inputs", "widget_values"):
                        extra[k] = v
                if extra:
                    entry["other_metadata"] = extra

                all_nodes_output[nid] = entry

            s = json.dumps(all_nodes_output, indent=2)
            if len(s) > 30000:
                s = s[:30000] + "\n…(truncated)…"
            return (s,)

        # ------------------------------------------------------
        # 2) “single-node” mode
        # ------------------------------------------------------
        def _canon(s: str):
            return s.strip().replace(" ", "").lower()

        # build title_index {canonical_title: max_id}
        title_index = {}
        for n in nodes_meta:
            for key in ("title", "label", "name"):
                if key in n and isinstance(n[key], str):
                    c = _canon(n[key])
                    if c not in title_index or int(n["id"]) > int(title_index[c]):
                        title_index[c] = n["id"]
            props = n.get("properties", {})
            if isinstance(props, dict) and "Node name for S&R" in props:
                nick = props["Node name for S&R"]
                c = _canon(nick)
                if c not in title_index or int(n["id"]) > int(title_index[c]):
                    title_index[c] = n["id"]

        # find chosen_nid (by id or by title)
        chosen_nid = None
        if node_id and str(node_id) in prompt:
            chosen_nid = str(node_id)
        else:
            want = _canon(node_title)
            if want and want in title_index:
                chosen_nid = str(title_index[want])

        if not chosen_nid:
            return (f"[PrintPromptValues] ERROR: No node found for ID={node_id} or title='{node_title}'",)

        node_dict = prompt.get(chosen_nid, {})
        if not node_dict:
            return (f"[PrintPromptValues] ERROR: prompt['{chosen_nid}'] is missing",)

        # 2a) “dump selected node” mode
        if return_selected_node_widgets:
            s = json.dumps(node_dict, indent=2)
            if len(s) > 20000:
                s = s[:20000] + "\n…(truncated)…"
            return (s,)

        # 2b) “return one widget” mode
        val = None
        if "inputs" in node_dict and widget_name in node_dict["inputs"]:
            val = node_dict["inputs"][widget_name]
        elif "widget_values" in node_dict and widget_name in node_dict["widget_values"]:
            val = node_dict["widget_values"][widget_name]
        elif widget_name in node_dict:
            val = node_dict[widget_name]

        if val is None:
            return (f"[PrintPromptValues] WARN: widget '{widget_name}' not found in node {chosen_nid}",)

        if isinstance(val, float):
            fmt = "{:." + str(allowed_float_decimals) + "f}"
            val = fmt.format(val)

        # append unique_id so Preview Any updates each iteration
        tag = f" [{unique_id}]" if unique_id else ""
        return (f"{val}{tag}",)


# ─────────────────────────────────────────────────────────────────────────────
# PromptProvider – now hidden-prompt, no required inputs
# ─────────────────────────────────────────────────────────────────────────────
class PromptProvider:
    CATEGORY = "Debug"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            # Move 'prompt' into the hidden section so ComfyUI injects it automatically
            "hidden": {
                "prompt": "PROMPT",
            }
        }

    # This node simply emits a PROMPT output
    RETURN_TYPES = ("PROMPT",)
    FUNCTION = "run"

    def run(self, prompt: dict = None, **_):
        """
        Just return whatever prompt dict was injected.
        Because INPUT_TYPES hides prompt, you never have to hook anything up.
        """
        return (prompt or {},)


# ─────────────────────────────────────────────────────────────────────────────
# Registration dictionaries (ComfyUI merges these automatically)
# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "PrintPromptValues": PrintPromptValues,
    "PromptProvider": PromptProvider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PrintPromptValues": "Print Prompt Values (Extended)",
    "PromptProvider": "Prompt Provider",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]