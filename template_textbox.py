import comfy.model_management as model_management
import comfy.sd
import nodes


class templateTextbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "input_value": ("STRING,INT,FLOAT", {"default": ""}),
                "input_value2": ("STRING,INT,FLOAT", {"default": ""}),
                "input_value3": ("STRING,INT,FLOAT", {"default": ""}),
                "input_value4": ("STRING,INT,FLOAT", {"default": ""}),
            },
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Enter text with {input_value} {input_value2} {input_value3} {input_value4}",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, text, **kwargs):
        # Replace variables in text
        output_text = text
        for key in ["input_value", "input_value2", "input_value3", "input_value4"]:
            if key in kwargs and kwargs[key] is not None:
                output_text = output_text.replace(f"{{{key}}}", str(kwargs[key]))
        return (output_text,)


NODE_CLASS_MAPPINGS = {
    "TextboxReferenceNode": templateTextbox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextboxReferenceNode": "Template Textbox",
}
