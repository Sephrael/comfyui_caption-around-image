import comfy.model_management as model_management
import comfy.sd
import nodes


class templateTextbox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "input_value": ("STRING,INT,FLOAT", {"default": ""}),
            },
            "required": {
                "text": (
                    "STRING",
                    {"multiline": True, "default": "Enter text with {input_value}"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "process"
    CATEGORY = "custom"

    def process(self, text, input_value=None):
        # Replace variable in text
        output_text = text
        if input_value is not None:
            output_text = output_text.replace("{input_value}", str(input_value))
        return (output_text,)


NODE_CLASS_MAPPINGS = {
    "TextboxReferenceNode": templateTextbox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextboxReferenceNode": "Template Textbox",
}
