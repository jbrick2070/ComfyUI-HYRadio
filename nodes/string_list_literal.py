import json


class HYWorld_StringListLiteral:
    """
    Parse a JSON string literal into a STRING_LIST for injection into
    nodes that expect STRING_LIST input (e.g. HYWorld_CinematicTranslator).
    Enables hardcoding cinematic directives directly in a workflow without
    routing through the LLM bridge.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                }),
            }
        }

    RETURN_TYPES = ("STRING_LIST",)
    RETURN_NAMES = ("string_list",)
    FUNCTION = "parse"
    CATEGORY = "HYWorld/Utilities"

    def parse(self, json_text):
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"[HYWorld_StringListLiteral] JSON parse error: {e}. Returning empty list.")
            return ([],)

        if not isinstance(parsed, list):
            print(f"[HYWorld_StringListLiteral] Expected JSON array, got {type(parsed).__name__}. Returning empty list.")
            return ([],)

        # Coerce each element to a string. Dict/list elements are re-serialized
        # so downstream nodes that json.loads each string still work.
        out = []
        for item in parsed:
            if isinstance(item, str):
                out.append(item)
            else:
                out.append(json.dumps(item))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "HYWorld_StringListLiteral": HYWorld_StringListLiteral,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld_StringListLiteral": "HYWorld String List Literal (JSON)",
}
