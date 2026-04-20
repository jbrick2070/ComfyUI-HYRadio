import os


class HYRadio_RenderOutput:
    """Terminal sink for visual renders. Accepts a frames_pattern
    string from CinematicRenderer and logs where frames landed.
    Marks graph as having output so ComfyUI will execute it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_pattern": ("STRING",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "log_output"
    CATEGORY = "HYRadio/Output"

    def log_output(self, frames_pattern):
        # Handle both directory patterns and single strings
        pattern_str = frames_pattern[0] if isinstance(frames_pattern, list) else frames_pattern
        frames_dir = os.path.dirname(pattern_str) if "%" in pattern_str else pattern_str
        if os.path.isdir(frames_dir):
            frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            print(f"[HYRadio_RenderOutput] {frame_count} frames available at: {frames_dir}")
        else:
            print(f"[HYRadio_RenderOutput] frames_pattern: {pattern_str}")
        return ()


NODE_CLASS_MAPPINGS = {
    "HYRadio_RenderOutput": HYRadio_RenderOutput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYRadio_RenderOutput": "HYRadio Render Output (Terminal Sink)",
}
