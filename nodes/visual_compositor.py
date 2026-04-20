import os
import glob
import torch
import numpy as np
import subprocess
import time
from PIL import Image

try:
    import folder_paths
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
    COMFY_TEMP_DIR = folder_paths.get_temp_directory()
except ImportError:
    COMFY_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "output")
    COMFY_TEMP_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "temp")

def _get_ffmpeg_path():
    candidate = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe")
    if os.path.exists(candidate):
        return candidate
    return "ffmpeg"

class HYRadio_VideoCompositor:
    """
    Composites a ComfyUI Image Sequence (e.g. 3D rendered cinematic frames)
    over a baseline procedural video using FFMPEG.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames_pattern": ("STRING", {"forceInput": True}),
                "base_video_path": ("STRING", {"forceInput": True}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "composite_mode": (["full_overlay", "pip_top_right", "pip_bottom_right"], {"default": "full_overlay"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_video_path",)
    OUTPUT_NODE = True
    FUNCTION = "composite"
    CATEGORY = "HYWorld/Visuals"

    def composite(self, frames_pattern, base_video_path, fps=24, composite_mode="full_overlay"):
        os.makedirs(COMFY_OUTPUT_DIR, exist_ok=True)

        # Normalize INPUT_IS_LIST upstream nodes that may wrap single strings in a list.
        if isinstance(frames_pattern, list):
            frames_pattern = frames_pattern[0] if frames_pattern else ""
        if isinstance(base_video_path, list):
            base_video_path = base_video_path[0] if base_video_path else ""

        # ----- base video check -----
        if not isinstance(base_video_path, str) or not base_video_path or not os.path.exists(base_video_path):
            print(f"[HYRadio_VideoCompositor] ABORT: base video missing/invalid: {base_video_path!r}")
            return (str(base_video_path),)

        # ----- frames_pattern check (previously missing) -----
        # Upstream CinematicRenderer returns "" on failure. Also guard against a
        # non-string (old bug returned a tensor) and an empty frames_dir.
        if not isinstance(frames_pattern, str) or not frames_pattern:
            print(f"[HYRadio_VideoCompositor] CINEMATIC FRAMES UNAVAILABLE (frames_pattern={frames_pattern!r}) — "
                  f"returning base video only. The HYWorld cinematic overlay will be MISSING from final output. "
                  f"Check [CinematicRenderer] logs above for the real failure.")
            return (base_video_path,)

        # Expand the %05d pattern to count real files on disk.
        frames_dir = os.path.dirname(frames_pattern)
        if not os.path.isdir(frames_dir):
            print(f"[HYRadio_VideoCompositor] CINEMATIC FRAMES DIR MISSING: {frames_dir} — returning base video only.")
            return (base_video_path,)
        png_count = len(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        if png_count == 0:
            print(f"[HYRadio_VideoCompositor] CINEMATIC FRAMES DIR EMPTY: {frames_dir} — returning base video only.")
            return (base_video_path,)
        print(f"[HYRadio_VideoCompositor] frames_pattern={frames_pattern} ({png_count} PNGs on disk)")
        
        session_id = str(int(time.time()))
        
        # 2. Build FFMPEG command
        ffmpeg_bin = _get_ffmpeg_path()
        out_path = os.path.join(COMFY_OUTPUT_DIR, f"HYRadio_Composited_{session_id}.mp4")
        
        # We take the base video as [0:v][0:a], and the new frames as [1:v].
        # We overlay [1:v] onto [0:v].
        
        # We add -loop 1 to ensure the cinematic shots constantly loop to completely fill 
        # the entire timeframe of the broadcast.
        
        if composite_mode == "full_overlay":
            # Scale cinematic frames to match base video height, preserve aspect ratio, then center
            filter_str = "[1:v]scale=-1:1080[scaled];[0:v][scaled]overlay=x=(W-w)/2:y=(H-h)/2:eof_action=pass[outv]"
        elif composite_mode == "pip_top_right":
            # Scale to 30% and put in top right with 20px padding
            filter_str = "[1:v]scale=iw*0.3:ih*0.3[pip];[0:v][pip]overlay=W-w-20:20:eof_action=pass[outv]"
        elif composite_mode == "pip_bottom_right":
            # Scale to 30% and put in bottom right
            filter_str = "[1:v]scale=iw*0.3:ih*0.3[pip];[0:v][pip]overlay=W-w-20:H-h-20:eof_action=pass[outv]"
        else:
            filter_str = "[0:v][1:v]overlay=eof_action=pass[outv]"
            
        cmd = [
            ffmpeg_bin, "-y",
            "-i", base_video_path,
            "-framerate", str(fps),
            "-loop", "1",
            "-i", frames_pattern,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "0:a?", # Keep original audio
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            "-shortest", 
            out_path
        ]
        
        print(f"[HYRadio_VideoCompositor] Launching FFMPEG compositor ({composite_mode}, {png_count} overlay frames)...")
        try:
            # Capture stdout too — ffmpeg logs its real errors on stderr but we
            # want both streams if something surprising happens.
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Verify the output file was actually written and non-empty before claiming success.
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                err = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
                print(f"[HYRadio_VideoCompositor] FFMPEG claimed success but output is missing/empty at {out_path}")
                print(f"[HYRadio_VideoCompositor] ffmpeg stderr:\n{err}")
                return (base_video_path,)
            print(f"[HYRadio_VideoCompositor] SUCCESS! Saved to {out_path} ({os.path.getsize(out_path)} bytes)")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            print(f"[HYRadio_VideoCompositor] FFMPEG FAILED (returncode={e.returncode}):\n{err}")
            print(f"[HYRadio_VideoCompositor] cmd was: {' '.join(cmd)}")
            return (base_video_path,) # Fallback to original
        except FileNotFoundError as e:
            print(f"[HYRadio_VideoCompositor] FFMPEG BINARY NOT FOUND: {ffmpeg_bin} ({e}) — returning base video only.")
            return (base_video_path,)

        return {"ui": {"text": [out_path]}, "result": (out_path,)}

NODE_CLASS_MAPPINGS = {
    "HYRadio_VideoCompositor": HYRadio_VideoCompositor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYRadio_VideoCompositor": "🎬 Sequence Compositor (ffmpeg)"
}
