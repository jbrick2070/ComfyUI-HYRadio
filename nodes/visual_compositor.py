import os
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
                "cinematic_frames": ("IMAGE",),
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

    def composite(self, cinematic_frames, base_video_path, fps=24, composite_mode="full_overlay"):
        os.makedirs(COMFY_TEMP_DIR, exist_ok=True)
        os.makedirs(COMFY_OUTPUT_DIR, exist_ok=True)
        
        if not os.path.exists(base_video_path):
            print(f"[HYRadio_VideoCompositor] ERROR: base video not found at {base_video_path}")
            return (base_video_path,)
            
        B, H, W, C = cinematic_frames.shape
        
        # 1. Save tensor frames to a unique temp directory
        session_id = str(int(time.time()))
        frames_dir = os.path.join(COMFY_TEMP_DIR, f"cinematic_frames_{session_id}")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"[HYRadio_VideoCompositor] Saving {B} frames to {frames_dir}...")
        for i in range(B):
            img_tensor = cinematic_frames[i]
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
            
        frames_pattern = os.path.join(frames_dir, "frame_%05d.png")
        
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
        
        print(f"[HYRadio_VideoCompositor] Launching FFMPEG compositor...")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[HYRadio_VideoCompositor] SUCCESS! Saved to {out_path}")
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            print(f"[HYRadio_VideoCompositor] FFMPEG FAILED: {err}")
            return (base_video_path,) # Fallback to original
            
        return {"ui": {"text": [out_path]}, "result": (out_path,)}

NODE_CLASS_MAPPINGS = {
    "HYRadio_VideoCompositor": HYRadio_VideoCompositor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYRadio_VideoCompositor": "🎬 Sequence Compositor (ffmpeg)"
}
