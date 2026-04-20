import json
import torch
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def _validate_trajectory(traj: dict) -> dict:
    """SE(3) + intrinsics sanity. Auto-repairs common corruptions, logs issues."""
    c2ws = traj.get("c2ws")
    intrs = traj.get("intrs")
    if c2ws is None or intrs is None:
        print("[validate_trajectory] Missing c2ws or intrs — passing through.")
        return traj

    issues = []
    # Shape
    if c2ws.ndim != 3 or tuple(c2ws.shape[-2:]) != (4, 4):
        issues.append(f"c2ws shape {tuple(c2ws.shape)}")
    if intrs.ndim != 3 or tuple(intrs.shape[-2:]) != (3, 3):
        issues.append(f"intrs shape {tuple(intrs.shape)}")
    # NaN/Inf scrub
    if torch.isnan(c2ws).any() or torch.isinf(c2ws).any():
        issues.append("c2ws NaN/Inf scrubbed")
        c2ws = torch.nan_to_num(c2ws, nan=0.0, posinf=1e4, neginf=-1e4)
    if torch.isnan(intrs).any() or torch.isinf(intrs).any():
        issues.append("intrs NaN/Inf scrubbed")
        intrs = torch.nan_to_num(intrs, nan=0.0, posinf=1e4, neginf=-1e4)
    # Positive focal lengths
    if intrs.shape[-2:] == (3, 3):
        fx, fy = intrs[..., 0, 0], intrs[..., 1, 1]
        if (fx <= 0).any() or (fy <= 0).any():
            issues.append(f"non-positive focal: fx_min={fx.min():.2f} fy_min={fy.min():.2f}")
    # c2w bottom row drift
    if c2ws.shape[-2:] == (4, 4):
        bottom = c2ws[..., 3, :]
        expected = torch.tensor([0., 0., 0., 1.], device=bottom.device, dtype=bottom.dtype)
        if not torch.allclose(bottom, expected.expand_as(bottom), atol=1e-3):
            issues.append("c2ws bottom row repaired to [0,0,0,1]")
            c2ws = c2ws.clone()
            c2ws[..., 3, :] = expected

    traj["c2ws"] = c2ws
    traj["intrs"] = intrs
    if issues:
        print(f"[validate_trajectory] {'; '.join(issues)}")
    return traj


class HYWorld_CinematicTranslator:
    """
    Translates the JSON telemetry directives from the LLM into Extrinsics and Intrinsics.
    Provides mathematical tensors compatible with ComfyUI 3D engines and video generators.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cinematic_directives": ("STRING_LIST",),
            },
            "optional": {
                "scene_index": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1, "tooltip": "-1 means translate ALL scenes (native batching). 0+ translates a specific scene."}),
                "num_frames": ("INT", {"default": 25, "min": 4, "max": 81}),
                "image_width": ("INT", {"default": 512}),
                "image_height": ("INT", {"default": 512}),
                "fallback_preset": (["circular", "forward", "zoom_in", "zoom_out", "aerial"], {"default": "forward"}),
            }
        }
    
    RETURN_TYPES = ("CAMERA_TRAJECTORY", "EXTRINSICS", "INTRINSICS")
    RETURN_NAMES = ("trajectory", "extrinsics", "intrinsics")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "translate"
    CATEGORY = "HYWorld/Cinematography"
    
    def translate(self, cinematic_directives, scene_index=-1, num_frames=25, image_width=512, image_height=512, fallback_preset="forward"):
        # Graceful bound checks
        if not cinematic_directives:
            print("[HYWorld_CinematicTranslator] WARNING: empty directives list received.")
            cinematic_directives = [json.dumps({"preset": fallback_preset, "fov_deg": 70.0})]
            
        out_trajectory = []
        out_extrinsics = []
        out_intrinsics = []
        
        loop_indices = range(len(cinematic_directives))
        if scene_index != -1:
            if scene_index >= len(cinematic_directives):
                scene_index = len(cinematic_directives) - 1
            if scene_index < 0:
                scene_index = 0
            loop_indices = [scene_index]
            
        from .world_stereo import _build_trajectory, _c2w_to_w2c

        for idx in loop_indices:
            directive_str = cinematic_directives[idx]
            print(f"[HYWorld_CinematicTranslator] Parsing directive for Scene {idx+1}/{len(cinematic_directives)}: {directive_str}")
            
            try:
                params = json.loads(directive_str)
            except json.JSONDecodeError:
                print("[HYWorld_CinematicTranslator] Failed to parse JSON. Falling back to defaults.")
                params = {}
                
            def _get_float(key, default):
                val = params.get(key)
                return float(val) if val is not None else float(default)
                
            preset = params.get("preset", fallback_preset)
            fov_deg = _get_float("fov_deg", 70.0)
            radius = _get_float("radius", 1.0)
            speed = _get_float("speed", 0.05)
            elevation_deg = _get_float("elevation_deg", 15.0)
            median_depth = _get_float("median_depth", 1.0)
            
            # Dynamic Audio Sync 
            duration = params.get("duration_seconds", None)
            if duration is not None:
                scene_frames = max(4, int(float(duration) * 24))
            else:
                scene_frames = num_frames
                
                print(f" -> Executing math matrix: preset={preset}, FOV={fov_deg}, Frames={scene_frames}")
            
            # Generate the camera arrays
            c2ws, intrs = _build_trajectory(
                preset=preset,
                num_frames=scene_frames,
                radius=radius,
                speed=speed,
                elevation_deg=elevation_deg,
                fov_deg=fov_deg,
                width=image_width,
                height=image_height,
                median_depth=median_depth,
                custom_json="[]"
            )
            
            trajectory = {
                "c2ws": c2ws,
                "intrs": intrs,
                "width": image_width,
                "height": image_height,
            }
            
            trajectory = _validate_trajectory(trajectory)
            
            # Invert C2W to W2C for Extrinsics
            w2cs = _c2w_to_w2c(trajectory["c2ws"])
            
            out_trajectory.append(trajectory)
            out_extrinsics.append(w2cs.cpu().float())
            out_intrinsics.append(intrs.cpu().float())
            
            print(f"[OK] Master Cinematic Batch Complete. Releasing {len(out_trajectory)} payloads downstream.")
        
        return (out_trajectory, out_extrinsics, out_intrinsics)

NODE_CLASS_MAPPINGS = {
    "HYWorld_CinematicTranslator": HYWorld_CinematicTranslator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld_CinematicTranslator": "HYWorld Cinematic Translator"
}
