import torch
import numpy as np
import math
import os
import time
import gc
from PIL import Image

try:
    import folder_paths
    COMFY_TEMP_DIR = folder_paths.get_temp_directory()
except ImportError:
    COMFY_TEMP_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "temp")

try:
    from gsplat.rendering import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False

def _validate_splats(ply: dict) -> dict:
    """Splat tensor consistency. Auto-repairs NaN/Inf, logs issues, passes through."""
    if not isinstance(ply, dict):
        return ply
    splats = ply.get("splats")
    if not isinstance(splats, dict):
        return ply

    required = ["means", "quats", "scales", "opacities"]
    missing = [k for k in required if k not in splats]
    if missing:
        print(f"[validate_splats] Missing required keys: {missing}")
        return ply

    N = splats["means"].shape[0]
    issues = []
    for key in list(splats.keys()):
        t = splats[key]
        if not torch.is_tensor(t):
            continue
        if t.shape[0] != N:
            issues.append(f"{key} len {t.shape[0]} != means {N}")
        if torch.isnan(t).any() or torch.isinf(t).any():
            issues.append(f"{key} NaN/Inf scrubbed")
            splats[key] = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)

    op = splats["opacities"]
    if op.numel() > 0 and (op.min() < -0.01 or op.max() > 1.5):
        issues.append(f"opacity range [{op.min():.2f}, {op.max():.2f}] suspicious")

    if issues:
        print(f"[validate_splats] {'; '.join(issues)}")
    return ply

class HYRadio_CinematicRenderer:
    """
    Renders 3D PLY/Splat data into a 2D cinematic camera sequence using gsplat.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ply_data": ("PLY_DATA",),
                "trajectory": ("CAMERA_TRAJECTORY",),
            },
            "optional": {
                "bg_color": ("STRING", {"default": "0.0,0.0,0.0", "tooltip": "R,G,B float background color (0-1)."}),
                "render_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("frames_pattern", )
    INPUT_IS_LIST = True
    FUNCTION = "render"
    CATEGORY = "HYWorld/Visuals"

    def _render_scene(self, scene_ply, scene_traj, bg_col, scale, device, frames_dir, global_fi):
        if not GSPLAT_AVAILABLE or not isinstance(scene_ply, dict) or "splats" not in scene_ply or scene_ply["splats"] is None:
            print("[HYRadio_CinematicRenderer] Missing gsplat or splat data. Generating empty frames.")
            
            c2ws = scene_traj.get("c2ws")
            total_frames = c2ws.shape[0] if c2ws is not None else 1
            W = int(scene_traj.get("width", 512) * scale)
            H = int(scene_traj.get("height", 512) * scale)
            # Write black frames
            for _ in range(total_frames):
                img_np = np.zeros((H, W, 3), dtype=np.uint8)
                Image.fromarray(img_np).save(os.path.join(frames_dir, f"frame_{global_fi:05d}.png"))
                global_fi += 1
            return global_fi
            
        splats = scene_ply["splats"]
        c2ws = scene_traj["c2ws"]
        intrs = scene_traj["intrs"]
        total_frames = c2ws.shape[0]
        
        W = int(scene_traj.get("width", 512) * scale)
        H = int(scene_traj.get("height", 512) * scale)
        
        means = splats["means"].to(device)
        quats = splats["quats"].to(device)
        scales = splats["scales"].to(device)
        opacities = splats["opacities"].to(device)
        shs = splats["shs"].to(device) if "shs" in splats else None
        colors = splats["colors"].to(device) if "colors" in splats else None
        
        # If colors provided but not SHs, we convert colors to SH degree 0
        if shs is None and colors is not None:
            # SH0 = (RGB - 0.5) / 0.28209...
            shs = (colors - 0.5) / 0.28209479177387814
            # gsplat expects [N, 1, 3] for shs (degree 0) or [N, K, 3]
            if len(shs.shape) == 2:
                shs = shs.unsqueeze(1)
                
        # Hoisted out of loop (BS.8)
        opacities_in = opacities.squeeze(-1) if len(opacities.shape)>1 else opacities
        
        # Pre-scale Intrinsics if scale != 1.0
        if scale != 1.0:
            intrs = intrs.clone()
            intrs[:, 0, 0] *= scale
            intrs[:, 1, 1] *= scale
            intrs[:, 0, 2] *= scale
            intrs[:, 1, 2] *= scale
            
        # Move bulk to device (BS.8)
        c2ws = c2ws.to(device)
        intrs = intrs.to(device)

        bg_rgb = [float(x.strip()) for x in bg_col.split(",")]
        bg_tensor = torch.tensor(bg_rgb, dtype=torch.float32, device=device)
        
        print(f"[HYRadio_CinematicRenderer] Rendering {total_frames} frames via gsplat @ {W}x{H}...")
        
        for fi in range(total_frames):
            c2w = c2ws[fi]
            K = intrs[fi]
            
            # W2C is inverse of C2W
            w2c = torch.linalg.inv(c2w)
            viewmat = w2c.unsqueeze(0) # [1, 4, 4]
            K_in = K.unsqueeze(0)      # [1, 3, 3]
            
            try:
                # Resolve gsplat 1.4 vs 1.5+ signature dynamically
                import inspect
                from gsplat import rasterization
                raster_kwargs = {
                    "means": means,
                    "quats": quats,
                    "scales": scales,
                    "opacities": opacities_in,
                    "viewmats": viewmat,
                    "Ks": K_in,
                    "width": W,
                    "height": H,
                    "backgrounds": bg_tensor.unsqueeze(0)
                }
                
                sig = inspect.signature(rasterization).parameters
                if "sh_degree" in sig:
                    # gsplat 1.5.x+ requires SHs passed as colors with sh_degree
                    raster_kwargs["colors"] = shs
                    raster_kwargs["sh_degree"] = 0
                else:
                    # gsplat <= 1.4.x
                    raster_kwargs["colors"] = None
                    raster_kwargs["shs"] = shs
                
                render_colors, _, _ = rasterization(**raster_kwargs)
                
                # render_colors is [1, H, W, 3] float32 array
                img_out = render_colors[0].clamp(0, 1).cpu()
                img_np = (img_out.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(frames_dir, f"frame_{global_fi:05d}.png"))
                
                del img_out, img_np
                
            except Exception as e:
                # Fallback to zeros if frame fails
                print(f" [HYRadio_CinematicRenderer] Frame {fi} failed: {e}")
                img_np = np.zeros((H, W, 3), dtype=np.uint8)
                Image.fromarray(img_np).save(os.path.join(frames_dir, f"frame_{global_fi:05d}.png"))
            
            if global_fi % 100 == 0:
                gc.collect()
                
            global_fi += 1
                
        return global_fi

    def render(self, ply_data, trajectory, bg_color=["0.0,0.0,0.0"], render_scale=[1.0]):
        from .cinematography import _validate_trajectory # Dynamic import of the validator
        print(f"[HYRadio_CinematicRenderer] ENTER")
        try:
            bg_col = bg_color[0] if isinstance(bg_color, list) else bg_color
            scale = render_scale[0] if isinstance(render_scale, list) else render_scale

            # Trajectory shape check
            if not isinstance(trajectory, list) or not trajectory or \
               not isinstance(trajectory[0], dict):
                print(f"[CinematicRenderer] Invalid trajectory shape: {type(trajectory).__name__}")
                return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)

            # Normalize PLY list, pad if short
            if not isinstance(ply_data, list):
                ply_data = [ply_data]
            if len(ply_data) < len(trajectory):
                print(f"[CinematicRenderer] WARNING: {len(ply_data)} PLY(s) < "
                      f"{len(trajectory)} scene(s). Padding with last PLY.")
                ply_data = ply_data + [ply_data[-1]] * (len(trajectory) - len(ply_data))

            # Per-scene render
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # BS.9 Storage
            session_id = str(int(time.time()))
            frames_dir = os.path.join(COMFY_TEMP_DIR, f"cinematic_frames_{session_id}")
            os.makedirs(frames_dir, exist_ok=True)
            global_fi = 0
            
            for i, (scene_ply, scene_traj) in enumerate(zip(ply_data, trajectory)):
                print(f"[CinematicRenderer] Scene {i+1}/{len(trajectory)}...")
                scene_ply = _validate_splats(scene_ply)          # P2.2
                scene_traj = _validate_trajectory(scene_traj)     # P2.1
                global_fi = self._render_scene(
                    scene_ply, scene_traj, bg_col, scale, device, frames_dir, global_fi
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            frames_pattern = os.path.join(frames_dir, "frame_%05d.png")
            print(f"[CinematicRenderer] EXIT OK: {global_fi} frames saved to {frames_pattern}")
            return (frames_pattern,)

        except Exception as e:
            import traceback
            print(f"[CinematicRenderer] EXIT FAIL: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)

NODE_CLASS_MAPPINGS = {
    "HYRadio_CinematicRenderer": HYRadio_CinematicRenderer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYRadio_CinematicRenderer": "🎬 Cinematic Renderer (gsplat)"
}
