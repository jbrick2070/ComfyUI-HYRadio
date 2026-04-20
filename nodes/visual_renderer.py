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
    import gsplat as _gsplat_probe
    if hasattr(_gsplat_probe, "_C") and getattr(_gsplat_probe, "_C", None) is not None:
        GSPLAT_AVAILABLE = True
    else:
        GSPLAT_AVAILABLE = False
        print("[HYRadio_CinematicRenderer] gsplat wrapper loaded but C++ backend is None - routing to FastPLYRenderer")
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
        if not isinstance(scene_ply, dict) or "splats" not in scene_ply or scene_ply["splats"] is None:
            # Prefer pts3d_filtered — world_mirror_v2 emits this already flat (N,3) and mask-applied.
            # Fall back to raw pts3d (1,S,H,W,3) + filter_mask only if filtered version is missing.
            pts3d = scene_ply.get("pts3d_filtered") if isinstance(scene_ply, dict) else None
            if pts3d is None and isinstance(scene_ply, dict):
                raw = scene_ply.get("pts3d")
                if raw is not None and isinstance(raw, torch.Tensor):
                    if raw.ndim > 2:
                        raw = raw.reshape(-1, 3)
                    fm = scene_ply.get("filter_mask")
                    if fm is not None and fm.shape[0] == raw.shape[0]:
                        pts3d = raw[fm.bool()]
                    else:
                        pts3d = raw
            
            if pts3d is not None and isinstance(pts3d, torch.Tensor) and pts3d.shape[0] > 0:
                # Build colors from images tensor shaped (1, S, 3, H, W) CHW.
                colors_flat = None
                images = scene_ply.get("images") if isinstance(scene_ply, dict) else None
                if images is not None and isinstance(images, torch.Tensor):
                    if images.ndim == 5:
                        colors_flat = images[0].permute(0, 2, 3, 1).reshape(-1, 3)
                    elif images.ndim == 4 and images.shape[1] == 3:
                        colors_flat = images.permute(0, 2, 3, 1).reshape(-1, 3)
                    elif images.ndim == 4:
                        colors_flat = images.reshape(-1, 3)
                    
                    if colors_flat is not None:
                        if colors_flat.max() > 1.5:
                            colors_flat = colors_flat / 255.0
                        colors_flat = colors_flat.clamp(0, 1)
                        
                        # If colors still includes filtered-out pixels, apply filter_mask.
                        fm = scene_ply.get("filter_mask")
                        if fm is not None and colors_flat.shape[0] == fm.shape[0] and colors_flat.shape[0] != pts3d.shape[0]:
                            colors_flat = colors_flat[fm.bool()]
                
                if colors_flat is None or colors_flat.shape[0] != pts3d.shape[0]:
                    colors_flat = torch.ones((pts3d.shape[0], 3), device=pts3d.device)
                
                P = pts3d.shape[0]
                print(f"[HYRadio_CinematicRenderer] pts3d fallback: {P} points, colors shape {tuple(colors_flat.shape)}")
                scene_ply["splats"] = {
                    "means": pts3d.float(),
                    "colors": colors_flat.to(device=pts3d.device, dtype=torch.float32),
                    "opacities": torch.ones((P, 1), device=pts3d.device),
                    "scales": torch.ones((P, 3), device=pts3d.device) * 0.01,
                    "quats": torch.zeros((P, 4), device=pts3d.device)
                }
                scene_ply["splats"]["quats"][:, 0] = 1.0
            else:
                print("[HYRadio_CinematicRenderer] Missing gsplat or splat data. Generating empty frames.")
                c2ws = scene_traj.get("c2ws")
                total_frames = c2ws.shape[0] if c2ws is not None else 1
                W = int(scene_traj.get("width", 512) * scale)
                H = int(scene_traj.get("height", 512) * scale)
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
        # TODO: Remove once VNCCS nodes stop emitting [1, N, D] shapes.
        # Tracking issue: Upstream VNCCS batch leak
        def force_unbatched(t, name, expected_dims=2):
            if t is None: return None
            # Strip PyTorch batch dimension if VNCCS leaked it [1, N, ...]
            # E.g. [1, N, 3] -> [N, 3]
            if t.dim() > expected_dims and t.shape[0] == 1:
                old_shape = list(t.shape)
                t_sq = t.squeeze(0)
                print(f"[HYRadio_CinematicRenderer] force_unbatched: stripped leading dim from {name}, shape {old_shape} -> {list(t_sq.shape)}")
                return t_sq
            return t

        means_raw = force_unbatched(splats.get("means"), "means", expected_dims=2)
        means = means_raw.to(device) if means_raw is not None else None

        # Scene-aware trajectory clamp — prevents camera from flying past the
        # reconstructed scene. Diameter = bounding-box diagonal of the point cloud.
        if means is not None and means.shape[0] > 0:
            from .cinematography import _validate_trajectory as _vt_clamp
            bb = means.max(dim=0).values - means.min(dim=0).values
            scene_diameter = bb.norm().item()
            # Recenter point cloud to world origin so trajectory presets
            # (which assume scene-at-origin) aim at the geometry instead of
            # flying past it. Bbox size is unchanged, so scene_diameter stays valid.
            bbox_center = means.mean(dim=0)
            means = means - bbox_center
            splats["means"] = means  # keep dict in sync for downstream consumers
            scene_traj = _vt_clamp(scene_traj, scene_diameter=scene_diameter)
            c2ws = scene_traj["c2ws"]   # re-fetch in case translations were scaled
            print(f"[HYRadio_CinematicRenderer] Scene clamp diag: "
                  f"scene_diameter={scene_diameter:.3f}, "
                  f"cam_norm_range=[{c2ws[:,:3,3].norm(dim=-1).min().item():.3f}, "
                  f"{c2ws[:,:3,3].norm(dim=-1).max().item():.3f}], "
                  f"means_bbox_center={means.mean(dim=0).tolist()}, "
                  f"means_bbox_size={(means.max(dim=0).values - means.min(dim=0).values).tolist()}")

        quats_raw = force_unbatched(splats.get("quats"), "quats", expected_dims=2)
        quats = quats_raw.to(device) if quats_raw is not None else None
        
        scales_raw = force_unbatched(splats.get("scales"), "scales", expected_dims=2)
        scales = scales_raw.to(device) if scales_raw is not None else None
        
        opacities_raw = force_unbatched(splats.get("opacities"), "opacities", expected_dims=1)
        opacities = opacities_raw.to(device) if opacities_raw is not None else None
        
        # HY-World 2.0 native key is "sh"; v1 and generic gsplat stacks use "shs".
        # Some variants stash pre-activated color under "features_dc" or "rgb".
        _SH_KEYS = ("shs", "sh", "features_dc", "rgb", "colors")
        shs_t = next(
            (splats[k] for k in _SH_KEYS if k in splats and splats[k] is not None),
            None,
        )

        if shs_t is None:
            raise KeyError(
                f"[HYRadio] No color/SH tensor found in splats. "
                f"Available keys: {list(splats.keys())}"
            )

        # ONE-SHOT DIAGNOSTIC (Requested by QA Review)
        print('\n--- SPLAT KEYS ---')
        print('splats keys:', list(splats.keys()))
        for k_idx, v_idx in splats.items():
            if hasattr(v_idx, 'shape'):
                print(f'  {k_idx}: {tuple(v_idx.shape)} dtype={v_idx.dtype}')

        print('\n--- SH DIAGNOSTIC ---')
        print('shape:', tuple(shs_t.shape))
        print('dtype:', shs_t.dtype)
        print('min:', shs_t.min().item())
        print('max:', shs_t.max().item())
        print('mean:', shs_t.mean().item())
        print('std:', shs_t.std().item())
        print('any negative:', (shs_t < 0).any().item())
        print('---------------------\n')

        shs_raw = force_unbatched(shs_t, "shs", expected_dims=3)
        shs = shs_raw.to(device)
        
        # If the key matched "colors" or "rgb", it might actually be 2D pre-activated RGB.
        colors_raw = force_unbatched(splats.get("colors"), "colors", expected_dims=2)
        colors = colors_raw.to(device) if colors_raw is not None else None
        
        import inspect
        from gsplat import rasterization
        sig = inspect.signature(rasterization).parameters
        is_gsplat_15 = "sh_degree" in sig
        
        # In 1.5.x, we can pass native RGB straight as colors if shs is missing.
        # In <= 1.4, we ALWAYS need shs.
        if shs is None and colors is not None:
            if not is_gsplat_15:
                shs = (colors - 0.5) / 0.28209479177387814
                if len(shs.shape) == 2:
                    shs = shs.unsqueeze(1)
            else:
                # 1.5.x handles native RGB perfectly. We leave shs=None.
                pass
                
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

        # Invert all cameras to w2c simultaneously
        w2cs = torch.linalg.inv(c2ws).contiguous()

        # === HYRADIO SPLAT DIAGNOSTIC ===
        # Prints splat bbox + camera-to-scene distance so we can see whether
        # WorldMirror V2 collapsed the scene into a tiny cluster (zero-baseline
        # trap) and where the cinematic camera sits relative to it.
        print("=== HYRADIO SPLAT DIAGNOSTIC ===")
        print(f"Total splats: {means.shape[0]}")
        if means.shape[0] > 0:
            bbox_min = means.min(dim=0)[0]
            bbox_max = means.max(dim=0)[0]
            bbox_center = (bbox_min + bbox_max) / 2
            scene_diameter = torch.norm(bbox_max - bbox_min).item()
            print(f"BBox min:    {bbox_min.tolist()}")
            print(f"BBox max:    {bbox_max.tolist()}")
            print(f"BBox center: {bbox_center.tolist()}")
            print(f"Scene diameter: {scene_diameter:.4f}")
            try:
                cam_start = torch.linalg.inv(w2cs[0])[:3, 3]
                cam_end = torch.linalg.inv(w2cs[-1])[:3, 3]
                dist_start = torch.norm(cam_start - bbox_center).item()
                dist_end = torch.norm(cam_end - bbox_center).item()
                print(f"Cam start pos: {cam_start.tolist()}")
                print(f"Cam end pos:   {cam_end.tolist()}")
                print(f"Cam start -> scene center: {dist_start:.4f}")
                print(f"Cam end -> scene center:   {dist_end:.4f}")
            except Exception as e:
                print(f"Pose diag failed: {e}")
            # Disambiguate H1 (pose convention flip) vs trajectory-walks-out-the-back.
            # If cam_forward_0 and direction_to_center point roughly the same way,
            # camera is LOOKING AT the scene (recenter fix is right).
            # If opposite, camera is looking AWAY from the scene (H1 flip).
            try:
                cam_forward_0 = torch.linalg.inv(w2cs[0])[:3, 2]  # camera's +Z axis in world
                print(f"Cam forward vector @ frame 0: {cam_forward_0.tolist()}")
                direction_to_center = (bbox_center - cam_start).tolist()
                print(f"Direction from cam to scene center: {direction_to_center}")
            except Exception as e:
                print(f"Forward diag failed: {e}")
        print("================================")

        # Set chunk size to prevent VRAM overflow on output frames (e.g. 32 frames * 1080p * 3 = ~800MB)
        chunk_size = 32 if is_gsplat_15 else 1
        
        for start_idx in range(0, total_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, total_frames)
            
            viewmat_chunk = w2cs[start_idx:end_idx].contiguous() # [C, 4, 4]
            K_chunk = intrs[start_idx:end_idx].contiguous()      # [C, 3, 3]
            
            try:
                raster_kwargs = {
                    "means": means,
                    "quats": quats,
                    "scales": scales,
                    "opacities": opacities_in,
                    "viewmats": viewmat_chunk,
                    "Ks": K_chunk,
                    "width": W,
                    "height": H,
                    "backgrounds": bg_tensor.unsqueeze(0).expand(viewmat_chunk.shape[0], -1).contiguous() if is_gsplat_15 else bg_tensor.unsqueeze(0)
                }
                
                if is_gsplat_15:
                    if shs is not None:
                        raster_kwargs["colors"] = shs
                        raster_kwargs["sh_degree"] = 0
                    else:
                        raster_kwargs["colors"] = colors
                else:
                    raster_kwargs["colors"] = None
                    raster_kwargs["shs"] = shs

                if GSPLAT_AVAILABLE:
                    # Only resolve CameraModelType if gsplat will actually rasterize.
                    # CameraModelType is not top-level re-exported in all gsplat versions;
                    # try the import defensively, fall through to rasterization's default otherwise.
                    try:
                        from gsplat import CameraModelType
                        _CAM_KEYS = ("camera_config", "cameras", "camera_info", "camera")
                        cam_cfg = next(
                            (scene_ply[k] for k in _CAM_KEYS if k in scene_ply and scene_ply[k] is not None),
                            None,
                        )
                        raster_kwargs["camera_model"] = (
                            cam_cfg.CameraModelType
                            if cam_cfg is not None and hasattr(cam_cfg, "CameraModelType")
                            else CameraModelType.PINHOLE
                        )
                    except ImportError:
                        pass
                    render_colors, _, _ = rasterization(**raster_kwargs)
                    img_out = render_colors.clamp(0, 1).cpu().numpy()
                else:
                    # FastPLYRenderer fallback path (no gsplat C++ backend)
                    import sys
                    wm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worldmirror")
                    if wm_dir not in sys.path:
                        sys.path.insert(0, wm_dir)
                    from src.utils.fast_ply_render import FastPLYRenderer
                    fast_renderer = FastPLYRenderer(device)
                    
                    fov_rad = 2 * math.atan2(W / 2, float(K_chunk[0, 0, 0]))
                    fov_deg = math.degrees(fov_rad)
                    bg_col_tuple = (bg_rgb[0], bg_rgb[1], bg_rgb[2])
                    
                    rendered_list = []
                    for b in range(viewmat_chunk.shape[0]):
                        out_t = fast_renderer.render(
                            means=means, colors=colors if colors is not None else shs[:, 0] * 0.28209 + 0.5,
                            opacities=opacities_in, scales=scales, c2w=c2ws[start_idx+b],
                            width=W, height=H, fov_deg=fov_deg, bg_color=bg_col_tuple
                        )
                        rendered_list.append(out_t)
                    render_colors = torch.stack(rendered_list)
                    img_out = render_colors.clamp(0, 1).cpu().numpy()
                
                img_np = (img_out * 255.0).clip(0, 255).astype(np.uint8)
                
                for c_i in range(img_np.shape[0]):
                    Image.fromarray(img_np[c_i]).save(os.path.join(frames_dir, f"frame_{global_fi:05d}.png"))
                    global_fi += 1
                
                del render_colors, img_out, img_np
                
            except Exception as e:
                import traceback
                print(f" [HYRadio_CinematicRenderer] Frames {start_idx}-{end_idx} failed: {e}")
                print(traceback.format_exc())
                for _ in range(end_idx - start_idx):
                    img_np = np.zeros((H, W, 3), dtype=np.uint8)
                    Image.fromarray(img_np).save(os.path.join(frames_dir, f"frame_{global_fi:05d}.png"))
                    global_fi += 1
            
            if start_idx > 0 and start_idx % 128 < chunk_size:
                gc.collect()
                
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
            # Verify at least one frame actually landed on disk before returning success.
            # An empty frames_dir would make downstream ffmpeg silently fall back to base.
            if global_fi <= 0:
                print(f"[CinematicRenderer] EXIT FAIL: 0 frames written to {frames_dir}")
                return ("",)
            print(f"[CinematicRenderer] EXIT OK: {global_fi} frames saved to {frames_pattern}")
            return (frames_pattern,)

        except Exception as e:
            import traceback
            print(f"[CinematicRenderer] EXIT FAIL: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            # Return empty string (not a tensor) so downstream STRING consumer
            # can detect the failure cleanly and emit a loud error instead of
            # silently falling back to base video with no cinematic overlay.
            return ("",)

NODE_CLASS_MAPPINGS = {
    "HYRadio_CinematicRenderer": HYRadio_CinematicRenderer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYRadio_CinematicRenderer": "🎬 Cinematic Renderer (gsplat)"
}
