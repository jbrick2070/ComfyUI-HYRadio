"""
HYWorld_PerSceneSplats — wrapper node that loops WorldMirror V2 inference
once per input panorama, emitting a LIST of PLY_DATA dicts (one per scene).

Existing V2 (VNCCS_WorldMirrorV2_3D) takes a multi-view batch and produces
one PLY. We invoke it N times with 1 image each for TEST speed; for FULL
pipeline quality, callers can pre-batch multi-view per scene upstream.
"""

import torch
import gc


class HYWorld_PerSceneSplats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),          # [N, H, W, C] — N scenes
                "worldmirror_model": ("WORLDMIRROR_MODEL",),
            },
            "optional": {
                "target_size": ("INT", {"default": 518, "min": 140, "max": 1024, "step": 14}),
                "use_gsplat": ("BOOLEAN", {"default": True}),
                "views_per_scene": ("INT", {"default": 1, "min": 1, "max": 16,
                                            "tooltip": "If >1, consumes this many consecutive images per scene from image_batch."}),
                "camera_intrinsics": ("TENSOR",),
                "camera_poses": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("PLY_DATA",)
    RETURN_NAMES = ("ply_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split_and_splat"
    CATEGORY = "HYWorld/Splats"

    def split_and_splat(self, image_batch, worldmirror_model,
                        target_size=518, use_gsplat=True, views_per_scene=1,
                        camera_intrinsics=None, camera_poses=None):
        from .world_mirror_v2 import VNCCS_WorldMirrorV2_3D
        runner = VNCCS_WorldMirrorV2_3D()

        B = image_batch.shape[0]
        if B % views_per_scene != 0:
            print(f"[PerSceneSplats] WARNING: batch {B} not divisible by "
                  f"views_per_scene={views_per_scene}; trailing images ignored.")
        num_scenes = B // views_per_scene

        ply_list = []
        for i in range(num_scenes):
            start = i * views_per_scene
            stop = start + views_per_scene
            scene_views = image_batch[start:stop]
            scene_intrinsics = camera_intrinsics[start:stop] if camera_intrinsics is not None else None
            scene_poses = camera_poses[start:stop] if camera_poses is not None else None
            
            print(f"[PerSceneSplats] Scene {i+1}/{num_scenes}: "
                  f"{views_per_scene} view(s) -> V2 inference...")
            try:
                result = runner.run_inference(
                    model=worldmirror_model,
                    images=scene_views,
                    target_size=target_size,
                    use_gsplat=use_gsplat,
                    camera_intrinsics=scene_intrinsics,
                    camera_poses=scene_poses
                )
                ply_list.append(result[0])  # PLY_DATA is result[0]
                pts_count = '?'
                if isinstance(result[0], dict):
                    splats = result[0].get('splats')
                    if splats is not None and splats.get('means') is not None:
                        pts_count = splats['means'].shape[0]
                print(f"[PerSceneSplats] Scene {i+1}: PLY ok ({pts_count} pts)")
            except Exception as e:
                import traceback
                print(f"[PerSceneSplats] Scene {i+1} FAILED: {type(e).__name__}: {e}")
                print(traceback.format_exc())
                ply_list.append({"splats": None, "error": str(e)})
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"[PerSceneSplats] Emitted {len(ply_list)} PLY(s).")
        return (ply_list,)


NODE_CLASS_MAPPINGS = {
    "HYWorld_PerSceneSplats": HYWorld_PerSceneSplats,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld_PerSceneSplats": "HYWorld Per-Scene Splats (V2 Loop)",
}
