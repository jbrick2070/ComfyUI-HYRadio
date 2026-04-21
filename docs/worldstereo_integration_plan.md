# WorldStereo Integration Plan

Status: planning only. No code, no installs, no workflow changes in this commit.

Upstream: https://github.com/FuchengSu/WorldStereo (Tencent-recommended interim release)
Author: Jeffrey Brick
Date: 2026-04-20
Commit under review: head of main branch at this date.

## 1. WorldStereo spec (from upstream README)

### Input format

WorldStereo has two operating modes.

**Single-view camera-control** (the mode HYRadio would use):
- `image.png` -- one perspective reference image
- `prompt.json` -- `{"short caption", "medium caption", "long caption"}`
- `camera.json` -- `{"motion_list", "extrinsic", "intrinsic"}`
- Extrinsics are `[4x4] w2c in the opencv coordinate` (same convention the HYRadio pipeline already uses)

**Multi-trajectory / memory-augmented** (panorama reconstruction mode):
- `start_frame.png` for depth initialization (via MoGe)
- `meta_info.json` with `scene_type = "perspective" | "panorama"`
- Optional `panorama.png` -- triggers VLM single-path inference
- Per-trajectory bundle: `render.mp4`, `render_mask.mp4`, `camera.json`

### Output format

- **Video-generation mode**: multi-view consistent video frames (`N x H x W x 3`)
- **Reconstruction mode**: dense point cloud `.ply`, plus optional "metric-scale depth, surface normals, camera poses, and Gaussian Splat renderings" via the WorldMirror post-processing hook
- Coordinate system: OpenCV w2c throughout
- Dense point cloud is derived from `MoGe depth estimation on the start frame lifts it to a point cloud`, then refined by `feedforward reconstruction via HY-World 2.0 WorldMirror enforces multi-view depth consistency; final global alignment produces a unified point cloud`

### Hardware requirements

README does **not** list VRAM, torch version, or CUDA version explicitly. Indirect evidence:

- Recommended inference invocation is `torchrun --nproc_per_node=8 run_multi_traj.py --fsdp`, i.e. 8-GPU FSDP
- Install recipe uses `python=3.11`, requires `pytorch3d` (stable branch) and `MoGe` from microsoft/MoGe
- No mention of Blackwell / sm_120 / RTX 5000-series
- The optional HY-World 2.0 dependency pulls in heavy WorldMirror bits

However, the HYRadio repo **already has** a partial WorldStereo integration in `nodes/world_stereo.py` (799 lines, three node classes). That file's `VNCCS_LoadWorldStereoModel` tooltips record first-hand VRAM numbers from someone who got each variant running:

| Variant | Transformer size | VRAM floor | Notes |
|---------|------------------|-----------|-------|
| `worldstereo-camera` | 10.9 GB | 16 GB with `model_cpu_offload` | Feasible on 5080 Laptop (16 GB) |
| `worldstereo-memory` | ~22 GB | 24+ GB | Not feasible on 16 GB |
| `worldstereo-memory-dmd` | 34.9 GB distilled | 40+ GB | Not feasible on 16 GB |

Additional base model requirement from `_download_worldstereo_components`:
- Wan2.1-I2V-14B-480P (~40 GB disk) for VAE + T5 + CLIP
- MoGe (~1-2 GB) for monocular depth bootstrapping

### Blackwell / sm_120 compatibility

Not explicitly addressed by upstream. Risk factors:

- `pytorch3d` historically lags on new CUDA arches. Existing node wraps it in `try/except ImportError` with a runtime guard (`PYTORCH3D_AVAILABLE`), so circular-preset cameras degrade gracefully. The forward and zoom presets do not depend on pytorch3d.
- MoGe is a plain torch model -- should work wherever torch ships sm_120 kernels.
- Wan2.1 VAE is bf16 plus T5/CLIP text encoders -- stock torch, should be fine.
- `optimum-quanto` for fp8/fp4 weight quantization is an optional path; bf16 is the default and is what the plan assumes.

Recommendation: smoke-test `worldstereo-camera` + bf16 + `model_cpu_offload` on the 5080 Laptop before adopting. If pytorch3d fails to build, limit camera presets to `forward` / `zoom_in` / `zoom_out` / `custom`.

## 2. WorldStereo output vs V2 input contract

`nodes/per_scene_splats.py::HYWorld_PerSceneSplats.INPUT_TYPES` specifies V2's contract:

```python
required:
  image_batch:        IMAGE  # [N, H, W, C]
  worldmirror_model:  WORLDMIRROR_MODEL
optional:
  target_size:        INT   # default 518, 14-step multiple (DINOv2)
  use_gsplat:         BOOL  # default True -- this is the quality lever
  views_per_scene:    INT   # default 1
  camera_intrinsics:  TENSOR  # [N, 3, 3]
  camera_poses:       TENSOR  # [N, 4, 4]
```

`nodes/world_stereo.py::VNCCS_WorldStereoGenerate` already emits V2-shaped tensors:

```python
RETURN_TYPES = ("IMAGE",        "TENSOR",       "TENSOR")
RETURN_NAMES = ("video_frames", "camera_poses", "camera_intrinsics")
```

- `video_frames`: `[N, H, W, 3]` -- drop-in for `image_batch`
- `camera_poses`: `[N, 4, 4]` c2w -- drop-in for `camera_poses`
- `camera_intrinsics`: `[N, 3, 3]` -- drop-in for `camera_intrinsics`

The existing node's own docstring confirms this: `Outputs video_frames + camera_poses + camera_intrinsics for VNCCS_WorldMirrorV2_3D.` This is a direct, pre-validated wire.

The one conceptual mismatch: WorldStereo takes a **single perspective reference image** (via `_prepare_pipeline_inputs`, which calls `T.ToTensor()(image_pil)`), not an equirectangular panorama. To bridge this, the pipeline either (a) picks one canonical perspective crop from the panorama (front-center, middle-pitch) and hands that to WorldStereo, which then synthesizes trajectory views via MoGe depth + WorldStereo's diffusion, or (b) drops the panorama step entirely and uses a flat hero image as the input.

## 3. Integration options

### Option a: WorldStereo replaces the Equirect360ToViews slicer

```
Panorama -> [pano-to-canonical-view crop] -> WorldStereoGenerate -> PerSceneSplats -> V2 -> CinematicRenderer
```

What changes:
- Node 2 (`VNCCS_Equirect360ToViews`) is replaced by:
  - A small "canonical view extractor" (perspective crop at yaw=0, pitch=0) producing 1 IMAGE
  - `VNCCS_CameraTrajectoryBuilder` producing a trajectory (reuse existing HYWorld trajectory, or a fresh one)
  - `VNCCS_WorldStereoGenerate` producing N views + camera tensors
- Node 4 (`HYWorld_PerSceneSplats`) wires stay identical -- it already accepts `camera_intrinsics` + `camera_poses` as optional inputs and passes them through to V2.

Pros:
- Real parallax via MoGe + WorldStereo diffusion, instead of flat panorama re-projections
- Output tensors already V2-compatible (zero glue code)
- Works on 16 GB VRAM with `worldstereo-camera` + `model_cpu_offload`
- Keeps V2 as the splat-producing backbone Jeffrey already understands

Cons:
- Panorama input becomes semi-advisory: we pick one viewpoint and let WorldStereo hallucinate the rest. Loses 360-degree coverage unless we loop WorldStereo per yaw band and concatenate.
- Inference time: a 20-step diffusion call per scene on a 16 GB card with offload is minutes, not seconds. Not compatible with "fast" iteration workflow.
- Needs Wan2.1 base (~40 GB disk) cached.

### Option b: WorldStereo replaces V2 entirely

```
Panorama -> WorldStereoGenerate (memory mode) -> direct PLY -> CinematicRenderer
```

Upstream README documents WorldStereo in reconstruction mode as producing a `dense point cloud (.ply)` with optional Gaussian Splat renderings via its own WorldMirror hook.

Pros:
- Single model does everything: depth + multi-view + 3D representation
- Skips the V2 reconstruction layer -- simpler graph

Cons:
- Reconstruction mode uses `worldstereo-memory` or `worldstereo-memory-dmd`, which the existing node's own tooltip says need 24 GB and 40 GB VRAM respectively. **Not feasible on 16 GB.**
- WorldStereo internally delegates the Gaussian-splat step to HY-World 2.0 WorldMirror anyway, so we would not actually be removing V2, just burying it under more abstraction.
- `VNCCS_WorldStereoGenerate` as currently coded is single-view camera mode only; reconstruction mode is not wired and would need new node work.

### Option c: WorldStereo runs parallel to V2 for A/B comparison

```
Panorama -> [tee] -> { Equirect360ToViews -> V2 -> CinematicRenderer_A,
                       WorldStereoGenerate -> V2 -> CinematicRenderer_B }
```

Pros:
- Direct quality comparison on one panorama input

Cons:
- Doubles VRAM pressure (V2 gets loaded twice unless shared)
- Doubles wall-clock time per test
- WorldStereo's ~15 GB effective footprint + V2's 2.5 GB + scheduler overhead leaves no room for SD3.5 (7.6 GB) to stay resident. Either SD3.5 unloads, or A/B runs sequentially, which is not really parallel.
- Redundant for the primary goal (getting a reference-quality render); better as a manual test once option a is in place.

## 4. VRAM budget

5080 Laptop, 16 GB VRAM. Current steady-state load on the fast workflow:

| Component | VRAM (typical) |
|-----------|----------------|
| SD3.5 (panorama pre-gen, already unloaded for this stage) | 7.6 GB |
| WorldMirror V2 (`gs=True`) | ~3.5-4 GB |
| Mistral (Kokoro + LLM bridge) | 2.5 GB |
| ComfyUI scheduler + overhead | ~1 GB |

Fast-workflow stage (SD3.5 unloaded, V2 active): ~7 GB used, ~9 GB free.

Adding `worldstereo-camera` with `model_cpu_offload`:
- Transformer (10.9 GB) swaps in/out via offload; peak resident ~11-12 GB
- Wan2.1 VAE (~0.5 GB active) + T5 encoder (swappable, ~2 GB peak)
- MoGe depth bootstrap: 1-2 GB for the preprocessing step, then returned to CPU

Peak VRAM at the WorldStereo step, with V2 **unloaded** (i.e. WorldStereo runs, produces outputs, then V2 gets loaded next): ~12 GB. Fits.

Peak VRAM if V2 is **concurrently resident**: ~15-16 GB. Borderline OOM. Must keep them sequential -- load WorldStereo, run, unload, then load V2.

The existing `VNCCS_LoadWorldStereoModel.load_model` already calls `pipeline.enable_model_cpu_offload()` when offload_mode is default. That handles the swap. But V2 and WorldStereo cannot both be loaded at once. The workflow graph must emit them in sequence and the VRAM guardian (`nodes/vram_guardian.py`) should unload WorldStereo explicitly before V2 runs.

`worldstereo-memory` (22 GB) and `worldstereo-memory-dmd` (34.9 GB) are **not viable** on this hardware. Plan limits to `worldstereo-camera` only.

## 5. Recommendation

**Option a, phased.**

Phase 0 (this commit): plan only. No code.

Phase 1 (after approval): smoke-test `VNCCS_LoadWorldStereoModel` + `VNCCS_WorldStereoGenerate` in isolation on a trivial 4-frame trajectory. Goals:
- Confirm Wan2.1-I2V-14B-480P downloads cleanly (~40 GB disk write)
- Confirm transformer loads on Blackwell + bf16 without kernel errors
- Confirm MoGe runs on Blackwell
- Measure actual VRAM peak and wall-clock time per frame

Phase 2: add a panorama-to-perspective crop node (small, one function). Wire it into a copy of `HYRadio-Visual-Fast.json` as `HYRadio-WorldStereo-Test.json`. Keep original fast workflow unchanged.

Phase 3: render a 48-frame trajectory through the new workflow. Visual compare against the reference Palace render. Decide whether to retire the Equirect360ToViews path in Fast and Full.

Phase 4 (optional): explore looping WorldStereo per yaw band (8 bands x 6 frames = 48) to recover 360-degree coverage, if single-viewpoint hallucination proves too narrow.

### Rationale

- WorldStereo nodes already exist in the repo and already emit V2-ready tensors. The integration surface is **one wiring change in the workflow JSON plus one small preprocessor node**, not a from-scratch build.
- Only the `worldstereo-camera` variant fits 16 GB VRAM. The memory variants are out.
- Option b requires node work HYRadio does not have (reconstruction mode is un-wired) and needs VRAM Jeffrey does not have.
- Option c doubles cost for marginal benefit. Can be done manually once option a ships.
- The pointillist-splat problem in the fast workflow (documented in `yoga_summary.md`) is orthogonal: that is a `gs=False` + placeholder-splat issue in `visual_renderer.py`. Fixing it still helps the Equirect360ToViews path. WorldStereo integration does not depend on it.

### Open questions for Jeffrey

1. Disk budget: is ~55 GB free on the drive holding `models/` for Wan2.1 + WorldStereo-camera + MoGe?
2. Preferred trajectory preset for the first smoke test: `forward` (safe, pytorch3d-optional) or `circular` (more cinematic, needs pytorch3d)?
3. Should Phase 1 smoke-test results gate Phase 2, or run Phase 2 optimistically in parallel?
4. Do you want to keep the Equirect360ToViews-based Fast workflow as a fallback, or retire it once WorldStereo-Fast is validated?
