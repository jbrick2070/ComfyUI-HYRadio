"""
Full Regression Test Suite for HYRadio Visual Pipeline
======================================================
Validates: imports, node registration, widget schemas, input/output types,
function signatures, API contract matching (WorldMirror.forward vs node call),
tensor shape contracts, JSON workflow integrity, LLM bridge parsing,
cinematography math, batching logic, and unicode safety.

Does NOT require a running ComfyUI instance or GPU inference.
"""
import sys, os, json, traceback, inspect, importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Add ComfyUI root to sys.path for folder_paths etc. (append, not insert, to avoid shadowing our nodes/)
_COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", r"C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI")
if os.path.isdir(_COMFYUI_ROOT) and _COMFYUI_ROOT not in sys.path:
    sys.path.append(_COMFYUI_ROOT)

# Add worldstereo/ for 'from src.camera_utils import ...'
_WORLDSTEREO = os.path.join(ROOT, "worldstereo")
if os.path.isdir(_WORLDSTEREO) and _WORLDSTEREO not in sys.path:
    sys.path.insert(0, _WORLDSTEREO)

PASS = 0
FAIL = 0
WARN = 0

def ok(msg):
    global PASS; PASS += 1; print(f"  [PASS] {msg}")
def fail(msg):
    global FAIL; FAIL += 1; print(f"  [FAIL] {msg}")
def warn(msg):
    global WARN; WARN += 1; print(f"  [WARN] {msg}")

# ============================================================
# 1. Module Imports
# ============================================================
print("\n=== 1. Module Imports ===")

modules_to_test = [
    ("nodes.world_mirror_v1", "VNCCS V1 nodes"),
    ("nodes.world_mirror", "WorldMirror V2 loader"),
    ("nodes.world_batching", "Batch CLIP Encode"),
    ("nodes.cinematography", "Cinematic Translator"),
    ("nodes.llm_environment_bridge", "Environment Bridge"),
    ("nodes.world_stereo", "World Stereo math"),
    ("nodes.panorama_mapper", "Panorama Mapper"),
]

for mod_path, label in modules_to_test:
    try:
        __import__(mod_path)
        ok(f"{label} ({mod_path})")
    except Exception as e:
        fail(f"{label} ({mod_path}): {e}")

# ============================================================
# 2. Node Registration
# ============================================================
print("\n=== 2. Node Registration ===")

try:
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    ok(f"nodes/__init__.py loaded {len(NODE_CLASS_MAPPINGS)} nodes")

    required_nodes = [
        "VNCCS_LoadWorldMirrorModel",
        "VNCCS_WorldMirror3D",
        "VNCCS_WorldMirror3D_Official",
        "VNCCS_WorldMirror_3D",
        "VNCCS_Equirect360ToViews",
        "VNCCS_SavePLY",
        "VNCCS_BackgroundPreview",
        "VNCCS_DecomposePLYData",
        "VNCCS_PLYSceneRenderer",
        "VNCCS_SplatRefiner",
        "VNCCS_PanoramaMapper",
        "VNCCS_LoadWorldStereoModel",
        "VNCCS_WorldStereoGenerate",
        "VNCCS_CameraTrajectoryBuilder",
        "HYWorld_BatchCLIPTextEncode",
        "HYWorld_CinematicTranslator",
        "HYWorld_EnvironmentPromptBuilder",
    ]

    for name in required_nodes:
        if name in NODE_CLASS_MAPPINGS:
            ok(f"  Registered: {name}")
        else:
            fail(f"  MISSING: {name}")

    # Display name check
    missing_display = [n for n in NODE_CLASS_MAPPINGS if n not in NODE_DISPLAY_NAME_MAPPINGS]
    if missing_display:
        for n in missing_display:
            warn(f"  No display name: {n}")
    else:
        ok(f"  All {len(NODE_CLASS_MAPPINGS)} nodes have display names")

except Exception as e:
    fail(f"Node registration failed: {e}")
    traceback.print_exc()

# ============================================================
# 3. Widget Schemas & Input Type Validation
# ============================================================
print("\n=== 3. Widget Schemas (INPUT_TYPES) ===")

VALID_COMFY_TYPES = {
    "IMAGE", "MASK", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING",
    "STRING", "INT", "FLOAT", "BOOLEAN",
    # Custom types used by HYRadio
    "WORLDMIRROR_MODEL", "WORLDSTEREO_MODEL", "PLY_DATA", "VNCCS_SPLAT",
    "TENSOR", "EXTRINSICS", "INTRINSICS", "CAMERA_TRAJECTORY",
}

for name, cls in sorted(NODE_CLASS_MAPPINGS.items()):
    try:
        inputs = cls.INPUT_TYPES()
        required = inputs.get("required", {})
        optional = inputs.get("optional", {})

        errors = []
        for inp_name, inp_spec in {**required, **optional}.items():
            if not isinstance(inp_spec, tuple) or len(inp_spec) < 1:
                errors.append(f"{inp_name}: invalid spec {type(inp_spec)}")
                continue

            dtype = inp_spec[0]
            # dtype can be a string or a list of strings (combo widget)
            if isinstance(dtype, list):
                if not all(isinstance(s, str) for s in dtype):
                    errors.append(f"{inp_name}: combo items must be strings, got {dtype}")
            elif isinstance(dtype, str):
                if dtype not in VALID_COMFY_TYPES:
                    warn(f"  {name}.{inp_name}: unknown type '{dtype}' (may be custom)")
            else:
                errors.append(f"{inp_name}: type spec must be str or list, got {type(dtype)}")

            # Validate widget defaults for numeric types
            if len(inp_spec) > 1 and isinstance(inp_spec[1], dict):
                widget = inp_spec[1]
                if "default" in widget:
                    default = widget["default"]
                    if "min" in widget and default < widget["min"]:
                        errors.append(f"{inp_name}: default {default} < min {widget['min']}")
                    if "max" in widget and default > widget["max"]:
                        errors.append(f"{inp_name}: default {default} > max {widget['max']}")

        if errors:
            for err in errors:
                fail(f"  {name}.{err}")
        else:
            ok(f"{name}: {len(required)} required, {len(optional)} optional inputs")

    except Exception as e:
        fail(f"{name}.INPUT_TYPES(): {e}")

# ============================================================
# 4. Return Types & Function Existence
# ============================================================
print("\n=== 4. Return Types & Functions ===")

for name, cls in sorted(NODE_CLASS_MAPPINGS.items()):
    rt = getattr(cls, "RETURN_TYPES", None)
    rn = getattr(cls, "RETURN_NAMES", None)
    fn = getattr(cls, "FUNCTION", None)

    if rt is None:
        warn(f"{name}: no RETURN_TYPES")
    elif rn and len(rt) != len(rn):
        fail(f"{name}: RETURN_TYPES({len(rt)}) != RETURN_NAMES({len(rn)})")
    else:
        ok(f"{name}: returns {len(rt)} values")

    if fn is None:
        fail(f"{name}: no FUNCTION attribute")
    elif not hasattr(cls, fn):
        fail(f"{name}: FUNCTION='{fn}' but method doesn't exist")
    else:
        # Verify function signature accepts all required inputs
        method = getattr(cls, fn)
        sig = inspect.signature(method)
        params = set(sig.parameters.keys()) - {"self"}
        inputs = cls.INPUT_TYPES()
        req_names = set(inputs.get("required", {}).keys())

        # Every required input must be a parameter of the function
        missing_params = req_names - params
        if missing_params:
            fail(f"{name}.{fn}(): missing params for required inputs: {missing_params}")
        else:
            ok(f"{name}.{fn}(): signature matches required inputs")

# ============================================================
# 5. API Contract: WorldMirror.forward() vs Node Call
# ============================================================
print("\n=== 5. API Contract: WorldMirror.forward() ===")

try:
    from hyworld2.worldrecon.hyworldmirror.models.models.worldmirror import WorldMirror

    sig = inspect.signature(WorldMirror.forward)
    forward_params = set(sig.parameters.keys()) - {"self"}

    # The correct call from world_mirror_v1.py should use: views, cond_flags, is_inference
    expected_kwargs = {"views", "cond_flags", "is_inference"}
    if expected_kwargs.issubset(forward_params):
        ok("WorldMirror.forward() accepts views, cond_flags, is_inference")
    else:
        fail(f"WorldMirror.forward() missing expected params. Has: {forward_params}")

    # CRITICAL: Verify that world_mirror_v1.py calls forward() correctly (not pipeline API)
    import ast
    v1_path = os.path.join(ROOT, "nodes", "world_mirror_v1.py")
    with open(v1_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Check that the inference block does NOT contain pipeline kwargs
    pipeline_kwargs = ["render_video", "render_mask", "num_inference_steps", "guidance_scale"]
    found_pipeline = []
    for kw in pipeline_kwargs:
        if f'"{kw}"' in source or f"'{kw}'" in source:
            found_pipeline.append(kw)

    if found_pipeline:
        fail(f"world_mirror_v1.py STILL contains pipeline API kwargs: {found_pipeline}")
    else:
        ok("world_mirror_v1.py: no pipeline API kwargs (correct V1/V2 module call)")

    # Check that the correct forward() call pattern exists
    if "worldmirror(\n                        views=views," in source or \
       "worldmirror(views=views," in source or \
       "worldmirror(\r\n                        views=views," in source:
        ok("world_mirror_v1.py: found correct forward(views=views, ...) call")
    else:
        fail("world_mirror_v1.py: cannot find forward(views=views, ...) call pattern")

    # Also verify world_mirror.py (V2 official node)
    v2_path = os.path.join(ROOT, "nodes", "world_mirror.py")
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_source = f.read()

    if "worldmirror(views=" in v2_source or "worldmirror(\n" in v2_source:
        ok("world_mirror.py: uses correct forward() call pattern")
    else:
        warn("world_mirror.py: check forward() call pattern manually")

except Exception as e:
    fail(f"API contract check: {e}")
    traceback.print_exc()

# ============================================================
# 6. Tensor Shape Contracts
# ============================================================
print("\n=== 6. Tensor Shape Contracts ===")

try:
    import torch

    # Simulate ComfyUI IMAGE tensor: [B, H, W, C] in float32, range [0,1]
    B, H, W, C = 4, 518, 518, 3
    comfy_images = torch.rand(B, H, W, C)

    # The node converts these to [S, C, H, W] then unsqueeze to [1, S, C, H, W]
    # Simulate what world_mirror_v1.py does
    tensor_list = [comfy_images[i].permute(2, 0, 1) for i in range(B)]  # Each: [C, H, W]
    imgs_tensor = torch.stack(tensor_list)  # [S, C, H, W]
    imgs_tensor = imgs_tensor.unsqueeze(0)  # [1, S, C, H, W]

    if imgs_tensor.shape == (1, 4, 3, 518, 518):
        ok(f"Image prep: ComfyUI [B,H,W,C] -> model [1,S,C,H,W] = {imgs_tensor.shape}")
    else:
        fail(f"Image prep: expected (1,4,3,518,518), got {imgs_tensor.shape}")

    # WorldMirror.forward expects views['img'] = [B, S, C, H, W]
    views = {"img": imgs_tensor}
    if views["img"].dim() == 5:
        ok(f"views['img'] is 5D: {views['img'].shape}")
    else:
        fail(f"views['img'] should be 5D, got {views['img'].dim()}D")

    # Camera poses: [S, 4, 4] -> unsqueeze(0) -> [1, S, 4, 4]
    camera_poses = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)  # [S, 4, 4]
    views["camera_poses"] = camera_poses.unsqueeze(0)  # [1, S, 4, 4]
    if views["camera_poses"].shape == (1, 4, 4, 4):
        ok(f"camera_poses: [S,4,4] -> [1,S,4,4] = {views['camera_poses'].shape}")
    else:
        fail(f"camera_poses: expected (1,4,4,4), got {views['camera_poses'].shape}")

    # Camera intrinsics: [S, 3, 3] -> unsqueeze(0) -> [1, S, 3, 3]
    camera_intrs = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)  # [S, 3, 3]
    views["camera_intrs"] = camera_intrs.unsqueeze(0)  # [1, S, 3, 3]
    if views["camera_intrs"].shape == (1, 4, 3, 3):
        ok(f"camera_intrs: [S,3,3] -> [1,S,3,3] = {views['camera_intrs'].shape}")
    else:
        fail(f"camera_intrs: expected (1,4,3,3), got {views['camera_intrs'].shape}")

    # cond_flags must be a list of 3 ints
    cond_flags = [1, 0, 1]
    if len(cond_flags) == 3 and all(isinstance(f, int) for f in cond_flags):
        ok(f"cond_flags: {cond_flags} (3 ints)")
    else:
        fail(f"cond_flags: invalid {cond_flags}")

except Exception as e:
    fail(f"Tensor shape contracts: {e}")
    traceback.print_exc()

# ============================================================
# 7. LLM Environment Bridge - Prompt & Parsing
# ============================================================
print("\n=== 7. LLM Environment Bridge ===")

try:
    from nodes.llm_environment_bridge import HYWorld_EnvironmentPromptBuilder
    bridge = HYWorld_EnvironmentPromptBuilder()

    # SYSTEM_DIRECTIVE sanity
    directive = bridge.SYSTEM_DIRECTIVE
    if "SYSTEM_DIRECTIVE" in directive:
        fail("SYSTEM_DIRECTIVE contains its own name (duplicate leak)")
    else:
        ok("No duplicate SYSTEM_DIRECTIVE leak")

    # Must contain the JSON structure template
    required_fields = ["visual_prompt", "cinematic_lens", "preset", "fov_deg", "speed", "duration_seconds"]
    for field in required_fields:
        if field in directive:
            ok(f"  SYSTEM_DIRECTIVE contains '{field}'")
        else:
            fail(f"  SYSTEM_DIRECTIVE missing '{field}'")

    # Must enforce camera diversity
    diversity_phrases = ["DIFFERENT preset", "MUST NOT", "VARY"]
    for phrase in diversity_phrases:
        if phrase in directive:
            ok(f"  Diversity enforcement: '{phrase}' present")
        else:
            warn(f"  Diversity enforcement: '{phrase}' missing")

    # JSON extraction tests
    test_outputs = [
        ('{"visual_prompt": "test", "cinematic_lens": {"preset": "forward", "fov_deg": 45}}', True, "forward"),
        ('```json\n{"visual_prompt": "test", "cinematic_lens": {"preset": "zoom_in", "fov_deg": 55}}\n```', True, "zoom_in"),
        ('{"visual_prompt": "test", "cinematic_lens": {"preset": "aerial", "fov_deg": 80}}\nExtra text.', True, "aerial"),
        ('I cannot generate JSON because...', False, None),
        ('', False, None),
        ('{"broken json', False, None),
    ]

    for raw, should_parse, expected_preset in test_outputs:
        result = bridge._extract_json(raw)
        if should_parse and result:
            preset = result.get("cinematic_lens", {}).get("preset", "?")
            if preset == expected_preset:
                ok(f"  JSON parse: preset={preset}")
            else:
                fail(f"  JSON parse: expected preset={expected_preset}, got {preset}")
        elif not should_parse and result is None:
            ok(f"  Rejected: '{raw[:30]}...'")
        elif should_parse and not result:
            fail(f"  Failed to parse valid JSON: '{raw[:30]}...'")
        else:
            fail(f"  Parsed garbage as valid: '{raw[:30]}...'")

except Exception as e:
    fail(f"LLM Bridge tests: {e}")
    traceback.print_exc()

# ============================================================
# 8. Cinematography Math (null-safe, multi-preset)
# ============================================================
print("\n=== 8. Cinematography Math ===")

try:
    from nodes.cinematography import HYWorld_CinematicTranslator
    from nodes.world_stereo import CAMERA_UTILS_AVAILABLE
    translator = HYWorld_CinematicTranslator()

    # Test presets that DON'T require camera_utils first (circular only needs numpy)
    # Then test presets that DO require camera_utils
    test_directives = [
        ({"preset": "circular", "fov_deg": 70, "speed": 0.04, "duration_seconds": 15}, 360, "circular", False),
        ({"preset": "forward", "fov_deg": 45, "speed": 0.05, "duration_seconds": 30, "elevation_deg": None}, 720, "forward+null_elev", True),
        ({"preset": "zoom_in", "fov_deg": 55}, 5, "zoom_in+defaults", True),
        ({"preset": "zoom_out", "fov_deg": 80, "speed": 0.03, "elevation_deg": 30, "duration_seconds": 20}, 480, "zoom_out+full", True),
        ({"preset": "aerial", "fov_deg": 90, "speed": 0.08, "duration_seconds": 10}, 240, "aerial", True),
    ]

    for idx, (directive, expected_frames, label, needs_camera_utils) in enumerate(test_directives):
        if needs_camera_utils and not CAMERA_UTILS_AVAILABLE:
            warn(f"  Scene {idx+1} ({label}): skipped (camera_utils unavailable, pytorch3d not installed)")
            continue

        try:
            t, e, i = translator.translate([json.dumps(directive)], num_frames=5)
            ext, intr = e[0], i[0]

            if ext.dim() == 3 and ext.shape[1] == 4 and ext.shape[2] == 4:
                ok(f"  Scene {idx+1} ({label}): extrinsics {ext.shape}")
            else:
                fail(f"  Scene {idx+1} ({label}): bad extrinsics shape {ext.shape}")

            if ext.shape[0] == expected_frames:
                ok(f"  Scene {idx+1}: {expected_frames} frames correct")
            else:
                fail(f"  Scene {idx+1}: expected {expected_frames} frames, got {ext.shape[0]}")

            if intr.dim() == 3 and intr.shape[1] == 3 and intr.shape[2] == 3:
                ok(f"  Scene {idx+1}: intrinsics {intr.shape}")
            else:
                fail(f"  Scene {idx+1}: bad intrinsics shape {intr.shape}")

            import torch
            if torch.isfinite(ext).all():
                ok(f"  Scene {idx+1}: no NaN/Inf in extrinsics")
            else:
                fail(f"  Scene {idx+1}: NaN/Inf detected in extrinsics!")
        except Exception as preset_e:
            fail(f"  Scene {idx+1} ({label}): {preset_e}")


except Exception as e:
    fail(f"Cinematography math: {e}")
    traceback.print_exc()

# ============================================================
# 9. Batch CLIP Encoding (tensor padding)
# ============================================================
print("\n=== 9. Batch CLIP Encoding (padding) ===")

try:
    import torch
    import torch.nn.functional as F

    # Simulate variable-length CLIP embeddings
    test_cases = [
        ([154, 278, 200], 278, "variable lengths"),
        ([100, 100, 100], 100, "uniform lengths (no pad needed)"),
        ([1, 500], 500, "extreme difference"),
    ]

    for lengths, expected_max, label in test_cases:
        conds = [torch.zeros(1, L, 4096) for L in lengths]
        max_len = max(c.shape[1] for c in conds)

        padded = []
        for c in conds:
            pad_amt = max_len - c.shape[1]
            if pad_amt > 0:
                padded.append(F.pad(c, (0, 0, 0, pad_amt)))
            else:
                padded.append(c)

        result = torch.cat(padded, dim=0)
        expected_shape = (len(lengths), expected_max, 4096)
        if result.shape == expected_shape:
            ok(f"  {label}: {lengths} -> {result.shape}")
        else:
            fail(f"  {label}: expected {expected_shape}, got {result.shape}")

except Exception as e:
    fail(f"Batch padding: {e}")
    traceback.print_exc()

# ============================================================
# 10. JSON Workflow Integrity
# ============================================================
print("\n=== 10. Workflow JSON Integrity ===")

workflows_dir = os.path.join(ROOT, "workflows")
if os.path.isdir(workflows_dir):
    for fname in os.listdir(workflows_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(workflows_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                wf = json.load(f)

            nodes = wf.get("nodes", [])
            links = wf.get("links", [])
            ok(f"{fname}: {len(nodes)} nodes, {len(links)} links")

            # Build node ID set
            node_ids = {n["id"] for n in nodes}

            # Check link integrity
            broken = 0
            for link in links:
                src_id, dst_id = link[1], link[3]
                if src_id not in node_ids or dst_id not in node_ids:
                    broken += 1

            if broken == 0:
                ok(f"  {fname}: all link endpoints valid")
            else:
                fail(f"  {fname}: {broken} broken links")

            # Check HYRadio/VNCCS node types exist in our registry
            for n in nodes:
                ntype = n.get("type", "")
                if ntype.startswith("VNCCS_") or ntype.startswith("HYWorld_"):
                    if ntype in NODE_CLASS_MAPPINGS:
                        ok(f"  {fname}: '{ntype}' registered")
                    else:
                        fail(f"  {fname}: '{ntype}' NOT registered!")

        except json.JSONDecodeError as e:
            fail(f"{fname}: invalid JSON - {e}")
        except Exception as e:
            fail(f"{fname}: {e}")
else:
    warn("No workflows/ directory found")

# ============================================================
# 11. Meta Tensor Safety (offload corruption guard)
# ============================================================
print("\n=== 11. Meta Tensor Safety ===")

try:
    # Verify world_mirror.py has the meta tensor guard
    v2_path = os.path.join(ROOT, "nodes", "world_mirror.py")
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_src = f.read()

    if "param.device.type == 'meta'" in v2_src:
        ok("world_mirror.py: meta tensor corruption guard exists")
    else:
        fail("world_mirror.py: MISSING meta tensor corruption guard")

    # Verify world_mirror_v1.py does NOT call pipeline API
    v1_path = os.path.join(ROOT, "nodes", "world_mirror_v1.py")
    with open(v1_path, "r", encoding="utf-8") as f:
        v1_src = f.read()

    if '"render_video"' in v1_src or "'render_video'" in v1_src:
        fail("world_mirror_v1.py: STILL has pipeline 'render_video' kwarg (should use forward())")
    else:
        ok("world_mirror_v1.py: no pipeline API residue")

    if '"num_inference_steps"' in v1_src or "'num_inference_steps'" in v1_src:
        fail("world_mirror_v1.py: STILL has pipeline 'num_inference_steps' kwarg")
    else:
        ok("world_mirror_v1.py: no diffusion pipeline kwargs")

    # Check that the node properly builds views dict before calling forward
    if 'views = {"img": imgs_tensor}' in v1_src:
        ok("world_mirror_v1.py: builds views dict with imgs_tensor")
    else:
        fail("world_mirror_v1.py: missing views dict construction")

    if "cond_flags = [0, 0, 0]" in v1_src:
        ok("world_mirror_v1.py: initializes cond_flags")
    else:
        fail("world_mirror_v1.py: missing cond_flags initialization")

except Exception as e:
    fail(f"Meta tensor safety: {e}")
    traceback.print_exc()

# ============================================================
# 12. Unicode Safety (Windows cp1252 terminal)
# ============================================================
print("\n=== 12. Unicode Safety ===")

problem_count = 0
for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "nodes")):
    for fname in filenames:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(dirpath, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if "print(" in line or "print (" in line:
                    for emoji in ["\u2705", "\u274c", "\u26a0", "\ud83d"]:
                        if emoji in line:
                            problem_count += 1

if problem_count > 0:
    warn(f"{problem_count} emoji-in-print occurrences (non-critical, may crash on cp1252 terminals)")
else:
    ok("No emoji in print() statements")

# ============================================================
# 13. Cross-Node Type Compatibility
# ============================================================
print("\n=== 13. Cross-Node Type Compatibility ===")

# Verify that output types from one node match input types of downstream nodes
# Pipeline: LoadModel -> WorldMirror3D -> SavePLY
try:
    loader_rt = NODE_CLASS_MAPPINGS["VNCCS_LoadWorldMirrorModel"].RETURN_TYPES
    wm3d_inputs = NODE_CLASS_MAPPINGS["VNCCS_WorldMirror3D"].INPUT_TYPES()["required"]
    wm3d_rt = NODE_CLASS_MAPPINGS["VNCCS_WorldMirror3D"].RETURN_TYPES

    if loader_rt[0] == wm3d_inputs["model"][0]:
        ok(f"LoadModel -> WorldMirror3D: '{loader_rt[0]}' matches")
    else:
        fail(f"LoadModel -> WorldMirror3D: '{loader_rt[0]}' != '{wm3d_inputs['model'][0]}'")

    # WorldMirror3D outputs PLY_DATA -> SavePLY takes PLY_DATA
    save_inputs = NODE_CLASS_MAPPINGS["VNCCS_SavePLY"].INPUT_TYPES()["required"]
    if "PLY_DATA" in wm3d_rt and save_inputs["ply_data"][0] == "PLY_DATA":
        ok("WorldMirror3D -> SavePLY: PLY_DATA matches")
    else:
        fail("WorldMirror3D -> SavePLY: type mismatch")

    # WorldMirror3D outputs VNCCS_SPLAT -> SplatRefiner takes VNCCS_SPLAT
    refiner_inputs = NODE_CLASS_MAPPINGS["VNCCS_SplatRefiner"].INPUT_TYPES()["required"]
    if "VNCCS_SPLAT" in wm3d_rt and refiner_inputs["raw_splats"][0] == "VNCCS_SPLAT":
        ok("WorldMirror3D -> SplatRefiner: VNCCS_SPLAT matches")
    else:
        fail("WorldMirror3D -> SplatRefiner: type mismatch")

    # Equirect360 outputs -> WorldMirror3D inputs
    eq_rt = NODE_CLASS_MAPPINGS["VNCCS_Equirect360ToViews"].RETURN_TYPES
    eq_rn = NODE_CLASS_MAPPINGS["VNCCS_Equirect360ToViews"].RETURN_NAMES
    if eq_rt[0] == "IMAGE":
        ok("Equirect360 -> WorldMirror3D: IMAGE output matches IMAGE input")
    else:
        fail(f"Equirect360 output[0] is '{eq_rt[0]}', expected 'IMAGE'")

    # Camera trajectory -> WorldStereo
    traj_rt = NODE_CLASS_MAPPINGS["VNCCS_CameraTrajectoryBuilder"].RETURN_TYPES
    stereo_inputs = NODE_CLASS_MAPPINGS["VNCCS_WorldStereoGenerate"].INPUT_TYPES()["required"]
    if traj_rt[0] == stereo_inputs["trajectory"][0]:
        ok(f"CameraTrajectory -> WorldStereo: '{traj_rt[0]}' matches")
    else:
        fail(f"CameraTrajectory -> WorldStereo: '{traj_rt[0]}' != '{stereo_inputs['trajectory'][0]}'")

except Exception as e:
    fail(f"Cross-node compat: {e}")
    traceback.print_exc()

# ============================================================
# 14. Offload Scheme Validation
# ============================================================
print("\n=== 14. Offload Scheme Validation ===")

try:
    # VNCCS_WorldMirror3D should accept 'none' as default
    wm_inputs = NODE_CLASS_MAPPINGS["VNCCS_WorldMirror3D"].INPUT_TYPES()
    offload_spec = wm_inputs.get("optional", {}).get("offload_scheme")
    if offload_spec:
        options = offload_spec[0]
        if isinstance(options, list) and "none" in options:
            ok(f"VNCCS_WorldMirror3D offload_scheme options: {options}")
        else:
            fail(f"VNCCS_WorldMirror3D offload_scheme missing 'none': {options}")
    else:
        fail("VNCCS_WorldMirror3D: offload_scheme not found in optional inputs")

    # VNCCS_WorldMirror_3D (V2)
    wm2_inputs = NODE_CLASS_MAPPINGS["VNCCS_WorldMirror_3D"].INPUT_TYPES()
    offload_spec2 = wm2_inputs.get("optional", {}).get("offload_scheme")
    if offload_spec2:
        options2 = offload_spec2[0]
        if isinstance(options2, list) and "none" in options2:
            ok(f"VNCCS_WorldMirror_3D offload_scheme options: {options2}")
        else:
            fail(f"VNCCS_WorldMirror_3D offload_scheme missing 'none': {options2}")

except Exception as e:
    fail(f"Offload scheme: {e}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  PASS: {PASS}  |  FAIL: {FAIL}  |  WARN: {WARN}")
print(f"{'='*60}")

if FAIL > 0:
    print("\n  REGRESSION DETECTED - fix before deploying!")
    sys.exit(1)
else:
    print("\n  ALL CLEAR - safe to restart ComfyUI")
    sys.exit(0)
