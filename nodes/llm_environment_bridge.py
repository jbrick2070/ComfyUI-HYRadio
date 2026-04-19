import torch
import gc
import json
import re

_ALLOWED_PRESETS = ("forward", "zoom_in", "zoom_out", "aerial", "circular")

_LEAK_PREFIXES = re.compile(
    r"^\s*(?:Desperate|Sprawling|Claustrophobic|Epic|Dramatic|Ominous)[^:]*:\s*",
    re.IGNORECASE
)
_LEAK_WORDS = re.compile(
    r"\b(?:MANDATORY(?:\s+elements?)?|REQUIREMENTS?|ELEMENTS?\s*:)\b\s*:?\s*",
    re.IGNORECASE
)
_NUMBERED_ITEMS = re.compile(r"\(\d+\)\s*")

def _strip_leaks(prompt: str) -> str:
    if not prompt:
        return prompt
    original = prompt
    prompt = _LEAK_PREFIXES.sub("", prompt)
    prompt = _LEAK_WORDS.sub("", prompt)
    prompt = _NUMBERED_ITEMS.sub("", prompt)
    prompt = re.sub(r"\s*,\s*,+", ", ", prompt)
    prompt = re.sub(r"\s{2,}", " ", prompt).strip()
    if prompt != original:
        print(f"[bridge] _strip_leaks removed scaffolding from prompt.")
    return prompt

def _clamp_cinematic(lens: dict, used_presets: list) -> dict:
    preset = lens.get("preset")
    if preset not in _ALLOWED_PRESETS:
        unused = [p for p in _ALLOWED_PRESETS if p not in used_presets]
        preset = unused[0] if unused else _ALLOWED_PRESETS[0]
    elif preset in used_presets[-2:]:
        unused = [p for p in _ALLOWED_PRESETS if p not in used_presets]
        if unused:
            print(f"[bridge] Preset collision '{preset}' -> '{unused[0]}'.")
            preset = unused[0]
    lens["preset"] = preset

    def _clamp(key, lo, hi, default):
        try:
            v = float(lens.get(key, default))
        except (TypeError, ValueError):
            v = default
        lens[key] = max(lo, min(hi, v))

    _clamp("fov_deg", 30.0, 120.0, 70.0)
    _clamp("speed", 0.01, 0.10, 0.04)
    _clamp("duration_seconds", 10.0, 60.0, 20.0)
    return lens

def _enforce_diversity(lens: dict, prev_lens: dict | None) -> dict:
    if prev_lens is None:
        return lens
    if abs(lens["fov_deg"] - prev_lens["fov_deg"]) < 15.0:
        target = 90.0 if prev_lens["fov_deg"] < 75.0 else 45.0
        lens["fov_deg"] = max(30.0, min(120.0, target))
        print(f"[bridge] FOV nudged to {lens['fov_deg']} (was too close to previous).")
    if abs(lens["speed"] - prev_lens["speed"]) < 0.02:
        lens["speed"] = 0.08 if prev_lens["speed"] < 0.05 else 0.03
        print(f"[bridge] Speed nudged to {lens['speed']} (was too close to previous).")
    return lens

class HYWorld_EnvironmentPromptBuilder:
    """
    Intelligent Evaluator Node:
    Uses a REAL LLM to analyze each narrative scene and generate optimized
    SD3.5 panoramic prompts + cinematic camera telemetry.
    
    When connected to a live LLM (via model_id dropdown), it sends each scene
    to the model with a cinematography-aware system prompt and parses the
    structured JSON response. Falls back to keyword-based heuristics if the
    LLM is unavailable or returns invalid output.
    
    After generation, aggressively flushes the LLM from VRAM so SD3.5 gets
    the full GPU memory budget.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Accept a JSON string from the HYRadio audio splitter to guarantee 1:1 sync
                "scene_log_json": ("STRING", {"multiline": True, "default": '["Scene 1: Mars", "Scene 2: Moon"]'}),
                "max_words_per_prompt": ("INT", {"default": 60, "min": 10, "max": 150, "step": 5}),
                "model_id": ([
                    "mistralai/Mistral-Nemo-Instruct-2407",
                    "google/gemma-4-E4B-it",
                    "google/gemma-2-9b-it",
                    "google/gemma-2-2b-it",
                    "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
                ], {"default": "mistralai/Mistral-Nemo-Instruct-2407"}),
                "optimization_profile": ([
                    "Pro (Ultra Quality)",
                    "Standard",
                    "Lite (Low VRAM)",
                ], {"default": "Standard"}),
                "audio_duration_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 7200.0, "step": 1.0,
                    "tooltip": "Total audio length in seconds. If > 0, LLM scene durations "
                               "are scaled proportionally to sum to this value. If 0, LLM "
                               "durations are used as-is."
                }),
            },
        }

    RETURN_TYPES = ("STRING_LIST", "STRING_LIST")
    RETURN_NAMES = ("environmental_prompts", "cinematic_directives")
    FUNCTION = "evaluate_and_flush"
    CATEGORY = "HYWorld/LLM"

    # ── System prompt for the visual-optimization LLM call ────────────────
    SYSTEM_DIRECTIVE = (
        "You are an expert environmental prompt engineer and virtual cinematographer "
        "for an AI that generates 360-degree equirectangular panoramas.\n\n"
        "Return ONLY a JSON object with this exact shape — no markdown fences, no prose, no preamble:\n\n"
        "{\n"
        '  "visual_prompt": "<comma-separated keyword prompt for SD3.5>",\n'
        '  "cinematic_lens": { "preset": "<one of forward|zoom_in|zoom_out|aerial|circular>", '
        '"fov_deg": <30-120>, "speed": <0.01-0.10>, "duration_seconds": <10-60> }\n'
        "}\n\n"
        "WORKED EXAMPLE — input and output:\n\n"
        "INPUT SCENE:\n"
        "Abandoned orbital station interior, rotating ring structure, sunlight shafting "
        "through shattered viewports, zero-g debris drifting.\n\n"
        "OUTPUT:\n"
        "{\n"
        '  "visual_prompt": "equirectangular 360 panorama, massive derelict orbital ring '
        'interior, rotational centrifuge architecture curving overhead and underfoot, '
        'hard shafts of unfiltered sunlight through shattered hexagonal viewports, dust '
        'and debris drifting in zero-g, brushed aluminum bulkheads with oxidation blooms, '
        'emergency red strip lighting half-dead, deep atmospheric haze, Kubrick 2001 '
        'symmetry with Cuarón verité grain, no foreground subjects, cinematic masterpiece, 8k",\n'
        '  "cinematic_lens": { "preset": "circular", "fov_deg": 55, "speed": 0.03, '
        '"duration_seconds": 25 }\n'
        "}\n\n"
        "HARD RULES:\n"
        "- Do not use the word MANDATORY. Do not number items (1), (2), (3).\n"
        "- Do not start visual_prompt with an article or a caption label.\n"
        "- Comma-separated keyword phrases only. No full sentences.\n"
        "- No preset, no FOV, no speed can repeat within two scenes of itself.\n"
        "- FOV must differ by at least 15 degrees from the previous scene.\n"
        "- Speed must differ by at least 0.02 from the previous scene.\n"
        "- Preserve director references and material details from the input scene verbatim "
        "where they fit. Add atmospheric particles and depth markers if missing."
    )

    def evaluate_and_flush(self, scene_log_json, max_words_per_prompt, 
                           model_id="mistralai/Mistral-Nemo-Instruct-2407",
                           optimization_profile="Standard",
                           audio_duration_seconds=0.0):
        print("\n[HYWorld_EnvironmentBridge] Syncing with HYRadio Scene Log...")
        
        try:
            scenes = json.loads(scene_log_json)
            if not isinstance(scenes, list):
                scenes = [scene_log_json]
        except json.JSONDecodeError:
            scenes = [scene_log_json]
            
        print(f"[HYWorld_EnvironmentBridge] Detected {len(scenes)} official narrative scenes.")
        print(f"[HYWorld_EnvironmentBridge] LLM: {model_id} ({optimization_profile})")
        
        # ── Load the LLM using the shared infrastructure ─────────────────
        llm_available = False
        try:
            from .story_orchestrator import _generate_with_llm
            llm_available = True
            print("[HYWorld_EnvironmentBridge] LLM inference engine connected.")
        except ImportError as e:
            print(f"[HYWorld_EnvironmentBridge] LLM import failed: {e}")
            print("[HYWorld_EnvironmentBridge] Falling back to keyword heuristics.")
        
        extracted_visuals = []
        extracted_cinematics = []
        
        system_prompt = self.SYSTEM_DIRECTIVE.replace("{max_words}", str(max_words_per_prompt))
        
        for i, scene_text in enumerate(scenes):
            print(f"\n[HYWorld_EnvironmentBridge] Scene {i+1}/{len(scenes)}: {scene_text[:80]}...")
            
            visual_prompt = None
            cinematic_json = None
            
            # ── Attempt real LLM inference ────────────────────────────────
            if llm_available:
                try:
                    # Inject scene context so the LLM knows to vary its output
                    scene_context = f"This is scene {i+1} of {len(scenes)} in the episode."
                    
                    used_so_far = [json.loads(c).get("preset") for c in extracted_cinematics if c]
                    if used_so_far:
                        scene_context += (
                            f" Presets used so far in this episode: {used_so_far}. "
                            f"Unused: {sorted(set(_ALLOWED_PRESETS) - set(used_so_far))}. "
                            "Pick from the unused set. If all five are used, pick the one used longest ago."
                        )
                    
                    user_prompt = f"{system_prompt}\n\n{scene_context}\n\n--- SCENE ---\n{scene_text}\n--- END SCENE ---"
                    
                    raw_output = _generate_with_llm(
                        user_prompt,
                        model_id=model_id.split(" ")[0],  # Strip [ALPHA] etc
                        max_new_tokens=512,
                        temperature=0.75,  # Higher temp for creative diversity
                        top_p=0.92,
                        optimization_profile=optimization_profile,
                    )
                    
                    # Parse the JSON from the LLM response
                    parsed = self._extract_json(raw_output)
                    if parsed:
                        used = [json.loads(c).get("preset", "") for c in extracted_cinematics if c]
                        lens = _clamp_cinematic(parsed.get("cinematic_lens", {}), used)
                        prev = json.loads(extracted_cinematics[-1]) if extracted_cinematics else None
                        lens = _enforce_diversity(lens, prev)
                        visual_prompt = _strip_leaks(parsed.get("visual_prompt", ""))

                        cinematic_json = json.dumps(lens)
                        print(f" [LLM] Visual: {visual_prompt[:80]}...")
                        print(f" [LLM] Camera: {cinematic_json}")
                    else:
                        print(" [LLM] Failed to parse JSON response, using fallback.")
                        
                except Exception as e:
                    print(f" [LLM] Inference error: {e}")
                    print(" [LLM] Falling back to keyword heuristics for this scene.")
            
            # ── Fallback: keyword-based heuristics ────────────────────────
            if visual_prompt is None:
                visual_prompt = self._fallback_visual(scene_text)
            if cinematic_json is None:
                cinematic_json = self._fallback_cinematic(scene_text)
                
            extracted_visuals.append(visual_prompt)
            extracted_cinematics.append(cinematic_json)
            
        # Scale LLM-assigned durations to perfectly match target audio length before flushing
        extracted_cinematics = self._scale_to_audio(extracted_cinematics, audio_duration_seconds)
            
        # ── THE FLUSH: Aggressively reclaim VRAM for SD3.5 ────────────────
        print("\n[HYWorld_EnvironmentBridge] Executing strict VRAM flush...")
        try:
            from .story_orchestrator import _unload_llm
            _unload_llm()
            print("[HYWorld_EnvironmentBridge] LLM unloaded via shared infrastructure.")
        except ImportError:
            pass
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            print("[HYWorld_EnvironmentBridge] VRAM Flush Complete. Releasing synchronized array to SD3.5.")
        
        return (extracted_visuals, extracted_cinematics)
    
    @staticmethod
    def _scale_to_audio(cinematics: list, audio_duration_seconds: float) -> list:
        """Scale LLM-returned durations proportionally so their sum matches audio."""
        if audio_duration_seconds <= 0 or not cinematics:
            return cinematics

        parsed = [json.loads(c) for c in cinematics]
        llm_total = sum(float(p.get("duration_seconds", 0.0)) for p in parsed)
        if llm_total <= 0:
            print("[bridge] _scale_to_audio: LLM returned zero total duration. Skipping.")
            return cinematics

        scale = audio_duration_seconds / llm_total
        print(f"[bridge] Scaling durations by {scale:.2f}x "
              f"(LLM={llm_total:.1f}s -> audio={audio_duration_seconds:.1f}s).")

        out = []
        for p in parsed:
            p["duration_seconds"] = float(p.get("duration_seconds", 0.0)) * scale
            out.append(json.dumps(p))
        return out
        
    # ── JSON extraction from potentially messy LLM output ─────────────────
    @staticmethod
    def _extract_json(text):
        """Extract the first valid JSON object from LLM output."""
        if not text:
            return None
            
        def try_parse(s):
            try: return json.loads(s.strip())
            except json.JSONDecodeError: pass
            try: return json.loads(s.strip() + "}")
            except json.JSONDecodeError: pass
            try: return json.loads(s.strip() + "}}")
            except json.JSONDecodeError: return None
            
        # Try direct parse first
        res = try_parse(text)
        if res: return res
        # Try extracting from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            res = try_parse(match.group(1))
            if res: return res
        # Try finding first { ... } block
        match = re.search(r'\{[^{}]*"visual_prompt"[^{}]*\}', text, re.DOTALL)
        if match:
            res = try_parse(match.group(0))
            if res: return res
        # Try finding nested JSON with cinematic_lens
        match = re.search(r'\{.*?"visual_prompt".*?"cinematic_lens".*?\}[\s]*\}', text, re.DOTALL)
        if match:
            res = try_parse(match.group(0))
            if res: return res
        
        # Super-lenient fallback for truncated json (no closing brace matched by regex):
        match = re.search(r'\{.*?"visual_prompt".*?"cinematic_lens".*', text, re.DOTALL)
        if match:
            res = try_parse(match.group(0))
            if res: return res
            
        return None
    
    # ── Keyword-based fallback for visual prompts ─────────────────────────
    @staticmethod
    def _fallback_visual(scene_text):
        """Generate a visual prompt using keyword analysis when LLM is unavailable."""
        spatial = "equirectangular 360 panorama, massive scale, deep environmental spatial depth, no extreme close-up foreground objects, unobstructed distant horizon"
        
        text_lower = scene_text.lower()
        if any(w in text_lower for w in ["dark", "tunnel", "narrow", "emergency", "escape"]):
            style = "dramatic chiaroscuro lighting, Ridley Scott aesthetic, deep shadows"
        elif any(w in text_lower for w in ["lab", "science", "experiment", "mutation", "contain"]):
            style = "clinical cold fluorescent lighting, Denis Villeneuve color palette, sterile atmosphere"
        elif any(w in text_lower for w in ["control", "command", "station", "facility"]):
            style = "warm amber instrument glow, Kubrickian symmetry, technological grandeur"
        else:
            style = "cinematic volumetric lighting, atmospheric haze, moody ambience"
        
        return f"A sweeping environmental landscape: {scene_text} | {spatial}, {style}, masterpiece, 8k resolution, photorealistic, highly detailed"
    
    # ── Keyword-based fallback for camera telemetry ───────────────────────
    @staticmethod
    def _fallback_cinematic(scene_text):
        """Generate camera telemetry using keyword analysis when LLM is unavailable."""
        text_lower = scene_text.lower()
        
        if any(w in text_lower for w in ["tunnel", "corridor", "narrow", "access", "crawl"]):
            profile = {"preset": "forward", "fov_deg": 40.0, "speed": 0.02, "duration_seconds": 20.0}
        elif any(w in text_lower for w in ["lab", "experiment", "containment", "mutation", "bacteria"]):
            profile = {"preset": "zoom_in", "fov_deg": 65.0, "speed": 0.05, "radius": 1.2, "duration_seconds": 30.0}
        elif any(w in text_lower for w in ["vast", "facility", "underground", "labyrinth", "massive"]):
            profile = {"preset": "forward", "fov_deg": 90.0, "speed": 0.03, "duration_seconds": 45.0}
        elif any(w in text_lower for w in ["escape", "chase", "urgent", "emergency", "run"]):
            profile = {"preset": "forward", "fov_deg": 35.0, "speed": 0.08, "duration_seconds": 15.0}
        elif any(w in text_lower for w in ["sky", "city", "skyline", "aerial", "above", "dawn"]):
            profile = {"preset": "aerial", "fov_deg": 55.0, "speed": 0.04, "elevation_deg": 30.0, "duration_seconds": 25.0}
        else:
            profile = {"preset": "forward", "fov_deg": 70.0, "speed": 0.04, "duration_seconds": 20.0}
        
        return json.dumps(profile)


NODE_CLASS_MAPPINGS = {
    "HYWorld_EnvironmentPromptBuilder": HYWorld_EnvironmentPromptBuilder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYWorld_EnvironmentPromptBuilder": "HYWorld Smart Prompt Evaluator (LLM)"
}
