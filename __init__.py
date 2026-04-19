"""HY-World 2.0 ComfyUI nodes — WorldMirror V1 & V2 3D reconstruction."""

import os
import sys

# Ensure repo root is in sys.path so `import hyworld2` and `from src...` work
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from .nodes import NODE_CLASS_MAPPINGS as HYWORLD_MAPPINGS
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as HYWORLD_DISPLAY_MAPPINGS
except Exception as e:
    import traceback
    print(f"❌ [HYWorld2] Failed to load visual nodes: {e}")
    traceback.print_exc()
    HYWORLD_MAPPINGS = {}
    HYWORLD_DISPLAY_MAPPINGS = {}

NODE_CLASS_MAPPINGS = {**HYWORLD_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**HYWORLD_DISPLAY_MAPPINGS}

# ---------------------------------------------------------
# HYRadio Node Integration
# ---------------------------------------------------------
import importlib
import logging
import warnings

log = logging.getLogger("HYRadio")

# Suppress HuggingFace Telemetry/Warnings
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
try:
    import huggingface_hub.utils._logging as hfh_logging
    hfh_logging.set_verbosity_error()
except Exception:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\..*")
warnings.filterwarnings("ignore", category=UserWarning,   module=r"transformers\..*")

_NODE_MODULES = {
    "HYRadio_LLMScriptWriter": (".nodes.story_orchestrator", "LLMScriptWriter", "📻 LLM Story Writer"),
    "HYRadio_LLMDirector":     (".nodes.story_orchestrator", "LLMDirector",      "🎬 LLM Director"),
    "HYRadio_BarkTTS":            (".nodes.bark_tts",           "BarkTTSNode",          "🎙️ Bark TTS (Suno)"),
    "HYRadio_SFXGenerator":       (".nodes.sfx_generator",      "SFXGenerator",         "💥 SFX Generator"),
    "HYRadio_SceneSequencer":     (".nodes.scene_sequencer",     "SceneSequencer",       "🎞️ Scene Sequencer"),
    "HYRadio_EpisodeAssembler":   (".nodes.scene_sequencer",     "EpisodeAssembler",     "📼 Episode Assembler"),
    "HYRadio_AudioEnhance":       (".nodes.audio_enhance",       "AudioEnhance",         "🔊 Spatial Audio Enhance"),
    "HYRadio_BatchBarkGenerator": (".nodes.batch_bark_generator", "BatchBarkGenerator",   "⚡ Batch Bark Generator"),
    "HYRadio_BatchKokoroGenerator":(".nodes.batch_kokoro_generator", "BatchKokoroGenerator","⚡ Batch Kokoro (4GB)"),
    "HYRadio_BatchAudioGenGenerator":(".nodes.batch_audiogen_generator", "BatchAudioGenGenerator","⚡ Batch AudioGen (Foley)"),
    "HYRadio_BatchProceduralSFX": (".nodes.batch_procedural_sfx", "BatchProceduralSFX",   "⚡ Batch Procedural SFX (Obsidian)"),
    "HYRadio_SignalLostVideo":    (".nodes.video_engine",          "SignalLostVideoRenderer", "📺 Signal Lost Video"),
    "HYRadio_ProjectStateLoader": (".nodes.project_state",         "ProjectStateLoader",      "📖 Project State Loader"),
    "HYRadio_KokoroAnnouncer":    (".nodes.kokoro_announcer",      "KokoroAnnouncer",         "🎙️ Kokoro Announcer"),
    "HYRadio_MusicGenTheme":      (".nodes.musicgen_theme",        "MusicGenTheme",           "🎺 MusicGen Theme"),
    "HYRadio_VRAMGuardian":       (".nodes.vram_guardian",          "VRAMGuardian",            "🛡️ VRAM Guardian"),
    "HYRadio_CinematicRenderer":  (".nodes.visual_renderer",   "HYRadio_CinematicRenderer", "🎬 Cinematic Renderer (gsplat)"),
    "HYRadio_VideoCompositor":    (".nodes.visual_compositor", "HYRadio_VideoCompositor",   "🎞️ Sequence Compositor"),
}

for node_name, (module_path, class_name, display_name) in _NODE_MODULES.items():
    try:
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, class_name)

        NODE_CLASS_MAPPINGS[node_name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name

        if node_name.startswith("HYRadio_"):
            legacy_name = node_name[4:]
            if legacy_name not in NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS[legacy_name] = cls

    except Exception as e:
        log.warning("[HYRadio] Failed to load '%s': %s", node_name, e)
        print(f"[HYRadio] ⚠️ Skipped '{node_name}': {e}")

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
