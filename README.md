# ComfyUI HYRadio — The Standalone Master Package

ComfyUI custom nodes for end-to-end Cinematic Production from a single text block. This suite mathematically orchestrates **Local LLMs**, **Spatial Audio (TTS)**, and **HY-World 2.0 (Tencent) 3D Scene Reconstruction** natively, without requiring external modules.

---

This is structurally a monolithic architecture incorporating:
1. Native Python Batch Iteration logic for handling infinitely scaling, hardware-safe scenes.
2. Synchronized `List Iterators` that automatically align Audio Output, 3D Mesh Output, and Cinematic Camera Math mapping arrays inside a 16GB VRAM limit.

## System Architecture

If you are just getting started, open `workflows/HYRadio-Full-Pipeline.json`. This massive JSON file encompasses everything:
*   **The Orchestrator:** Feed it an array of strings (e.g. `["Scene 1: Mars", "Scene 2: Moon"]`) and the LLM breaks it down into autonomous TTS pipelines and Visual Gen mappings.
*   **Audio Core:** Uses advanced `BarkTTS`, `Kokoro`, and `AudioEnhance` pipelines to bake stereo spatialized dialogue.
*   **Visual Core:** The Slicer logic dynamically wraps the SD3.5 Panoramas into Python lists, spinning up and flushing the Tencent `WorldMirror` geometry engine securely between batches.
*   **The AI Painter:** `WorldStereo` literally hallucinates (repaints) blind spots mathematically matched to your intrinsic/extrinsic arrays.

## Nodes

| Node | Description |
|------|-------------|
| `HYRadio_LLMScriptWriter` | LLM Story Orchestrator & Screenplay generation |
| `HYRadio_LLMDirector` | Automated Telemetry & Casting Logic |
| `HYRadio_BarkTTS` | Spatialized Text-to-Speech Engine |
| `HYRadio_SceneSequencer` | Assembles Audio chunks linearly |
| `HYRadio_EpisodeAssembler` | Stitches final MP4/WAV and theme paths |
| `HYWorld_EnvironmentPromptBuilder` | Visual Prompt LLM generator (VRAM Flushing) |
| `VNCCS_WorldMirrorV2_3D` | 3D Splatting / Geometry Engine (Tencent) |
| `HYWorld_BatchCLIPTextEncode` | Encodes iterative loops |
| `VNCCS_Equirect360ToViews` | Slices Panoramas and converts them to sequential Python Batches |

---

## Installation

### Manual Clone (Preferred)

Because of the aggressive List Iteration and custom telemetry, ensure you use the latest version of this repository natively:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jbrick2070/ComfyUI_HYWorld2.git
cd ComfyUI_HYWorld2
pip install -r requirements.txt
python install.py
```

*Note: The `requirements.txt` is fully merged, meaning all TTS packages and 3D Splat algorithms compile simultaneously.*
