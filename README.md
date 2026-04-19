# ComfyUI-HYRadio

**Real science news → Mistral 12B LLM Scripts → Bark TTS Voices → Kokoro Narration → HunyuanWorld V2 Gaussian Splat Environments → LLM-Directed Camera → Synced CRT MP4 Episode.**

Fully automated. Zero API keys. Drop into `custom_nodes/` and queue.

---

## 🚀 Quick Start (The "Zero-Click" Path)
1. **Get ComfyUI**: Use the [Official Desktop Installer](https://www.comfy.org/download).
2. **Install HYRadio**: Use **Install via Git URL** in the ComfyUI Manager and paste our repo link.
3. **Run**: Drag `workflows/HYRadio-Visual-Test.json` into your browser and hit **Queue Prompt**.
4. **Walk Away**: Everything else (Scripting, Voices, Splatting, Rendering, and Mastery) is safely executed offline on your local GPU.

### Step 5 — Continuous 24/7 Broadcast (OBS Automation)
Run HYRadio as a live generative broadcast — each output episode auto-loads into OBS as it finishes.

**Prerequisites:**
- [OBS Studio](https://obsproject.com/download)
- [Media Playlist Source (OBS Plugin)](https://obsproject.com/forum/resources/media-playlist-source.1765/)
- [Directory Sorter for OBS](https://github.com/CodeYan01/directory_sorter_for_obs)

**Setup:**
1. Install OBS and the Media Playlist Source plugin.
2. In OBS: **Tools → Scripts → Python Settings** → point to your Python path.
3. Load the `directory_sorter_for_obs` script → point to `ComfyUI/output/hyradio/`.
4. Add a **Media Playlist Source** scene item pointed to the same folder.
5. OBS picks up each new MP4 automatically as the pipeline finishes.

---

## What It Does
ComfyUI-HYRadio is an end-to-end proc-gen media machine. It fuses our legacy OldTimeRadio audio stack with the cutting-edge visual horsepower of HunyuanWorld V2. 

The pipeline handles the entire production:
- **Audio Stack**: A Model-Independent LLM (with **Mistral Nemo 12B** as the flagship default) writes a multi-act sci-fi radio drama from real RSS feeds. Kokoro v1.0 narrates, Bark TTS voices the characters, and procedural 48kHz SFX glue the audio together.
- **Visual Stack (Gate 0 Architecture)**: The LLM writes visual environment prompts and directs the virtual camera. HunyuanWorld V2 processes these environments into **per-scene Gaussian Splats**. The pipeline enforces boundary validation and prompt diversity, then renders the 3D pathways.
- **Mastering**: The system proportionally syncs the visual Gaussian rendering exact to the audio duration and bakes the final seamless `.mp4` episode locally.

---

## Hardware Requirements & Benchmarks
This is an automated 3D cinematic pipeline entirely run on local silicon:
- **Recommended**: RTX 5080 (Blackwell sm_120 structure) or RTX 4090 (16GB+ VRAM required).
- **Minimum**: 12GB VRAM.

> [!WARNING]
> **This is a heavy pipeline.** Producing over 11,000 continuous frames for a ~500 second episode forces a huge GPU footprint. The pipeline dynamically swaps models off the GPU (LLM -> Bark -> Visual Splats) to aggressively protect your VRAM pool, but intermediate frame accumulation heavily taxes standard CPU RAM limits (~37 GB system RAM draw at generation peaks).

---

## Pipeline Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. THE SCRIPTWRITER & AUDIO FORGE 📻                                                               │
│                                                                                                    │
│ ┌──────────────────────────┐     ┌───────────────────────────────────────────────────────────────┐ │
│ │ LLM Script & Director    │────►│ Audio Stack: Bark TTS + Kokoro + MusicGen + Procedural SFX    │ │
│ │ Mistral Nemo 12B         │     │ Natively masters an exact 48kHz audio duration string.        │ │
│ └──────────────────────────┘     └───────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬─────────────────────────────────────────────────────────────────────────┘
                           │ audio timeline & visual context metadata passed to rendering loop
                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. HUNYUANWORLD V2 VISUAL PIPELINE 🎥  [PER-SCENE SPLAT ARCHITECTURE]                              │
│                                                                                                    │
│ ┌───────────────────────────────┐   ┌───────────────────────────────┐   ┌────────────────────────┐ │
│ │ LLM_EnvironmentBridge         │   │ per_scene_splats.py           │   │ visual_renderer.py     │ │
│ │ Enforces diversity heuristics │   │ Generates individual PLY      │   │ Renders all PLY files  │ │
│ │ Strips scaffolding / leaks    │──►│ objects for each scene        │──►│ boundary validations   │ │
│ │ Scales visual duration params │   │ Memory boundaries enforced    │   │ and 3D path generation │ │
│ │ to match AUDIO explicitly.    │   │ between per-scene boundaries  │   │ exact to camera prompts│ │
│ └───────────────────────────────┘   └───────────────────────────────┘   └────────────────────────┘ │
└──────────────────────────┬────────────────────────────────────────┬────────────────────────────────┘
                           │ 3D Rendered Array                      │ Extracted Foley / SFX String
                           ▼                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. 🎬 HYRADIO CRT COMPOSITOR 🎬                                                                    │
│                                                                                                    │
│ Assembles 11,000+ frames with the mastered 48kHz audio runtime, executes h264_nvenc processing,    │
│ applies the stylized CRT aesthetic frame wrapper, and outputs your final HD '.mp4' cinematic track.│
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Node Reference

### The Visual Drama Engine
| Node | What It Does |
|------|-------------|
| **HYWorld_EnvironmentPromptBuilder** | Dynamically guides the camera (FOV, Speed, Preset) utilizing prompt constraints, restricts prompt leaks, maintains trajectory bounds matching the LLM output, and enforces diversity limits between scenes iteratively utilizing `_strip_leaks()`. |
| **HYWorld_PerSceneSplats** | Wraps the WorldMirror V2 inferencer, creating a list of `.PLY` generated splat objects *per-scene*. Flushes VRAM safely sequentially. |
| **HYRadio_CinematicRenderer** | Natively array-aware rendering module. Applies mathematical validations per PLY `_validate_splats()` and trajectory parameters `_validate_trajectory()` correcting bounding issues algorithmically before triggering standard 3D mapping. |

### The Audio Engine Backbone
*(Inherited securely from OldTimeRadio `v1.7`)*
| Node | What It Does |
|------|-------------|
| **LLM Story Writer** | Multi-act script generations built from live-RSS feeds. Mistral Nemo 12B is the flagship engine. Evaluates narrative spines automatically. |
| **Batch Bark / Kokoro Announcer** | Dedicated generative models for actor dialog tracking and BBC-style omniscient framing narration. |
| **Make It Sound Awesome** | The automated acoustic mixer natively expanding Haas-frequencies and equalizations safely up to 48kHz. |

---

## The Roadmap

### [✓] Gate 0: Render Engine Stabilization (Current)
The first priority. We patched fatal list-shaping execution halts, embedded fail-soft memory validators inside the rendering structure, attached per-scene parameter duration math to safely scale Gaussian rendering against audio length constraints independently, and officially wrapped WorldMirror V2 into a multi-act scene loop capable of producing distinct PLY spheres sequentially. 

### [ ] Gate 0.5: Memory Relief
If users frequently encounter Out-Of-Memory system crashes holding the 11,772+ list tensor arrays in standard PC memory, we will formally integrate `BS.9` streaming configurations. This will directly offload internal frame blocks out to a harddisk `.png` sequence to unburden the active system during execution.

### [ ] Gate 1: True Audiovisual Anchoring
Moving past scaled interpolation limits. Gate 1 introduces script timestamps directly mapping individual Bark character generations locally. Instead of forcing raw timing percentages, visual and auditory triggers will directly match individual lines/acts locally creating unified 3D-camera and semantic interactions.

---

## 🛠️ Developer Protocol
*For the next AI assistant or contributor:*

1. **Test Sovereignty:** Execute all testing modules locally. Confirm pipeline paths do not leak VRAM.
2. **Flash Attention 2 / SageAttention:** Native Blackwell sm_120 compilation is not directly provided gracefully by FA2 windows structures. SageAttention correctly runs optimal scaling paths.
3. **P0 Validations First**: If you expand rendering configurations or metadata trajectories, always push them back through `_validate_trajectory()` and `_validate_splats()`. Do not circumvent the NaN scrubbers—Hunyuan outputs consistently require sanitation bounding!

---

## License & Credits
MIT License

- **Mistral Nemo 12B** by [Mistral AI](https://huggingface.co/mistralai)
- **HunyuanWorld V2** by [Tencent Hunyuan](https://github.com/Tencent/Hunyuan3D-2)
- **Bark** by [Suno AI](https://github.com/suno-ai/bark)
- **Kokoro TTS** by [hexgrad](https://huggingface.co/hexgrad)
- Built directly on top of the ComfyUI API standards.
- Engineered by [Jeffrey Brick](https://github.com/jbrick2070)
