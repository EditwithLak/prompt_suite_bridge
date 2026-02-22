# 🧩 Prompt Suite Bridge (Vault-first) — v0.4.20

A **Stable Diffusion WebUI extension** (Forge Neo / A1111-style) that bundles:

- 🧱 **Prompt Composer** — build prompts fast, with Vault autocomplete + preset workflows
- 🗃️ **Vault + Maps** — manage **tags / packs / base prompts / LoRA+TI registry / mapsets**
- 🧪 **Map Generator** — batch/single **Canny / OpenPose / Depth** map generation + ControlNet routing
- 🏷️ **Caption Tools** — optional captioning/tagging backends (run via a **separate Python env**)

---

## ✅ Requirements

### Must-have
- **Stable Diffusion WebUI Forge Neo** (recommended) or another A1111-compatible fork
- Your WebUI’s default Python env (no extra deps required for **Vault + Prompt Composer**)

### Optional (but recommended)
- **ControlNet**
  - Forge: built-in `internal_controlnet`
  - A1111: classic ControlNet extension
- Separate Python env(s) for:
  - 🧪 Map Generator (ControlNet Aux detectors)
  - 🏷️ Caption Tools (JoyCaption / Florence-2 / BLIP / WD14)
  - ✍️ Local GGUF LLM prompt helper (text-only)

---

## 📦 Install

### Option A — Git clone (best)
1. Go to your WebUI extensions folder:
   - Forge Neo: `...\sd-webui-forge-neo\extensions\`
   - A1111: `...\stable-diffusion-webui\extensions\`
2. Clone this repo **so the folder name is exactly**:
   - `extensions/prompt_suite_bridge/`
3. Restart the WebUI

### Option B — Manual zip
1. Download this repo as zip
2. Extract into:
   - `extensions/prompt_suite_bridge/`
3. Restart the WebUI

After restart you should see a new top-level tab:
- **Prompt Suite (Vault-first) v0.4.20**

---

## 🧭 What’s inside (folder layout)

```text
prompt_suite_bridge/
├─ scripts/
│  ├─ prompt_suite_tab.py          # Main UI tabs
│  └─ prompt_suite_injector.py     # Queued-map → ControlNet injection on Generate
├─ lib/
│  ├─ prompt_builder_embed.py      # Prompt Composer UI + logic
│  ├─ vault_embed.py              # Vault UI + CRUD
│  ├─ map_generator_embed.py       # Map Generator UI + subprocess runner
│  ├─ caption_supporter_embed.py   # Caption Tools UI + subprocess runner
│  └─ *_store.py                   # Persistent storage (user_data/)
├─ tools/
│  ├─ batch_hints.py               # Map generation CLI (controlnet_aux)
│  ├─ joy_gguf_caption_cli.py      # JoyCaption (GGUF + mmproj, llama-cpp-python)
│  ├─ florence2_caption_cli.py     # Florence-2 captions (Transformers)
│  ├─ blip_caption_cli.py          # BLIP / BLIP-2 captions (Transformers)
│  ├─ wd14_tagger_cli.py           # WD14 tagger (ONNX)
│  └─ llm_prompt_cli.py            # Text-only GGUF prompt helper (llama-cpp-python)
├─ libraries/                      # Library .md files → imported into Vault on first run
├─ presets/                        # UI presets config
├─ packs/                          # JSON “pack” templates used by Prompt Composer
├─ javascript/
│  └─ cnbridge_send.js             # “Send preview maps to ControlNet” (best-effort DOM bridge)
└─ user_data/                      # Your data lives here (DO NOT COMMIT)
```

---

## 🚀 Quickstart (intended workflow)

### 1) First launch: Vault auto-import 📚
On first run, Prompt Suite will **auto-import** any `.md` files in `libraries/` into:

- `user_data/vault_db.json` (tags + packs + base prompts)
- `user_data/assets/` (for mapset storage)

So you get usable tag libraries immediately.

### 2) Build your library (Vault + Maps)
Open **🗃️ Vault + Maps**:

- **🔑 Library**
  - **Keywords** = single insertable tokens (with aliases + optional description)
  - **Packs** = curated bundles of keywords for quick insertion
  - **Base prompts** = templates that can include `[tag1]...[tagN]` or `[tags]`
- **🎛️ LoRA / TI**
  - Scan your LoRA folder + Embeddings folder
  - Add triggers/keywords/default strength so autocomplete inserts cleanly
- **🗺️ MapSets**
  - Create a mapset, then import canny/openpose/depth maps into it
  - Used later from Prompt Composer + Map Generator

### 3) Generate maps (Map Generator)
Open **🧪 Map Generator**:

- Pick **Input folder** (or a single file)
- Choose outputs:
  - ✅ Canny
  - ✅ OpenPose
  - ✅ Depth (MiDaS)
- Run batch → maps land in:
  - `out_dir/<map_type>/<vertical|landscape>/...`
- Register the output as a **MapSet** with one click (optional)

Then preview maps and:
- **⚡ Send ALL (txt2img/img2img)** (direct DOM bridge)
- **📌 Queue ALL (next Generate)** (injector-based, usually more reliable)

### 4) Compose prompt + generate (Prompt Composer)
Open **🧱 Prompt Composer**:

- Write/edit your **Positive** and **Negative** prompt
- Use **Vault autocomplete** directly inside the Positive box:
  - Tags + Packs + LoRAs + TIs
  - LoRA suggestions: type `lora:` or `<lora:`
  - TI suggestions: type `ti:`
- Queue maps into **ControlNet Units** (0..7)
- Click **Generate** in txt2img/img2img

---

## 🧱 Tab Guide (what each tab does)

### 🧱 Prompt Composer
Main workstation.

Key features:
- 🧠 **Inline autocomplete** (Vault tags/packs + LoRA/TI registry)
- 🎭 **Character creation helper**
  - Filters safer libraries by gender/era
  - Uses category naming convention like:
    - `male_modern_safe_clothing`
    - `unisex_fantasy_safe_accessories`
    - `male_any_restricted_body_language` (only appears if “Show restricted libraries” is enabled)
- 🧲 **ControlNet routing**
  - Queue canny/openpose/depth maps into CN units + weights
  - Works with Forge `internal_controlnet` or classic ControlNet extension
- ✍️ **Local GGUF LLM Prompt Helper (optional)**
  - Uses `tools/llm_prompt_cli.py` via a user-chosen `python.exe`

Settings & saves:
- `user_data/prompt_presets.json`
- `user_data/llm_settings.json`
- `user_data/prompt_preset_assets/`

---

### 🗃️ Vault + Maps
Asset management only (no direct send-to-generation).

Includes:
- 🔑 Library (Keywords / Packs / Base prompts)
- 🎛️ LoRA / TI registry (scan folders + save trigger metadata)
- 🗺️ MapSets (saved canny/depth/openpose collections)
- 📚 Library Files manager (edit/import `.md` files live)

Data:
- `user_data/vault_db.json`
- `user_data/assets/`

---

### 🧪 Map Generator
Generates “hint maps” using `tools/batch_hints.py`.

Highlights:
- Batch or single-file generation
- Auto orientation split (**vertical** vs **landscape**)
- Optional overlays (PNG watermark + text)
- Preview generated maps → Send/Queue to ControlNet
- One-click register output folder as a **MapSet**

Settings:
- `user_data/mapgen_settings.json`

---

### 🏷️ Caption Tools
Runs external caption/tag pipelines from inside the WebUI, but **executed via subprocess** using your chosen `python.exe`.

Backends:
- ✅ **JoyCaption GGUF (llama.cpp)** (multimodal via mmproj)
- **Florence-2 (Transformers)**
- **BLIP / BLIP-2 (Transformers)**
- **WD14 Tagger (ONNX)**

Designed to run in a separate env so you don’t pollute your WebUI python.

---

## 🧪 Separate Python envs (recommended)

> You can use **one big env** for everything, but splitting keeps Forge stable ✅

### A) Map Generator env (controlnet_aux)
Purpose: run `tools/batch_hints.py` (Canny/OpenPose/MiDaS)

**Windows (PowerShell):**
```powershell
cd D:\AI\envs
python -m venv cn_hints_env
.\cn_hints_env\Scripts\activate
pip install -U pip
pip install pillow tqdm opencv-python
pip install controlnet-aux
# torch depends on your CUDA/CPU setup:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Notes:
- `batch_hints.py` uses:
  - `controlnet_aux.canny.CannyDetector`
  - `controlnet_aux.open_pose.OpenposeDetector`
  - `controlnet_aux.midas.MidasDetector`
- If OpenPose complains about missing extras, install:
  - `pip install mediapipe` (only if required by your controlnet_aux build)
- Hands/face drawing may require:
  - `pip install matplotlib`

Then in **Map Generator** tab:
- set **Python executable** to: `...\cn_hints_env\Scripts\python.exe`

---

### B) Caption env (Transformers + ONNX + optional llama.cpp)
Purpose: run caption tools and/or WD14

**Windows (PowerShell):**
```powershell
cd D:\AI\envs
python -m venv caption_env
.\caption_env\Scripts\activate
pip install -U pip

# Shared basics
pip install pillow numpy opencv-python huggingface-hub

# Transformers captioners
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate

# WD14 ONNX
pip install onnxruntime        # CPU
# OR: pip install onnxruntime-gpu  # if you want CUDA ONNX and it matches your setup

# Optional: JoyCaption / local GGUF
pip install llama-cpp-python
```

Then in **Caption Tools** tab:
- set **Python executable** to: `...\caption_env\Scripts\python.exe`

---

### C) GGUF Prompt Helper env (text-only LLM)
Purpose: run `tools/llm_prompt_cli.py` from Prompt Composer

You can reuse the caption env if it has `llama-cpp-python`.

Minimum:
```powershell
python -m venv llm_env
.\llm_env\Scripts\activate
pip install -U pip
pip install llama-cpp-python
```

Then in **Prompt Composer → Local LLM** settings:
- `python_exe` = `...\llm_env\Scripts\python.exe`
- `model_path` = path to your **text-only GGUF**

---

## 🧠 Models you may need to download

### ControlNet-related (Map Generator)
- `OpenposeDetector.from_pretrained("lllyasviel/ControlNet")`
- `MidasDetector.from_pretrained("lllyasviel/ControlNet")`

Usually auto-download from Hugging Face on first run (internet required).

### JoyCaption (GGUF + mmproj)
You must provide both:
- ✅ JoyCaption **GGUF** (main model)
- ✅ **mmproj GGUF** (vision projection model)

Paste both file paths in **Caption Tools → JoyCaption settings**.

### Florence-2 / BLIP / BLIP-2 (Transformers)
Downloaded automatically by model id, e.g.:
- `microsoft/Florence-2-base`
- `Salesforce/blip-image-captioning-base`
- `Salesforce/blip2-opt-2.7b-coco`

### WD14 Tagger (ONNX)
Auto-downloads from HF repo (default):
- `SmilingWolf/wd-v1-4-convnextv2-tagger-v2`
It pulls:
- `model.onnx`
- `selected_tags.csv`

---

## 🔌 ControlNet integration (how it works)

There are **two** ways Prompt Suite feeds maps into ControlNet:

### 1) 📌 Queue → Inject on Generate (recommended)
- Queue maps to Unit 0/1/2/etc
- On **Generate**, `scripts/prompt_suite_injector.py` injects them into ControlNet processing
- Best with Forge `internal_controlnet`, also supports classic ControlNet

### 2) ⚡ Send from previews (best-effort DOM bridge)
- Uses `javascript/cnbridge_send.js`
- Tries to find ControlNet file inputs inside txt2img/img2img and set them directly
- Can break if WebUI/ControlNet DOM changes

If you want maximum stability: use **Queue**.

---

## 🧷 Data & persistence (important)

All user content lives under `prompt_suite_bridge/user_data/`:

- `vault_db.json` — tags/packs/mapsets/loras registry
- `assets/` — mapset files
- `mapgen_settings.json` — Map Generator settings
- `llm_settings.json` — Local GGUF prompt helper settings
- `prompt_presets.json` — saved prompt presets
- `prompt_preset_assets/` — copied assets for presets

✅ Recommended: keep `user_data/` out of Git.

Add this to `.gitignore`:
```gitignore
__pycache__/
*.pyc

prompt_suite_bridge/user_data/**
!prompt_suite_bridge/user_data/.gitkeep
!prompt_suite_bridge/user_data/assets/.gitkeep
```

---

## 🧯 Troubleshooting

### “ControlNet bridge not found”
- Suite still works, but queue injection won’t apply.
- Install/enable ControlNet (Forge internal or A1111 extension), then restart.

### Map Generator errors (controlnet_aux / torch / mediapipe)
- Use a **separate env** (cn_hints_env)
- Ensure:
  - `pip install controlnet-aux opencv-python pillow tqdm`
  - `pip install torch` matching your CUDA/CPU
- If OpenPose complains about missing stuff, try:
  - `pip install mediapipe`

### WD14 errors (cv2 / onnxruntime)
Install in caption env:
- `pip install opencv-python onnxruntime huggingface-hub`

### JoyCaption doesn’t use GPU
GPU support depends on how `llama-cpp-python` was built.
CPU works everywhere.
For CUDA build (Windows), you may need:
```powershell
set CMAKE_ARGS=-DGGML_CUDA=on
pip install --no-cache-dir llama-cpp-python
```

### Nothing saves / Vault resets
Check write permissions for:
- `prompt_suite_bridge/user_data/`

---

## 🙏 Credits / Dependencies
- Gradio (WebUI UI layer)
- `controlnet-aux` (hint map detectors)
- Transformers + Torch (Florence/BLIP)
- `onnxruntime` (WD14)
- `llama-cpp-python` (GGUF prompt helper + JoyCaption)

---
