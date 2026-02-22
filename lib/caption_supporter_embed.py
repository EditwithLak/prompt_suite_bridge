import os
import sys
import subprocess
import gradio as gr
# --- simple tkinter folder picker (works on Windows)
def _pick_folder(initial_dir: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        if initial_dir and os.path.isdir(initial_dir):
            return filedialog.askdirectory(initialdir=initial_dir) or ""
        return filedialog.askdirectory() or ""
    except Exception:
        return ""


def _run_subprocess(py_exe: str, args: list[str], extra_env: dict | None = None) -> tuple[str, str]:
    """
    Returns (stdout, stderr). No streaming; keeps it simple & reliable.
    """
    if not py_exe:
        py_exe = sys.executable

    cmd = [py_exe] + args
    try:
        env = None
        if extra_env:
            env = os.environ.copy()
            env.update({k: str(v) for k, v in extra_env.items() if v is not None})
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        out = p.stdout.strip()
        err = p.stderr.strip()
        if p.returncode != 0:
            if not err:
                err = f"Process exited with code {p.returncode}"
        return out, err
    except Exception as e:
        return "", f"Failed to run: {e}"


def _tool_path(filename: str) -> str:
    here = os.path.dirname(__file__)
    ext_root = os.path.abspath(os.path.join(here, ".."))  # extensions/<this_ext>/
    return os.path.join(ext_root, "tools", filename)


def _default_models_hint() -> str:
    # Just a helpful note, not hardcoded requirement.
    return (
        "Tip: your JoyCaption GGUFs look like they live in something like:\n"
        r"C:\ComfyUI\models\LLM\GGUF\\"
        "\nYou can reuse those paths here."
    )


def build_caption_supporter_ui():
    with gr.Column():
        gr.Markdown("## 🏷️ Caption Supporter (Forge tab) — **no ComfyUI**")
        gr.Markdown(
            "This tab runs captioning models *from inside Forge UI*.\n\n"
            "**Backend 1 (implemented):** JoyCaption GGUF (llama.cpp)\n\n"
            "Backends: Florence-2 / BLIP / BLIP-2 / WD14 tagger (pick in dropdown)."
        )

        with gr.Accordion("Environment", open=True):
            py_exe = gr.Textbox(
                label="Python executable (recommended: caption_env python.exe)",
                placeholder=r"D:\AI\caption_env\Scripts\python.exe",
                value=""
            )
            gr.Markdown(_default_models_hint())

        with gr.Accordion("Mode", open=True):
            backend = gr.Dropdown(
                label="Caption backend",
                choices=[
                    "JoyCaption GGUF (llama.cpp)",
                    "Florence-2 (Transformers)",
                    "BLIP (Transformers)",
                    "BLIP-2 (Transformers)",
                    "WD14 Tagger (Danbooru tags)",
                ],
                value="JoyCaption GGUF (llama.cpp)",
            )

            hf_cache = gr.Textbox(
                label="HF cache dir (optional)",
                placeholder=r"F:\HF_cache  (leave empty = default user cache)",
                value="",
            )

        with gr.Accordion("JoyCaption GGUF settings", open=True):
            model_path = gr.Textbox(label="Model GGUF path", placeholder=r"C:\...\llama-joycaption-....Q6_K.gguf")
            mmproj_path = gr.Textbox(label="mmproj GGUF path", placeholder=r"C:\...\llava-mmproj-model-f16.gguf")

            with gr.Row():
                n_ctx = gr.Slider(1024, 8192, value=4096, step=256, label="Context (n_ctx)")
                n_gpu_layers = gr.Slider(-1, 120, value=-1, step=1, label="GPU layers (-1 = auto/all)")
                n_threads = gr.Slider(1, 32, value=8, step=1, label="CPU threads")

            with gr.Row():
                style = gr.Dropdown(
                    label="Prompt style",
                    choices=["Descriptive", "Stable Diffusion Prompt", "Custom"],
                    value="Stable Diffusion Prompt"
                )
                length = gr.Dropdown(
                    label="Caption length",
                    choices=["short", "any", "long"],
                    value="long"
                )

            custom_prompt = gr.Textbox(
                label="Custom prompt (only used if style=Custom)",
                lines=3,
                placeholder="Example: Describe the outfit + accessories as comma-separated SD tags. Avoid brand names."
            )

            with gr.Row():
                max_new_tokens = gr.Slider(16, 512, value=256, step=16, label="Max new tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.6, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Top-p")
                top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k (0=disabled)")

            with gr.Row():
                prefix = gr.Textbox(label="Prefix (optional)", placeholder="e.g., photo of ")
                suffix = gr.Textbox(label="Suffix (optional)", placeholder="e.g., , best quality")

        with gr.Accordion("Florence-2 settings", open=False):
            florence_model_id = gr.Textbox(
                label="Model ID",
                value="microsoft/Florence-2-base",
                placeholder="microsoft/Florence-2-base or microsoft/Florence-2-large",
            )
            florence_task = gr.Dropdown(
                label="Task prompt",
                choices=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>", "<OCR>", "<OD>"] ,
                value="<MORE_DETAILED_CAPTION>",
            )
            with gr.Row():
                tr_device = gr.Dropdown(label="Device", choices=["auto", "cuda", "cpu"], value="auto")
                tr_dtype = gr.Dropdown(label="Dtype", choices=["auto", "float16", "bfloat16", "float32"], value="auto")
            florence_max_new_tokens = gr.Slider(16, 512, value=128, step=8, label="Max new tokens")

        with gr.Accordion("BLIP / BLIP-2 settings", open=False):
            blip_model_id = gr.Textbox(
                label="Model ID",
                value="Salesforce/blip-image-captioning-base",
                placeholder="BLIP: Salesforce/blip-image-captioning-base  |  BLIP-2: Salesforce/blip2-opt-2.7b (or -coco)",
            )
            blip_prompt = gr.Textbox(
                label="Prompt (optional, mainly for BLIP-2)",
                value="",
                placeholder="Example: Describe the outfit + accessories. Avoid brand names.",
            )
            blip_max_new_tokens = gr.Slider(16, 256, value=64, step=8, label="Max new tokens")

        with gr.Accordion("WD14 Tagger settings", open=False):
            wd_repo_id = gr.Textbox(
                label="Repo ID",
                value="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
            )
            wd_model_dir = gr.Textbox(
                label="Model download dir (optional)",
                placeholder=r"F:\wd14_models (leave empty = cache under HF cache / default user cache)",
                value="",
            )
            with gr.Row():
                wd_general_thr = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="General threshold")
                wd_char_thr = gr.Slider(0.0, 1.0, value=0.85, step=0.01, label="Character threshold")
            wd_remove_underscore = gr.Checkbox(label="Remove underscores", value=True)
            wd_undesired = gr.Textbox(label="Undesired tags (comma-separated)", value="")

        with gr.Accordion("Batch processing", open=True):
            with gr.Row():
                in_dir = gr.Textbox(label="Input folder", placeholder=r"F:\images")
                pick_in = gr.Button("📁 Pick input")

            with gr.Row():
                out_dir = gr.Textbox(label="Output folder (leave empty = sidecar .txt next to image)", value="")
                pick_out = gr.Button("📁 Pick output")

            with gr.Row():
                recursive = gr.Checkbox(label="Recursive (include subfolders)", value=False)
                skip_existing = gr.Checkbox(label="Skip if .txt exists", value=True)
                start_from = gr.Number(label="Start from index", value=0, precision=0)

            sort_method = gr.Dropdown(
                label="Sort method",
                choices=["sequential", "alphabetical"],
                value="sequential"
            )

            run_batch = gr.Button("🚀 Run batch captioning")

        with gr.Accordion("Single image", open=False):
            single_img = gr.File(label="Drop 1 image (jpg/png/webp)", file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp"])
            run_single = gr.Button("✨ Caption this image")
            single_caption = gr.Textbox(label="Caption result", lines=6)

        log = gr.Textbox(label="Log", lines=14)

        # --- pickers
        pick_in.click(lambda cur: _pick_folder(cur), inputs=[in_dir], outputs=[in_dir])
        pick_out.click(lambda cur: _pick_folder(cur), inputs=[out_dir], outputs=[out_dir])

        def _run_batch(
            py_exe, hf_cache, backend,
            # Joy
            model_path, mmproj_path, n_ctx, n_gpu_layers, n_threads,
            style, length, custom_prompt, max_new_tokens, temperature, top_p, top_k,
            # Transformers shared
            tr_device, tr_dtype,
            # Florence
            florence_model_id, florence_task, florence_max_new_tokens,
            # BLIP
            blip_model_id, blip_prompt, blip_max_new_tokens,
            # WD14
            wd_repo_id, wd_model_dir, wd_general_thr, wd_char_thr, wd_remove_underscore, wd_undesired,
            # IO
            prefix, suffix, in_dir, out_dir, recursive, skip_existing, start_from, sort_method,
        ):

            if not in_dir or not os.path.isdir(in_dir):
                return "Input folder not found."

            backend = (backend or "").strip()

            if backend == "JoyCaption GGUF (llama.cpp)":
                tool = _tool_path("joy_gguf_caption_cli.py")
                args = [
                    tool,
                    "--in_dir", in_dir,
                    "--model", model_path,
                    "--mmproj", mmproj_path,
                    "--n_ctx", str(int(n_ctx)),
                    "--n_gpu_layers", str(int(n_gpu_layers)),
                    "--n_threads", str(int(n_threads)),
                    "--style", style,
                    "--length", length,
                    "--max_new_tokens", str(int(max_new_tokens)),
                    "--temperature", str(float(temperature)),
                    "--top_p", str(float(top_p)),
                    "--top_k", str(int(top_k)),
                    "--start_from", str(int(start_from)),
                    "--sort_method", sort_method,
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]
                if style == "Custom" and (custom_prompt or "").strip():
                    args += ["--custom_prompt", custom_prompt.strip()]

            elif backend == "Florence-2 (Transformers)":
                tool = _tool_path("florence2_caption_cli.py")
                args = [
                    tool,
                    "--in_dir", in_dir,
                    "--model_id", florence_model_id,
                    "--task", florence_task,
                    "--device", tr_device,
                    "--dtype", tr_dtype,
                    "--max_new_tokens", str(int(florence_max_new_tokens)),
                    "--hf_cache", hf_cache or "",
                    "--start_from", str(int(start_from)),
                    "--sort_method", sort_method,
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]

            elif backend in {"BLIP (Transformers)", "BLIP-2 (Transformers)"}:
                tool = _tool_path("blip_caption_cli.py")
                mode = "blip2" if backend.startswith("BLIP-2") else "blip"
                args = [
                    tool,
                    "--in_dir", in_dir,
                    "--mode", mode,
                    "--model_id", blip_model_id,
                    "--prompt", blip_prompt or "",
                    "--device", tr_device,
                    "--dtype", tr_dtype,
                    "--max_new_tokens", str(int(blip_max_new_tokens)),
                    "--hf_cache", hf_cache or "",
                    "--start_from", str(int(start_from)),
                    "--sort_method", sort_method,
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]

            elif backend == "WD14 Tagger (Danbooru tags)":
                tool = _tool_path("wd14_tagger_cli.py")
                args = [
                    tool,
                    "--in_dir", in_dir,
                    "--repo_id", wd_repo_id,
                    "--model_dir", wd_model_dir or "",
                    "--general_thr", str(float(wd_general_thr)),
                    "--char_thr", str(float(wd_char_thr)),
                    "--undesired", wd_undesired or "",
                    "--hf_cache", hf_cache or "",
                    "--start_from", str(int(start_from)),
                    "--sort_method", sort_method,
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]
                if wd_remove_underscore:
                    args += ["--remove_underscore"]

            else:
                return f"Unknown backend: {backend}"

            # shared batch flags
            if out_dir:
                args += ["--out_dir", out_dir]
            if recursive:
                args += ["--recursive"]
            if skip_existing:
                args += ["--skip_existing"]

            out, err = _run_subprocess(py_exe, args)
            return (out + ("\n\n" + err if err else "")).strip()

        run_batch.click(
            _run_batch,
            inputs=[
                py_exe, hf_cache, backend,
                model_path, mmproj_path, n_ctx, n_gpu_layers, n_threads,
                style, length, custom_prompt, max_new_tokens, temperature, top_p, top_k,
                tr_device, tr_dtype,
                florence_model_id, florence_task, florence_max_new_tokens,
                blip_model_id, blip_prompt, blip_max_new_tokens,
                wd_repo_id, wd_model_dir, wd_general_thr, wd_char_thr, wd_remove_underscore, wd_undesired,
                prefix, suffix,
                in_dir, out_dir, recursive, skip_existing, start_from, sort_method
            ],
            outputs=[log]
        )

        def _run_single(
            py_exe, hf_cache, backend,
            # Joy
            model_path, mmproj_path, n_ctx, n_gpu_layers, n_threads,
            style, length, custom_prompt, max_new_tokens, temperature, top_p, top_k,
            # Transformers shared
            tr_device, tr_dtype,
            # Florence
            florence_model_id, florence_task, florence_max_new_tokens,
            # BLIP
            blip_model_id, blip_prompt, blip_max_new_tokens,
            # WD14
            wd_repo_id, wd_model_dir, wd_general_thr, wd_char_thr, wd_remove_underscore, wd_undesired,
            # IO
            prefix, suffix,
            file_obj,
        ):

            if not file_obj:
                return "", "No file received."

            # gr.File gives dict-ish object; try common keys
            img_path = getattr(file_obj, "name", None) or file_obj
            if not img_path or not os.path.exists(img_path):
                return "", "Could not read uploaded image path."

            backend = (backend or "").strip()

            if backend == "JoyCaption GGUF (llama.cpp)":
                tool = _tool_path("joy_gguf_caption_cli.py")
                args = [
                    tool,
                    "--single", img_path,
                    "--model", model_path,
                    "--mmproj", mmproj_path,
                    "--n_ctx", str(int(n_ctx)),
                    "--n_gpu_layers", str(int(n_gpu_layers)),
                    "--n_threads", str(int(n_threads)),
                    "--style", style,
                    "--length", length,
                    "--max_new_tokens", str(int(max_new_tokens)),
                    "--temperature", str(float(temperature)),
                    "--top_p", str(float(top_p)),
                    "--top_k", str(int(top_k)),
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]
                if style == "Custom" and (custom_prompt or "").strip():
                    args += ["--custom_prompt", custom_prompt.strip()]

            elif backend == "Florence-2 (Transformers)":
                tool = _tool_path("florence2_caption_cli.py")
                args = [
                    tool,
                    "--single", img_path,
                    "--model_id", florence_model_id,
                    "--task", florence_task,
                    "--device", tr_device,
                    "--dtype", tr_dtype,
                    "--max_new_tokens", str(int(florence_max_new_tokens)),
                    "--hf_cache", hf_cache or "",
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]

            elif backend in {"BLIP (Transformers)", "BLIP-2 (Transformers)"}:
                tool = _tool_path("blip_caption_cli.py")
                mode = "blip2" if backend.startswith("BLIP-2") else "blip"
                args = [
                    tool,
                    "--single", img_path,
                    "--mode", mode,
                    "--model_id", blip_model_id,
                    "--prompt", blip_prompt or "",
                    "--device", tr_device,
                    "--dtype", tr_dtype,
                    "--max_new_tokens", str(int(blip_max_new_tokens)),
                    "--hf_cache", hf_cache or "",
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]

            elif backend == "WD14 Tagger (Danbooru tags)":
                tool = _tool_path("wd14_tagger_cli.py")
                args = [
                    tool,
                    "--single", img_path,
                    "--repo_id", wd_repo_id,
                    "--model_dir", wd_model_dir or "",
                    "--general_thr", str(float(wd_general_thr)),
                    "--char_thr", str(float(wd_char_thr)),
                    "--undesired", wd_undesired or "",
                    "--hf_cache", hf_cache or "",
                    "--prefix", prefix or "",
                    "--suffix", suffix or "",
                ]
                if wd_remove_underscore:
                    args += ["--remove_underscore"]
            else:
                return "", f"Unknown backend: {backend}"

            out, err = _run_subprocess(py_exe, args)
            caption = out.strip()
            full_log = (out + ("\n\n" + err if err else "")).strip()
            return caption, full_log

        run_single.click(
            _run_single,
            inputs=[
                py_exe, hf_cache, backend,
                model_path, mmproj_path, n_ctx, n_gpu_layers, n_threads,
                style, length, custom_prompt, max_new_tokens, temperature, top_p, top_k,
                tr_device, tr_dtype,
                florence_model_id, florence_task, florence_max_new_tokens,
                blip_model_id, blip_prompt, blip_max_new_tokens,
                wd_repo_id, wd_model_dir, wd_general_thr, wd_char_thr, wd_remove_underscore, wd_undesired,
                prefix, suffix,
                single_img
            ],
            outputs=[single_caption, log]
        )

    return
