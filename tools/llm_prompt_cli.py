# -*- coding: utf-8 -*-
"""
Text-only GGUF LLM prompt helper for SDXL Prompt Builder.
Runs outside Forge's Python via user-provided venv python.exe.
"""

import argparse
import json
import os
import re
import sys


SYSTEM_PROMPT = (
    "You are a Stable Diffusion XL prompt engineer. "
    "Goal: produce copy-ready SDXL prompts for image generation.\n"
    "Rules:\n"
    "- Keep content non-explicit. If sensual, keep it tasteful and non-graphic (editorial / implied).\n"
    "- Prefer concrete visual nouns/adjectives over prose.\n"
    "- Output STRICT JSON ONLY with keys: positive, negative.\n"
    "- positive should be a single line. negative should be a single line.\n"
)

def _safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return default

def _extract_json(text):
    text = (text or "").strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            # Try to clean trailing commas
            blob2 = re.sub(r",\s*}", "}", blob)
            blob2 = re.sub(r",\s*]", "]", blob2)
            try:
                return json.loads(blob2)
            except Exception:
                return None
    return None

def build_user_prompt(payload):
    mode = (payload.get("mode") or "Rewrite current prompt").strip()
    idea = (payload.get("idea") or "").strip()
    cur_pos = (payload.get("current_positive") or "").strip()
    cur_neg = (payload.get("current_negative") or "").strip()

    preset = (payload.get("preset") or "").strip()
    tag_style = (payload.get("tag_style") or "Comma tags (SDXL)").strip()
    family_safe = bool(payload.get("family_safe", False))

    camera = (payload.get("camera") or "").strip()
    lighting = (payload.get("lighting") or "").strip()
    extra_negative = (payload.get("extra_negative") or "").strip()

    if "Paragraph" in tag_style:
        style_line = (
            "Use a single-line natural paragraph prompt for photoreal SDXL: "
            "descriptive phrases, clear visual details, minimal comma-tag spam."
        )
    elif "Comma" in tag_style:
        style_line = "Use comma-separated tags (SDXL style)."
    else:
        style_line = "Use short bullet-ish blocks separated by ' • '."

    if mode.lower().startswith("generate"):
        if not idea:
            idea = "two male fashion models, cinematic editorial"
        parts = [
            "Task: Generate SDXL positive+negative prompts from this idea:",
            idea,
        ]
    else:
        parts = [
            "Task: Rewrite/improve these SDXL prompts (keep meaning, improve clarity, reduce redundancy):",
            "POSITIVE:",
            cur_pos if cur_pos else "(empty)",
            "NEGATIVE:",
            cur_neg if cur_neg else "(empty)",
        ]
        if idea:
            parts += ["Extra direction:", idea]

    if preset:
        parts += ["Preset context:", preset]

    if family_safe:
        parts += ["Safety: Force SFW/wholesome wording (no nudity, no erotic terms)."]

    if camera:
        parts += ["Camera tags to include (if relevant):", camera]
    if lighting:
        parts += ["Lighting tags to include (if relevant):", lighting]

    if extra_negative:
        parts += ["Extra negatives to include:", extra_negative]

    parts += [
        style_line,
        "Return JSON only: {\"positive\":\"...\",\"negative\":\"...\"}",
    ]
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--payload", required=True, help="Path to payload JSON")
    args = ap.parse_args()

    if not os.path.exists(args.payload):
        print(json.dumps({"error": "payload_not_found"}))
        return 2

    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_path = args.model
    if not os.path.exists(model_path):
        print(json.dumps({"error": "model_not_found", "model": model_path}))
        return 2

    temperature = _safe_float(payload.get("temperature", 0.7), 0.7)
    top_p = _safe_float(payload.get("top_p", 0.9), 0.9)
    max_tokens = _safe_int(payload.get("max_tokens", 512), 512)
    n_ctx = _safe_int(payload.get("n_ctx", 4096), 4096)
    n_gpu_layers = _safe_int(payload.get("n_gpu_layers", 0), 0)
    n_threads = _safe_int(payload.get("n_threads", 8), 8)

    user_prompt = build_user_prompt(payload)

    try:
        from llama_cpp import Llama
    except Exception as e:
        print(json.dumps({"error": "llama_cpp_import_failed", "detail": str(e)}))
        return 3

    llm = None
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
    except Exception as e:
        print(json.dumps({"error": "llama_init_failed", "detail": str(e)}))
        return 4

    out_text = ""
    # Try chat completion first
    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        out_text = resp["choices"][0]["message"]["content"]
    except Exception:
        # Fallback to plain completion
        prompt = SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\nJSON:"
        try:
            resp = llm.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=["\n\n"],
            )
            out_text = resp["choices"][0]["text"]
        except Exception as e:
            print(json.dumps({"error": "generation_failed", "detail": str(e)}))
            return 5

    data = _extract_json(out_text) or {}
    positive = (data.get("positive") or "").strip()
    negative = (data.get("negative") or "").strip()

    # If model didn't follow format, still return something
    if not positive:
        # try a heuristic: first non-empty line
        for ln in (out_text or "").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("{"):
                positive = ln
                break

    if not negative:
        negative = (payload.get("current_negative") or "").strip()

    print(json.dumps({"positive": positive, "negative": negative, "raw": out_text[:5000]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
