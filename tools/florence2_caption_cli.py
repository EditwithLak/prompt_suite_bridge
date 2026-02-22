"""Florence-2 image captioning CLI (Transformers).

Designed to be called from a Forge/Gradio extension via subprocess.

Examples:
  python florence2_caption_cli.py --single image.jpg --model_id microsoft/Florence-2-base --task <MORE_DETAILED_CAPTION>
  python florence2_caption_cli.py --in_dir F:\imgs --recursive --skip_existing --task <CAPTION>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _iter_images(in_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in in_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(paths)


def _sidecar_path(img_path: Path, out_dir: str, ext: str) -> Path:
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        return od / f"{img_path.stem}{ext}"
    return img_path.with_suffix(ext)


def _apply_affixes(text: str, prefix: str, suffix: str) -> str:
    t = text.strip()
    if prefix:
        t = f"{prefix}{t}"
    if suffix:
        t = f"{t}{suffix}"
    return t.strip()


def _torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def _load_model(model_id: str, device: str, dtype: str, hf_cache: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if hf_cache:
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_cache, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache, "transformers"))

    torch_dtype = _torch_dtype(dtype)

    # Florence-2 often requires trust_remote_code=True on HF.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "auto" else None,
    )

    if device in {"cuda", "cpu"}:
        model = model.to(device)

    model.eval()
    return model, processor


def _generate_caption(model, processor, img: Image.Image, task_prompt: str, max_new_tokens: int):
    import torch

    prompt = task_prompt.strip()
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    # Move tensors to model device when needed
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # Florence-2 returns a string that may include the prompt/task token; post-process via processor
    try:
        parsed = processor.post_process_generation(generated_text, task=prompt, image_size=img.size)
        # Most tasks return dict with a single key
        if isinstance(parsed, dict) and parsed:
            # Prefer common caption keys
            for k in ("<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>", "<CAPTION>"):
                if k in parsed:
                    return str(parsed[k])
            # fallback: first value
            return str(next(iter(parsed.values())))
    except Exception:
        pass
    return generated_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="")
    ap.add_argument("--single", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--start_from", type=int, default=0)
    ap.add_argument("--caption_ext", type=str, default=".txt")
    ap.add_argument("--sort_method", type=str, default="sequential", choices=["sequential", "alphabetical"])

    ap.add_argument("--model_id", type=str, default="microsoft/Florence-2-base")
    ap.add_argument("--task", type=str, default="<MORE_DETAILED_CAPTION>")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--hf_cache", type=str, default="")

    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--suffix", type=str, default="")

    args = ap.parse_args()

    if not args.single and not args.in_dir:
        raise SystemExit("Provide --single or --in_dir")

    model, processor = _load_model(args.model_id, args.device, args.dtype, args.hf_cache)

    def caption_one(img_path: Path) -> str:
        img = Image.open(img_path).convert("RGB")
        cap = _generate_caption(model, processor, img, args.task, args.max_new_tokens)
        return _apply_affixes(str(cap), args.prefix, args.suffix)

    if args.single:
        img_path = Path(args.single)
        if not img_path.exists():
            raise SystemExit(f"File not found: {img_path}")
        print(caption_one(img_path))
        return

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")

    images = _iter_images(in_dir, args.recursive)
    if args.sort_method == "alphabetical":
        images = sorted(images, key=lambda p: p.name.lower())

    if args.start_from > 0:
        images = images[int(args.start_from):]

    total = len(images)
    if total == 0:
        print("No images found.")
        return

    done = 0
    for i, img_path in enumerate(images, start=0):
        out_path = _sidecar_path(img_path, args.out_dir, args.caption_ext)
        if args.skip_existing and out_path.exists():
            continue
        try:
            cap = caption_one(img_path)
            out_path.write_text(cap + "\n", encoding="utf-8")
            done += 1
        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
    print(f"Done. Wrote {done} captions ({total} files scanned).")


if __name__ == "__main__":
    main()
