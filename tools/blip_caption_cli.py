"""BLIP / BLIP-2 image captioning CLI (Transformers).

Supports:
  - BLIP (Salesforce/blip-image-captioning-*)
  - BLIP-2 (Salesforce/blip2-*)

Examples:
  python blip_caption_cli.py --single image.jpg --mode blip --model_id Salesforce/blip-image-captioning-base
  python blip_caption_cli.py --in_dir F:\imgs --mode blip2 --model_id Salesforce/blip2-opt-2.7b-coco --prompt "Describe the scene" --recursive
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
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


def _setup_cache(hf_cache: str):
    if not hf_cache:
        return
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_cache, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache, "transformers"))


def _get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_dtype(dtype: str):
    if dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def _load_blip(model_id: str, device: torch.device, dtype):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return model, processor


def _load_blip2(model_id: str, device: torch.device, dtype):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    processor = Blip2Processor.from_pretrained(model_id)
    # device_map="auto" can be useful for large models, but keep it simple
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return model, processor


def _caption_one(mode: str, model, processor, img: Image.Image, prompt: str, max_new_tokens: int) -> str:
    if mode == "blip":
        inputs = processor(images=img, return_tensors="pt")
    else:
        # BLIP-2 supports optional prompt
        prompt = prompt or ""
        inputs = processor(images=img, text=prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text.strip()


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

    ap.add_argument("--mode", type=str, default="blip", choices=["blip", "blip2"])
    ap.add_argument("--model_id", type=str, default="Salesforce/blip-image-captioning-base")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--hf_cache", type=str, default="")

    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--suffix", type=str, default="")

    args = ap.parse_args()

    if not args.single and not args.in_dir:
        raise SystemExit("Provide --single or --in_dir")

    _setup_cache(args.hf_cache)
    device = _get_device(args.device)
    dtype = _get_dtype(args.dtype)

    if args.mode == "blip":
        model, processor = _load_blip(args.model_id, device, dtype)
    else:
        model, processor = _load_blip2(args.model_id, device, dtype)

    def caption_path(p: Path) -> str:
        img = Image.open(p).convert("RGB")
        cap = _caption_one(args.mode, model, processor, img, args.prompt, args.max_new_tokens)
        return _apply_affixes(cap, args.prefix, args.suffix)

    if args.single:
        p = Path(args.single)
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        print(caption_path(p))
        return

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")

    imgs = _iter_images(in_dir, args.recursive)
    if args.sort_method == "alphabetical":
        imgs = sorted(imgs, key=lambda p: p.name.lower())
    if args.start_from > 0:
        imgs = imgs[int(args.start_from):]

    total = len(imgs)
    if total == 0:
        print("No images found.")
        return

    done = 0
    for p in imgs:
        out_path = _sidecar_path(p, args.out_dir, args.caption_ext)
        if args.skip_existing and out_path.exists():
            continue
        try:
            cap = caption_path(p)
            out_path.write_text(cap + "\n", encoding="utf-8")
            done += 1
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    print(f"Done. Wrote {done} captions ({total} files scanned).")


if __name__ == "__main__":
    main()
