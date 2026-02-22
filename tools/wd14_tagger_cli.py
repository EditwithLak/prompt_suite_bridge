"""WD14 tagger CLI (ONNX, Hugging Face SmilingWolf repos).

Writes Danbooru-style tags as comma-separated text.

This implementation follows the common preprocessing used by WD14 taggers:
  - RGB -> BGR
  - pad to square with white (255)
  - resize to 448x448
  - float32, keep 0..255 range

Repo default: SmilingWolf/wd-v1-4-convnextv2-tagger-v2

Examples:
  python wd14_tagger_cli.py --single image.jpg
  python wd14_tagger_cli.py --in_dir F:\dataset --recursive --skip_existing --general_thr 0.35 --char_thr 0.85
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = 448


def _setup_cache(hf_cache: str):
    if not hf_cache:
        return
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_cache, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache, "transformers"))


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


def _preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Return float32 (1, 448, 448, 3) BGR with 0..255 range."""
    import cv2

    img = np.array(pil_img.convert("RGB"))
    img = img[:, :, ::-1]  # RGB -> BGR

    h, w = img.shape[:2]
    size = max(h, w)
    pad_x = size - w
    pad_y = size - h
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    img = np.pad(
        img,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    img = img.astype(np.float32)
    return img[None, ...]


def _download_files(repo_id: str, model_dir: str, filenames: list[str]) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    paths = {}
    for fn in filenames:
        paths[fn] = hf_hub_download(repo_id=repo_id, filename=fn, cache_dir=model_dir or None)
    return paths


def _load_tags(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data = rows[1:]

    # expected: tag_id,name,category,count
    name_idx = header.index("name") if "name" in header else 1
    cat_idx = header.index("category") if "category" in header else 2

    # WD14: first 4 are ratings, then general (category 0), then character (category 4)
    general_tags = [r[name_idx] for r in data if len(r) > cat_idx and r[cat_idx] == "0"]
    char_tags = [r[name_idx] for r in data if len(r) > cat_idx and r[cat_idx] == "4"]
    return general_tags, char_tags


def _run_onnx(onnx_path: str, batch: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: batch})[0]
    return out


def _tags_from_probs(
    probs: np.ndarray,
    general_tags: list[str],
    char_tags: list[str],
    general_thr: float,
    char_thr: float,
    remove_underscore: bool,
    undesired: set[str],
) -> str:
    # probs shape: (N, D)
    p = probs[0]
    combined: list[str] = []
    # Skip first 4 rating outputs
    tail = p[4:]
    for i, conf in enumerate(tail):
        if i < len(general_tags):
            if conf >= general_thr:
                tag = general_tags[i]
            else:
                continue
        else:
            j = i - len(general_tags)
            if j >= len(char_tags):
                continue
            if conf >= char_thr:
                tag = char_tags[j]
            else:
                continue

        if remove_underscore and len(tag) > 3:
            tag = tag.replace("_", " ")
        if tag in undesired:
            continue
        combined.append(tag)

    return ", ".join(combined)


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

    ap.add_argument("--repo_id", type=str, default="SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    ap.add_argument("--model_dir", type=str, default="")
    ap.add_argument("--general_thr", type=float, default=0.35)
    ap.add_argument("--char_thr", type=float, default=0.85)
    ap.add_argument("--remove_underscore", action="store_true")
    ap.add_argument("--undesired", type=str, default="")
    ap.add_argument("--hf_cache", type=str, default="")

    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--suffix", type=str, default="")

    args = ap.parse_args()
    if not args.single and not args.in_dir:
        raise SystemExit("Provide --single or --in_dir")

    _setup_cache(args.hf_cache)
    desired_dir = args.model_dir.strip()

    files = _download_files(args.repo_id, desired_dir, ["model.onnx", "selected_tags.csv"])
    onnx_path = files["model.onnx"]
    csv_path = files["selected_tags.csv"]

    general_tags, char_tags = _load_tags(csv_path)
    undesired = {t.strip() for t in args.undesired.split(",") if t.strip()}

    def tag_one(img_path: Path) -> str:
        img = Image.open(img_path)
        batch = _preprocess_image(img)
        probs = _run_onnx(onnx_path, batch)
        tags = _tags_from_probs(
            probs,
            general_tags,
            char_tags,
            args.general_thr,
            args.char_thr,
            args.remove_underscore,
            undesired,
        )
        t = tags.strip()
        if args.prefix:
            t = f"{args.prefix}{t}"
        if args.suffix:
            t = f"{t}{args.suffix}"
        return t.strip()

    if args.single:
        p = Path(args.single)
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        print(tag_one(p))
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
            tags = tag_one(p)
            out_path.write_text(tags + "\n", encoding="utf-8")
            done += 1
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    print(f"Done. Wrote {done} tag files ({total} files scanned).")


if __name__ == "__main__":
    main()
