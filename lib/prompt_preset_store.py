
import json
import shutil
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXT_ROOT = Path(__file__).resolve().parents[1]
USER_DATA = EXT_ROOT / "user_data"
PRESETS_PATH = USER_DATA / "prompt_presets.json"
ASSETS_DIR = USER_DATA / "prompt_preset_assets"

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure():
    USER_DATA.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not PRESETS_PATH.exists():
        PRESETS_PATH.write_text(json.dumps({"prompts": []}, indent=2), encoding="utf-8")

def _load() -> Dict[str, Any]:
    try:
        return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"prompts": []}

def _save(data: Dict[str, Any]):
    # Atomic-ish save to avoid partial writes.
    tmp = PRESETS_PATH.with_suffix(PRESETS_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(str(tmp), str(PRESETS_PATH))
    except Exception:
        PRESETS_PATH.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")

def _safe_ext(p: Path) -> str:
    ext = (p.suffix or "").lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return ext
    return ".png"

def _copy_asset(src_path: str, dest_dir: Path, stem: str, index: int = 0) -> str:
    if not src_path:
        return ""
    try:
        src = Path(src_path)
        if not src.exists():
            return ""
        ext = _safe_ext(src)
        name = f"{stem}{'' if index<=0 else f'_{index:02d}'}{ext}"
        dst = dest_dir / name
        # avoid overwrite
        k = 2
        while dst.exists():
            name = f"{stem}{'' if index<=0 else f'_{index:02d}'}_{k}{ext}"
            dst = dest_dir / name
            k += 1
        shutil.copy2(src, dst)
        return str(dst)
    except Exception:
        return ""


class PromptPresetStore:
    """
    Stores prompt presets (positive/negative) + optional links to Vault assets.
    Lives in user_data/ so updates won't wipe user content.
    """
    def __init__(self):
        _ensure()
        self.data = _load()

    def save(self):
        _save(self.data)

    def list_choices(self, q: str = "") -> List[Tuple[str, str]]:
        q = (q or "").strip().lower()
        out: List[Tuple[str, str]] = []
        for p in self.data.get("prompts", []):
            pid = p.get("id") or ""
            title = p.get("title") or "Untitled"
            if not pid:
                continue
            if q and q not in title.lower():
                continue
            out.append((title, pid))
        return out

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        for p in self.data.get("prompts", []):
            if p.get("id") == pid:
                return p
        return None

    def upsert(
        self,
        pid: str,
        title: str,
        positive: str,
        negative: str,
        linked: Dict[str, Any],
        cn_routing: Dict[str, Any],
        strengths: Dict[str, Any],
        assets: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        title = (title or "").strip()
        if not title:
            return None

        item = self.get(pid) if pid else None
        if item is None:
            pid = str(uuid.uuid4())
            item = {"id": pid, "created": _now_iso()}
            self.data.setdefault("prompts", []).append(item)

        # Copy linked assets (maps / composition / reference) into user_data for portability
        assets_dir = ASSETS_DIR / pid
        assets_dir.mkdir(parents=True, exist_ok=True)
        copied_assets: Dict[str, Any] = {}
        if assets:
            maps = (assets.get("maps") or {}) if isinstance(assets, dict) else {}
            comp = (assets.get("composition") or []) if isinstance(assets, dict) else []
            refs = (assets.get("reference") or []) if isinstance(assets, dict) else []
            copied_maps = {
                "canny": _copy_asset(maps.get("canny", ""), assets_dir, "map_canny"),
                "depth": _copy_asset(maps.get("depth", ""), assets_dir, "map_depth"),
                "openpose": _copy_asset(maps.get("openpose", ""), assets_dir, "map_openpose"),
            }
            copied_comp = []
            for i, p in enumerate(comp, start=1):
                cp = _copy_asset(p, assets_dir, "composition", i)
                if cp:
                    copied_comp.append(cp)
            copied_refs = []
            for i, p in enumerate(refs, start=1):
                rp = _copy_asset(p, assets_dir, "reference", i)
                if rp:
                    copied_refs.append(rp)
            copied_assets = {"maps": copied_maps, "composition": copied_comp, "reference": copied_refs}

        item.update({
            "title": title,
            "positive": positive or "",
            "negative": negative or "",
            "linked": linked or {},
            "cn_routing": cn_routing or {"unit0": "canny", "unit1": "depth", "unit2": "openpose"},
            "strengths": strengths or {"canny": 1.0, "depth": 1.0, "openpose": 1.0},
            "assets": copied_assets if assets else (item.get("assets") or {}),
            "updated": _now_iso(),
        })
        self.save()
        return pid

    def delete(self, pid: str):
        self.data["prompts"] = [p for p in self.data.get("prompts", []) if p.get("id") != pid]
        self.save()

    def maybe_migrate_legacy(self):
        """
        Migrate old data/library.json prompts into prompt_presets.json (no duplicates).
        """
        marker = USER_DATA / ".legacy_prompt_presets_migrated_v1"
        if marker.exists():
            return
        legacy_lib = EXT_ROOT / "data" / "library.json"
        if not legacy_lib.exists():
            marker.write_text("no legacy", encoding="utf-8")
            return
        try:
            legacy = json.loads(legacy_lib.read_text(encoding="utf-8"))
        except Exception:
            marker.write_text("bad legacy", encoding="utf-8")
            return
        prompts = legacy.get("prompts", []) or []
        if not prompts:
            marker.write_text("empty", encoding="utf-8")
            return

        # Only migrate if currently empty
        if self.data.get("prompts"):
            marker.write_text("skipped - already has presets", encoding="utf-8")
            return

        for lp in prompts:
            title = lp.get("title") or "Legacy Prompt"
            pos = lp.get("prompt") or ""
            neg = lp.get("negative") or ""
            self.upsert(
                "",
                title + " (legacy)",
                pos,
                neg,
                {"mapset_id": ""},  # relink manually
                {"unit0": "canny", "unit1": "depth", "unit2": "openpose"},
                {"canny": 1.0, "depth": 1.0, "openpose": 1.0},
            )

        marker.write_text("done", encoding="utf-8")
