
import os
import re
import json
import random
import time
import subprocess
import shutil
import tempfile
import unicodedata
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import gradio as gr

from vault_store import VaultStore
from prompt_preset_store import PromptPresetStore
from usage_store import UsageStore

# -----------------------------
# Local GGUF LLM settings (Prompt Suite)
# -----------------------------
EXT_ROOT = Path(__file__).resolve().parents[1]
USER_DATA_DIR = EXT_ROOT / "user_data"
LLM_SETTINGS_PATH = USER_DATA_DIR / "llm_settings.json"
LLM_CLI_PATH = EXT_ROOT / "tools" / "llm_prompt_cli.py"


def _load_llm_settings() -> Dict[str, Any]:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    d: Dict[str, Any] = {}
    try:
        d = json.loads(LLM_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    # defaults
    d.setdefault("python_exe", r"F:\MyTools\LLm Helper\.venv\Scripts\python.exe")
    d.setdefault("model_path", "")
    d.setdefault("n_ctx", 4096)
    d.setdefault("max_tokens", 256)
    d.setdefault("temperature", 0.7)
    d.setdefault("top_p", 0.95)
    d.setdefault("repeat_penalty", 1.15)
    d.setdefault("gpu_layers", 0)
    d.setdefault("threads", 8)
    return d


def _save_llm_settings(d: Dict[str, Any]) -> None:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        LLM_SETTINGS_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
# -----------------------------
# Paths / helpers
# -----------------------------

_NUM_ITEM = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")
_BULLET_ITEM = re.compile(r"^\s*-\s+(.*\S)\s*$")
_H1 = re.compile(r"^\s*#\s+(.*\S)\s*$")
_H2 = re.compile(r"^\s*##\s+(.*\S)\s*$")
_TAG = re.compile(r"\[([A-Za-z0-9_+-]+)\]")

EXCLUDE_MOMENT_FILES = {
    "Outfits.md",
    "Hairstyles Lists (Extra Colors Allowed).md",
    "# Locations Lists (with Time of Day + Vibe + Mood).md",
}

# -----------------------------
# Vault Tag/Packs Typeahead (Prompt Suite)
# -----------------------------
def _ps_norm_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s_-]+", " ", s).strip()
    return s

def _ps_active_token(prompt: str) -> str:
    p = (prompt or "")
    # work on the last line, and last comma-separated segment
    last_nl = p.rfind("\n")
    seg = p[last_nl+1:] if last_nl >= 0 else p
    last_comma = seg.rfind(",")
    token = seg[last_comma+1:] if last_comma >= 0 else seg
    token = token.strip()
    # allow optional triggers like @tag or #pack
    if token.startswith("@") or token.startswith("#"):
        token = token[1:].strip()
    return token

def _ps_replace_active_token(prompt: str, replacement: str) -> str:
    p = (prompt or "")
    rep = (replacement or "").strip()
    if not rep:
        return p
    last_nl = p.rfind("\n")
    line_start = last_nl+1 if last_nl >= 0 else 0
    last_comma = p.rfind(",", line_start)
    if last_comma >= 0:
        head = p[:last_comma+1].rstrip()
        # ensure comma then space
        if not head.endswith(","):
            head += ","
        head += " "
        return head + rep
    # no comma in the last line
    head = p[:line_start]
    prefix = p[line_start:].rstrip()
    # if there is already text, separate with comma
    if prefix:
        return head + prefix + ", " + rep
    return head + rep

def _ps_build_suggestions(token: str, limit: int = 15):
    raw = (token or "").strip()
    raw_l = raw.lower()

    # Detect explicit modes like: <lora:foo, lora:foo, lora foo, ti:foo, ti foo
    mode = None
    q = raw
    if raw_l.startswith("<lora:"):
        mode = "lora"
        q = raw[len("<lora:"):]
    elif raw_l.startswith("lora:"):
        mode = "lora"
        q = raw[len("lora:"):]
    elif raw_l.startswith("lora "):
        mode = "lora"
        q = raw[len("lora "):]
    elif raw_l in ("lora", "loras"):
        mode = "lora"
        q = ""
    elif raw_l.startswith("ti:"):
        mode = "ti"
        q = raw[len("ti:"):]
    elif raw_l.startswith("ti "):
        mode = "ti"
        q = raw[len("ti "):]
    elif raw_l in ("ti", "tis", "embedding", "embeddings"):
        mode = "ti"
        q = ""

    tok = _ps_norm_token(q if mode else token)

    # default: require at least 2 chars for suggestions
    if mode is None and len(tok) < 2:
        return []

    s = VaultStore()
    out = []

    def _fmt_w(x: float) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "0.80"

    # -----------------
    # LoRAs / TIs
    # -----------------
    if mode in ("lora", "ti"):
        kind = mode
        for it in (s.data.get("loras") or []):
            if (it.get("kind") or "lora") != kind:
                continue
            if not bool(it.get("enabled", True)):
                continue
            lid = it.get("id") or ""
            rel = (it.get("rel") or it.get("name") or "").strip()
            name = (it.get("name") or "").strip()
            cat = (it.get("category") or "").strip()
            if not lid or not name or not rel:
                continue
            label = f"{cat} › {rel}" if cat else rel
            hay = _ps_norm_token(label + " " + " ".join(it.get("triggers") or []) + " " + " ".join(it.get("keywords") or []))
            if tok and tok not in hay:
                continue
            if kind == "lora":
                w = _fmt_w(it.get("default_strength", 0.8))
                out.append((f"🧩 LoRA › {label}  ({w})", f"LORA:{lid}"))
                tr = [t.strip() for t in (it.get("triggers") or []) if t.strip()]
                if tr:
                    out.append((f"🧩 LoRA › {label}  ✨+triggers", f"LORAT:{lid}"))
            else:
                out.append((f"🧷 TI › {label}", f"TI:{lid}"))
            if len(out) >= limit:
                return out
        return out

    # -----------------
    # Keywords (back-compat: tags)
    # -----------------
    # Library-driven keywords live in `libraries/*__keywords*.md`
    kws = s._load_keywords()
    for tid, t in kws.items():
        name = (t.get("name") or "").strip()
        cat = (t.get("category") or "misc").strip()
        if not tid or not name:
            continue
        if not bool(t.get("enabled", True)):
            continue
        desc = (t.get("desc") or t.get("description") or "").strip()
        hay = _ps_norm_token(name + " " + " ".join(t.get("aliases") or []) + " " + cat)
        if tok in hay:
            out.append((f"🏷️ {cat} › {name}", f"TAG:{tid}"))
            if desc:
                out.append((f"🏷️ {cat} › {name}  ✨+desc", f"TAGD:{tid}"))
            if len(out) >= limit:
                return out

# -----------------
    # Library-driven packs live in `libraries/*__packs*.md`
    packs = s._load_packs()
    for pid, p in packs.items():
        title = (p.get("title") or "").strip()
        cat = (p.get("category") or "misc").strip()
        if not pid or not title:
            continue
        hay = _ps_norm_token(title + " " + cat + " " + (p.get("note") or ""))
        if tok in hay:
            out.append((f"📦 {cat} › {title}", f"PACK:{pid}"))
            if len(out) >= limit:
                return out

# -----------------
    # Bonus: if user types "lora"/"ti" without explicit mode, show a few anyway
    # -----------------
    if tok in ("lora", "loras"):
        return _ps_build_suggestions("lora", limit=limit)
    if tok in ("ti", "embedding", "embeddings"):
        return _ps_build_suggestions("ti", limit=limit)

    return out


def _ps_preview_for_value(val: str) -> str:
    if not val:
        return ""
    s = VaultStore()
    if val.startswith("TAG:") or val.startswith("TAGD:"):
        tid = val.split(":", 1)[1]
        t = s.get_tag(tid)
        if not t:
            return ""
        desc = (t.get("desc") or t.get("description") or "").strip()
        return f"**Description:** {desc}" if desc else ""

    if val.startswith("PACK:"):
        pid = val.split(":", 1)[1]
        p = s.get_pack(pid)
        if not p:
            return ""
        tids = p.get("tag_ids") or []
        names = []
        for tid in tids:
            t = s.get_tag(tid)
            if t and t.get("name"):
                names.append(t["name"])
        if not names:
            return ""
        return "**Pack contains:** " + ", ".join(names[:40]) + ("" if len(names) <= 40 else " …")

    if val.startswith("LORA:") or val.startswith("LORAT:") or val.startswith("TI:"):
        lid = val.split(":", 1)[1]
        it = s.get_lora(lid) or {}
        if not it:
            return ""
        rel = (it.get("rel") or it.get("name") or "").strip()
        cat = (it.get("category") or "").strip()
        tr = ", ".join([x for x in (it.get("triggers") or []) if x])
        kw = ", ".join([x for x in (it.get("keywords") or []) if x])
        note = (it.get("notes") or "").strip()
        parts = []
        parts.append(f"**{it.get('kind','lora').upper()}**: {cat + ' › ' if cat else ''}{rel}")
        if tr:
            parts.append(f"**Triggers:** {tr}")
        if kw:
            parts.append(f"**Keywords:** {kw}")
        if note:
            parts.append(f"**Notes:** {note}")
        return "\n\n".join(parts)

    return ""


def _ps_label_for_value(val: str) -> str:
    """Short label for recents/favorites lists."""
    if not val:
        return ""
    s = VaultStore()
    try:
        if val.startswith("TAG"):
            tid = val.split(":", 1)[1]
            t = s.get_tag(tid) or {}
            cat = (t.get("category") or "misc").strip()
            name = (t.get("name") or "").strip()
            if cat and name:
                return f"🏷️ {cat} › {name}" + (" ✨+desc" if val.startswith("TAGD:") else "")
        if val.startswith("PACK:"):
            pid = val.split(":", 1)[1]
            p = s.get_pack(pid) or {}
            cat = (p.get("category") or "misc").strip()
            title = (p.get("title") or "").strip()
            if cat and title:
                return f"📦 {cat} › {title}"
        if val.startswith("LORA") or val.startswith("TI:"):
            lid = val.split(":", 1)[1]
            it = s.get_lora(lid) or {}
            kind = (it.get("kind") or "lora").strip().lower()
            rel = (it.get("rel") or it.get("name") or "").strip()
            cat = (it.get("category") or "").strip()
            label = f"{cat} › {rel}" if cat else rel
            if kind == "ti":
                return f"🧷 TI › {label}" if label else ""
            # lora
            w = it.get("default_strength", 0.8)
            try:
                w = float(w)
            except Exception:
                w = 0.8
            base = f"🧩 LoRA › {label}  ({w:.2f})" if label else ""
            if val.startswith("LORAT:"):
                base += " ✨+triggers"
            return base
    except Exception:
        pass
    return val



def _ext_root() -> str:
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, ".."))

def _lib_dir(default_path: str = "") -> str:
    if default_path and os.path.isdir(default_path):
        return default_path
    return os.path.join(_ext_root(), "libraries")

def _packs_dir() -> str:
    return os.path.join(_ext_root(), "packs")

def _presets_path() -> str:
    return os.path.join(_ext_root(), "presets", "sdxl_presets.json")

def _user_data_dir() -> str:
    d = os.path.join(_ext_root(), "user_data")
    os.makedirs(d, exist_ok=True)
    return d

def _saved_presets_path() -> str:
    return os.path.join(_user_data_dir(), "saved_presets.json")

def _exports_dir(default_path: str = "") -> str:
    if default_path and os.path.isdir(default_path):
        return default_path
    d = os.path.join(_ext_root(), "exports")
    os.makedirs(d, exist_ok=True)
    return d

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)

def _load_json(path: str, fallback: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback

def _list_files(folder: str, suffix: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(suffix.lower()):
            out.append(os.path.join(folder, fn))
    return sorted(out)

def _safe_choice(lst: List[str], fallback: str = "") -> str:
    return random.choice(lst) if lst else fallback

def _extract_tags(text: str) -> Tuple[str, List[str]]:
    tags = _TAG.findall(text or "")
    clean = _TAG.sub("", text or "")
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    # tidy separators
    clean = clean.strip(" -–—,")
    return clean, sorted(set([t.strip() for t in tags if t.strip()]))

def _sanitize_filename(s: str, default: str = "export") -> str:
    s = (s or "").strip()
    if not s:
        s = default
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or default

# -----------------------------
# Markdown parsing
# -----------------------------

def _parse_numbered_items(md_text: str) -> List[str]:
    items: List[str] = []
    for line in md_text.splitlines():
        m = _NUM_ITEM.match(line)
        if m:
            items.append(m.group(2).strip())
    return items

def _parse_bullets_under_heading(md_text: str, heading: str) -> List[str]:
    """
    Find a heading line (case-insensitive exact match after stripping),
    then collect bullet lines until next heading or horizontal rule.
    """
    lines = md_text.splitlines()
    target = heading.strip().lower()
    out: List[str] = []
    in_block = False
    for line in lines:
        if line.strip().lower() == target:
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("#"):
                break
            m = _BULLET_ITEM.match(line)
            if m:
                out.append(m.group(1).strip())
            if line.strip().startswith("---"):
                break
    return out

def _parse_locations(md_text: str) -> Tuple[Dict[str, List[str]], List[str], List[str], List[str]]:
    categories: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in md_text.splitlines():
        h = _H2.match(line)
        if h:
            current = h.group(1).strip()
            categories[current] = []
            continue
        m = _NUM_ITEM.match(line)
        if m and current:
            categories[current].append(m.group(2).strip())

    day = _parse_bullets_under_heading(md_text, "## Day / Time Options (Pick List)")
    vibe = _parse_bullets_under_heading(md_text, "## Vibe Options (Pick List)")
    mood = _parse_bullets_under_heading(md_text, "## Mood Options (Pick List)")
    return categories, day, vibe, mood

def _split_dash_fields(s: str) -> Tuple[str, str, str, str]:
    parts = [p.strip() for p in (s or "").split("—")]
    parts = [p for p in parts if p]
    while len(parts) < 4:
        parts.append("")
    return parts[0], parts[1], parts[2], parts[3]

# -----------------------------
# Outfits parsing (sets + pools)
# -----------------------------

def _parse_outfit_sets_enhanced(md_text: str) -> List[Dict[str, Any]]:
    """
    Extract "100 Streetwear Outfit Sets + Hair + Props (Original List Enhanced)" if present,
    else fall back to any numbered list.
    """
    lines = md_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "100 Streetwear Outfit Sets + Hair + Props" in line:
            start_idx = i
            break
    content = "\n".join(lines[start_idx:]) if start_idx is not None else md_text
    items = _parse_numbered_items(content)

    out: List[Dict[str, Any]] = []
    for it in items:
        raw = it
        outfit = it
        hair = ""
        props = ""

        if "— Hair:" in it:
            left, right = it.split("— Hair:", 1)
            outfit = left.strip()
            hair = right.strip()
            if "— Props:" in hair:
                hair, props = hair.split("— Props:", 1)
                hair = hair.strip()
                props = props.strip()
        elif "— Props:" in it:
            left, props = it.split("— Props:", 1)
            outfit = left.strip()
            props = props.strip()

        outfit_clean, tags = _extract_tags(outfit)

        label = outfit_clean or outfit
        if tags:
            label += " " + "".join([f"[{t}]" for t in tags])
        if hair:
            label += f" | Hair: {hair}"
        if props:
            label += f" | Props: {props}"

        out.append({"label": label, "outfit": outfit_clean or outfit, "hair": hair, "props": props, "tags": tags, "raw": raw})
    return out

def _slice_h1_blocks(md_text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (h1_title, block_text) for each # Heading section.
    """
    lines = md_text.splitlines()
    h1_positions: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _H1.match(line)
        if m:
            h1_positions.append((i, m.group(1).strip()))
    blocks: List[Tuple[str, str]] = []
    for idx, (start, title) in enumerate(h1_positions):
        end = h1_positions[idx + 1][0] if idx + 1 < len(h1_positions) else len(lines)
        blocks.append((title, "\n".join(lines[start:end])))
    return blocks

def _parse_pools_from_block(block_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse pools inside a single H1 block. Returns dict: category -> list[{text, tags}]
    Categories: tops, bottoms, footwear, accessories, colorways
    """
    pools = {"tops": [], "bottoms": [], "footwear": [], "accessories": [], "colorways": []}

    lines = block_text.splitlines()
    current_cat = None
    for line in lines:
        h2 = _H2.match(line)
        if h2:
            name = h2.group(1).lower()
            if "tops" in name:
                current_cat = "tops"
            elif "bottom" in name:
                current_cat = "bottoms"
            elif "shoe" in name or "footwear" in name:
                current_cat = "footwear"
            elif "accessories" in name or "gear" in name:
                current_cat = "accessories"
            else:
                current_cat = None
            continue

        m = _NUM_ITEM.match(line)
        if m and current_cat in {"tops", "bottoms", "footwear", "accessories"}:
            item = m.group(2).strip()
            clean, tags = _extract_tags(item)
            pools[current_cat].append({"text": clean or item, "tags": tags})
            continue

        # colorways bullet list
        if line.strip().lower() == "## colorways / material vibes (optional pick list)".lower():
            # handled in second pass below
            pass

    # colorways (bullets under the heading)
    colorways = _parse_bullets_under_heading(block_text, "## COLORWAYS / MATERIAL VIBES (Optional Pick List)")
    pools["colorways"] = [{"text": c, "tags": []} for c in colorways]
    return pools

def _parse_outfits_document(md_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Dict[str, Any]]]], List[str]]:
    """
    Returns:
      - enhanced_sets
      - pool_groups: group_title -> pools dict
      - all_style_tags from pools
    """
    enhanced_sets = _parse_outfit_sets_enhanced(md_text)

    pool_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    all_tags: set = set()

    for title, block in _slice_h1_blocks(md_text):
        if "generator pools" in title.lower():
            pools = _parse_pools_from_block(block)
            pool_groups[title] = pools
            for cat in ("tops", "bottoms", "footwear", "accessories"):
                for it in pools.get(cat, []):
                    for t in it.get("tags", []):
                        all_tags.add(t)

    return enhanced_sets, pool_groups, sorted(all_tags)

def _filter_pool_items(items: List[Dict[str, Any]], required_tags: List[str]) -> List[Dict[str, Any]]:
    if not required_tags:
        return items
    req = set(required_tags)
    out = []
    for it in items:
        if req.issubset(set(it.get("tags", []))):
            out.append(it)
    return out

def _build_outfit_from_pools(pool_groups: Dict[str, Any], group_title: str, required_tags: List[str], accessories_n: int, colorway: str) -> str:
    g = (pool_groups or {}).get(group_title, {})
    tops = _filter_pool_items(g.get("tops", []), required_tags)
    bottoms = _filter_pool_items(g.get("bottoms", []), required_tags)
    footwear = _filter_pool_items(g.get("footwear", []), required_tags)
    accessories = _filter_pool_items(g.get("accessories", []), required_tags)

    # fallback gracefully: if filters too strict, relax to any
    if required_tags:
        if not tops: tops = g.get("tops", [])
        if not bottoms: bottoms = g.get("bottoms", [])
        if not footwear: footwear = g.get("footwear", [])
        if not accessories: accessories = g.get("accessories", [])

    pick = lambda lst: (random.choice(lst)["text"] if lst else "")
    a_n = max(0, int(accessories_n or 0))

    top = pick(tops)
    bot = pick(bottoms)
    shoe = pick(footwear)
    acc = []
    for _ in range(a_n):
        t = pick(accessories)
        if t and t not in acc:
            acc.append(t)

    parts = [p for p in [top, bot, shoe] if p]
    if acc:
        parts.append(" + ".join(acc))
    out = " + ".join(parts) if parts else ""

    if colorway:
        out = f"{out} — colorway/material vibe: {colorway}" if out else f"colorway/material vibe: {colorway}"
    return out.strip()

# -----------------------------
# Packs / presets / moments discovery
# -----------------------------

def _list_json_files(folder: str) -> List[str]:
    return _list_files(folder, ".json")

def _load_pack(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _pack_choices(packs: List[Dict]) -> Tuple[List[str], Dict[str, Dict]]:
    names = []
    by_name: Dict[str, Dict] = {}
    for p in packs:
        name = p.get("name") or os.path.basename(p.get("_path", "pack.json"))
        names.append(name)
        by_name[name] = p
    return names, by_name

def _discover_moment_libraries(lib_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: display_name -> {file, items}
    """
    libs: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(lib_path):
        return libs

    for fn in sorted(os.listdir(lib_path)):
        if not fn.lower().endswith(".md"):
            continue
        if fn in EXCLUDE_MOMENT_FILES:
            continue
        path = os.path.join(lib_path, fn)
        md = _read_text(path)
        items = _parse_numbered_items(md)
        if not items:
            continue
        display = os.path.splitext(fn)[0]
        libs[display] = {"file": fn, "items": items}
    return libs

def _load_presets_config() -> List[Dict[str, Any]]:
    cfg = _load_json(_presets_path(), {"presets": []})
    presets = cfg.get("presets", []) if isinstance(cfg, dict) else []
    # basic validation / normalization
    out = []
    for p in presets:
        if not isinstance(p, dict):
            continue
        label = p.get("label")
        lib = p.get("moment_library")
        if not label or not lib:
            continue
        out.append({
            "id": p.get("id") or _sanitize_filename(label),
            "label": label,
            "moment_library": lib,
            "default_subject": p.get("default_subject") or "adult subject",
            "family_safe": bool(p.get("family_safe", False)),
        })
    return out

# -----------------------------
# Prompt building
# -----------------------------

def _compose_prompt(
    pack: Dict,
    family_safe: bool,
    subject: str,
    moment: str,
    outfit: str,
    hair: str,
    props: str,
    place: str,
    time_of_day: str,
    vibe: str,
    mood: str,
    camera_key: str,
    lighting_key: str,
    extra_tags: str,
    tag_style: str,
) -> str:
    pos = pack.get("positive", {})
    prefix = (pos.get("prefix", "") or "").strip()
    suffix = (pos.get("suffix", "") or "").strip()

    camera = (pos.get("camera", {}) or {}).get(camera_key or "", "")
    lighting = (pos.get("lighting", {}) or {}).get(lighting_key or "", "")

    parts: List[str] = []

    if subject:
        parts.append(subject)

    if family_safe:
        parts.append("SFW, wholesome, warm, safe-for-work")

    if moment:
        parts.append(moment)

    if outfit:
        parts.append(outfit)
    if hair:
        parts.append(f"hair: {hair}")
    if props:
        parts.append(f"props: {props}")

    setting_bits = [b for b in [place, time_of_day, vibe, mood] if b]
    if setting_bits:
        parts.append("setting: " + ", ".join(setting_bits))

    if camera:
        parts.append(camera)
    if lighting:
        parts.append(lighting)

    if extra_tags:
        parts.append(extra_tags.strip())

    all_bits: List[str] = []
    if prefix:
        all_bits.append(prefix)
    all_bits += parts
    if suffix:
        all_bits.append(suffix)

    def _clean(x: str) -> str:
        return (x or "").strip().strip(",").strip()

    if tag_style == "Paragraph (Photo SDXL)":
        intro_bits = [b for b in [subject, moment] if b]
        desc_bits: List[str] = []
        if outfit:
            desc_bits.append(f"wearing {outfit}")
        if hair:
            desc_bits.append(f"with {hair} hair")
        if props:
            desc_bits.append(f"with {props}")

        setting_items = [b for b in [place, time_of_day, vibe, mood] if b]
        setting_clause = f"set in {', '.join(setting_items)}" if setting_items else ""

        cam_light_bits: List[str] = []
        if camera:
            cam_light_bits.append(camera)
        if lighting:
            cam_light_bits.append(lighting)
        if extra_tags and extra_tags.strip():
            cam_light_bits.append(extra_tags.strip())

        segments: List[str] = []
        if prefix:
            segments.append(prefix)
        if intro_bits:
            segments.append(", ".join(intro_bits))
        if family_safe:
            segments.append("wholesome, warm, safe-for-work")
        if desc_bits:
            segments.append(", ".join(desc_bits))
        if setting_clause:
            segments.append(setting_clause)
        if cam_light_bits:
            segments.append(", ".join([_clean(b) for b in cam_light_bits if _clean(b)]))
        if suffix:
            segments.append(suffix)

        # One-line paragraph prompt (photoreal-friendly)
        return ". ".join([_clean(s) for s in segments if _clean(s)])

    if tag_style == "Comma tags (SDXL)":
        return ", ".join([_clean(b) for b in all_bits if _clean(b)])
    return " • ".join([_clean(b) for b in all_bits if _clean(b)])

def _compose_negative(pack: Dict, male_only: bool, extra_negative: str) -> str:
    neg = pack.get("negative", {})
    base = (neg.get("base") or "").strip()
    addon = (neg.get("male_only_addon") or "").strip()
    parts = []
    if base:
        parts.append(base)
    if male_only and addon:
        parts.append(addon)
    if extra_negative and extra_negative.strip():
        parts.append(extra_negative.strip())
    return ", ".join([p.strip().strip(",") for p in parts if p.strip()])

# -----------------------------
# Saved presets (user_data)
# -----------------------------

def _load_saved_presets() -> List[Dict[str, Any]]:
    data = _load_json(_saved_presets_path(), {"version": 1, "items": []})
    items = data.get("items", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("name") and isinstance(it.get("data"), dict):
            out.append(it)
    # favorites first, then name
    out.sort(key=lambda x: (0 if x.get("favorite") else 1, (x.get("name") or "").lower()))
    return out

def _save_saved_presets(items: List[Dict[str, Any]]) -> None:
    _write_text(_saved_presets_path(), json.dumps({"version": 1, "items": items}, indent=2, ensure_ascii=False))

def _saved_choices(items: List[Dict[str, Any]]) -> List[str]:
    out = []
    for it in items:
        name = it.get("name", "")
        if it.get("favorite"):
            out.append(f"⭐ {name}")
        else:
            out.append(name)
    return out

def _strip_star(name: str) -> str:
    return (name or "").replace("⭐", "").strip()

# -----------------------------
# Cache
# -----------------------------

_CACHE: Dict[str, Any] = {}

def _reload(lib_path: str) -> Dict[str, Any]:
    lib_path = _lib_dir(lib_path)

    # packs
    pack_objs: List[Dict] = []
    for p in _list_json_files(_packs_dir()):
        obj = _load_pack(p)
        obj["_path"] = p
        pack_objs.append(obj)
    pack_names, pack_by_name = _pack_choices(pack_objs)

    # locations
    loc_path = os.path.join(lib_path, "# Locations Lists (with Time of Day + Vibe + Mood).md")
    loc_md = _read_text(loc_path) if os.path.exists(loc_path) else ""
    locations, day_list, vibe_list, mood_list = _parse_locations(loc_md) if loc_md else ({}, [], [], [])

    # outfits + pools
    outfits_path = os.path.join(lib_path, "Outfits.md")
    outfits_md = _read_text(outfits_path) if os.path.exists(outfits_path) else ""
    outfit_sets, pool_groups, style_tags = _parse_outfits_document(outfits_md) if outfits_md else ([], {}, [])

    # hair
    hair_path = os.path.join(lib_path, "Hairstyles Lists (Extra Colors Allowed).md")
    hair_md = _read_text(hair_path) if os.path.exists(hair_path) else ""
    hair_list = _parse_numbered_items(hair_md) if hair_md else []

    # moments
    moment_libs = _discover_moment_libraries(lib_path)

    # presets config
    presets = _load_presets_config()

    # saved presets
    saved = _load_saved_presets()

    _CACHE.update({
        "lib_path": lib_path,
        "packs": pack_objs,
        "pack_by_name": pack_by_name,
        "pack_names": pack_names,
        "locations": locations,
        "day_list": day_list,
        "vibe_list": vibe_list,
        "mood_list": mood_list,
        "outfit_sets": outfit_sets,
        "pool_groups": pool_groups,
        "style_tags": style_tags,
        "hair": hair_list,
        "moment_libs": moment_libs,
        "presets": presets,
        "saved": saved,
    })

    return _CACHE

# -----------------------------
# UI
# -----------------------------

def build_prompt_builder_ui(queue_cb=None, clear_queue_cb=None):
    # No legacy .md builder load needed (Vault-driven workflow)

    # Vault + prompt preset stores (persist in user_data/)
    try:
        _vs = VaultStore()
        _vs.maybe_migrate_legacy_prompt_map_vault()
    except Exception:
        _vs = VaultStore()
    try:
        _ps = PromptPresetStore()
        _ps.maybe_migrate_legacy()
    except Exception:
        _ps = PromptPresetStore()

    def _preset_labels() -> List[str]:
        return [p["label"] for p in (_CACHE.get("presets") or [])]

    def _preset_by_label(label: str) -> Optional[Dict[str, Any]]:
        for p in (_CACHE.get("presets") or []):
            if p.get("label") == label:
                return p
        return None

    def _moment_lib_choices() -> List[str]:
        return sorted(list((_CACHE.get("moment_libs") or {}).keys()))

    def _moment_lib_for_preset(label: str) -> Optional[str]:
        p = _preset_by_label(label)
        if not p:
            return None
        # map file -> display name
        target_file = p.get("moment_library")
        for disp, obj in (_CACHE.get("moment_libs") or {}).items():
            if obj.get("file") == target_file:
                return disp
        return None

    def _moment_items_for_display(disp: str) -> List[str]:
        return ((_CACHE.get("moment_libs") or {}).get(disp, {}) or {}).get("items", []) or []

    def _default_subject_for_preset(label: str) -> str:
        p = _preset_by_label(label)
        return (p.get("default_subject") if p else None) or "adult subject"

    def _family_safe_for_preset(label: str) -> bool:
        p = _preset_by_label(label)
        return bool(p.get("family_safe", False)) if p else False

    with gr.Column():
        gr.Markdown("## 🧩 Prompt Composer — Prompt Suite (Vault-first) v0.4.15")
        gr.Markdown(
            "### Your new flow ✅\n"
            "1) **Vault + Maps** → create/edit **tags / packs / mapsets / LoRA metadata**\n"
            "2) **Map Generator** → generate maps and save them as a named mapset\n"
            "3) **Prompt Composer (this tab)** → write prompts in the panel + use **inline autocomplete**\n"
            "4) Send to **txt2img / img2img** + queue maps to **ControlNet**\n\n"
            "**Tip:** type inside the **Positive prompt** box to get dropdown suggestions (tags, packs, LoRAs, TIs + optional descriptions)."
        )

        with gr.Accordion("Output", open=True):
            with gr.Column(elem_id="ps_pos_wrap"):
                positive_out = gr.Textbox(label="Positive prompt", lines=6, elem_id="ps_pos_prompt")

                # 🔎 Local typeahead (Vault tags/packs) — dropdown overlays inside the prompt box
                vault_suggest_dd = gr.Dropdown(
                    label="",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False,
                    elem_id="ps_pos_suggest",
                )

                gr.HTML(r"""
<style>
#ps_pos_wrap { position: relative; }
#ps_pos_wrap #ps_pos_suggest { position:absolute !important; left:12px; right:12px; bottom:12px; z-index:9999; }
#ps_pos_wrap #ps_pos_suggest label { display:none !important; }
#ps_pos_wrap #ps_pos_suggest .wrap { margin:0 !important; }
#ps_pos_wrap #ps_pos_suggest .gradio-dropdown,
#ps_pos_wrap #ps_pos_suggest select { width:100% !important; }
</style>
""")


                # -----------------------------
                # 🎭 Character creation (SFW)
                # - Filters keyword/pack libraries by gender + era
                # - Optional "restricted" libraries (non-explicit only) via toggle
                # -----------------------------
                _CC_EXPLICIT_TOKENS = {
                    "penis","vagina","anus","cock","dick","pussy",
                    "blowjob","handjob","cum","semen","orgasm",
                    "fuck","fucking","nude","nudity","sex"
                }

                def _cc_is_explicit(text: str) -> bool:
                    t = _ps_norm_token(text).replace(" ", "_")
                    for w in _CC_EXPLICIT_TOKENS:
                        if w in t:
                            return True
                    return False

                def _cc_parse_category(cat: str) -> dict:
                    raw = (cat or "").strip()
                    tokens = [t for t in re.split(r"[_\s]+", raw) if t]
                    out = {"category": raw, "gender": "", "era": "any", "rating": "safe", "group": "misc"}
                    if not tokens:
                        return out
                    g = tokens[0].lower()
                    if g not in ("male", "female", "unisex"):
                        # Character builder only uses gender-prefixed libs
                        out["gender"] = ""
                        out["group"] = "_".join(tokens).lower()
                        return out
                    out["gender"] = g
                    rest = tokens[1:]

                    # era
                    if rest and rest[0].lower() in ("modern", "fantasy", "futuristic", "any"):
                        out["era"] = rest[0].lower()
                        rest = rest[1:]

                    # rating (explicit token OR positional token)
                    # - allow either "..._restricted_..." anywhere, or "..._<safe|restricted>_..."
                    rating = "safe"
                    if rest and rest[0].lower() in ("safe", "restricted"):
                        rating = rest[0].lower()
                        rest = rest[1:]
                    elif any(t.lower() == "restricted" for t in rest):
                        rating = "restricted"
                        rest = [t for t in rest if t.lower() != "restricted"]
                    out["rating"] = rating

                    out["group"] = "_".join([t.lower() for t in rest]) or "misc"
                    return out

                def _cc_entries(kind: str, gender: str, era: str, show_restricted: bool) -> List[dict]:
                    kind = (kind or "kw").strip().lower()
                    lib_kind = "packs" if kind in ("pack", "packs") else "keywords"

                    g = (gender or "").strip().lower()
                    e = (era or "any").strip().lower()

                    cats = _vs.list_categories([lib_kind])
                    entries: List[dict] = []
                    for cat in cats:
                        meta = _cc_parse_category(cat)
                        if not meta.get("gender"):
                            continue

                        # hard exclude explicit libs from character builder
                        if _cc_is_explicit(cat) or _cc_is_explicit(meta.get("group","")):
                            continue

                        # gender: include selected + unisex always
                        mg = meta.get("gender")
                        if mg != "unisex" and mg != g:
                            continue

                        # era: allow "any" libs always
                        me = meta.get("era","any")
                        if e != "any" and me not in ("any", e):
                            continue

                        # restricted toggle
                        if meta.get("rating") == "restricted" and not show_restricted:
                            continue

                        entries.append(meta)

                    # stable ordering: group then category
                    entries.sort(key=lambda x: (x.get("group",""), x.get("category","")))
                    return entries

                def _cc_group_choices(entries: List[dict]) -> List[Tuple[str, str]]:
                    groups = []
                    seen = set()
                    for m in entries or []:
                        g = m.get("group") or "misc"
                        if g in seen:
                            continue
                        seen.add(g)
                        groups.append(g)
                    return [(g.replace("_", " "), g) for g in groups]

                def _cc_lib_choices(entries: List[dict], group: str) -> List[Tuple[str, str]]:
                    out = []
                    for m in entries or []:
                        if (m.get("group") or "misc") == (group or "misc"):
                            cat = m.get("category") or ""
                            if cat:
                                out.append((cat, cat))
                    return out

                def _cc_item_choices(kind: str, q: str, category: str) -> List[Tuple[str, str]]:
                    if not category:
                        return []
                    kind = (kind or "kw").strip().lower()
                    if kind in ("pack", "packs"):
                        return _vs.list_pack_choices(q or "", category=category)
                    # keywords
                    choices = _vs.list_tag_choices(q or "", category=category)
                    # extra safety: filter explicit keyword names too
                    safe = []
                    for label, kid in choices:
                        k = _vs.get_tag(kid) or {}
                        nm = k.get("name") or ""
                        if _cc_is_explicit(nm) or _cc_is_explicit(label):
                            continue
                        safe.append((label, kid))
                    return safe

                def _cc_split_csv(s: str) -> List[str]:
                    s = (s or "").strip()
                    if not s:
                        return []
                    parts = [p.strip() for p in s.split(",")]
                    return [p for p in parts if p]

                def _cc_norm_piece(s: str) -> str:
                    s = (s or "").strip().lower()
                    s = re.sub(r"[\s_-]+", "", s)
                    s = re.sub(r"[^a-z0-9]+", "", s)
                    return s

                def _cc_append_text(buf: str, add: str) -> str:
                    buf_items = _cc_split_csv(buf)
                    add_items = _cc_split_csv(add)
                    seen = {_cc_norm_piece(x) for x in buf_items}
                    out = list(buf_items)
                    for it in add_items:
                        n = _cc_norm_piece(it)
                        if not n or n in seen:
                            continue
                        seen.add(n)
                        out.append(it)
                    return ", ".join(out).strip(", ").strip()

                def _cc_insert_into_positive(pos_prompt: str, slot: str, insert_mode: str, char_text: str) -> Tuple[str, str]:
                    slot = (slot or "chr1").strip().lower()
                    insert_mode = (insert_mode or "chrblock").strip().lower()
                    char_text = (char_text or "").strip().strip(",")
                    if not char_text:
                        return pos_prompt, "⚠️ Nothing to insert."

                    marker = f"(({slot}))"
                    p = pos_prompt or ""

                    def _inject_after_break(which: str) -> str:
                        tokens = re.split(r"(\bBREAK\b)", p)
                        idxs = [i for i, t in enumerate(tokens) if t == "BREAK"]
                        if not idxs:
                            return _cc_append_text(p, char_text)
                        idx = idxs[0] if which == "first" else idxs[-1]
                        if idx + 1 >= len(tokens):
                            tokens.append("")
                        seg = tokens[idx + 1]
                        seg = seg.strip()
                        if seg:
                            seg_new = _cc_append_text(char_text, seg)  # char first
                        else:
                            seg_new = char_text
                        tokens[idx + 1] = " " + seg_new.strip().strip(",") + " "
                        return "".join(tokens)

                    if insert_mode in ("after_first_break", "after_first"):
                        newp = _inject_after_break("first")
                    elif insert_mode in ("after_last_break", "after_last"):
                        newp = _inject_after_break("last")
                    elif insert_mode in ("append_end", "append"):
                        newp = _cc_append_text(p, char_text)
                    else:
                        # inside ((chrX)) block
                        if marker in p:
                            i = p.find(marker) + len(marker)
                            m = re.search(r"\bBREAK\b", p[i:])
                            j = i + (m.start() if m else len(p) - i)
                            seg = p[i:j]
                            seg_new = _cc_append_text(seg, char_text)
                            newp = p[:i] + (" " if seg_new and not seg_new.startswith(" ") else "") + seg_new.strip() + p[j:]
                        else:
                            block = f"{marker} {char_text} BREAK"
                            newp = (p.rstrip() + "\n" + block) if p.strip() else block

                    newp = re.sub(r"\s+,", ",", newp)
                    newp = re.sub(r",\s*,+", ", ", newp)
                    newp = re.sub(r"\s{2,}", " ", newp)
                    return newp.strip(), "✅ Inserted."

                # Initial choices (build-time)
                _cc_init_entries = _cc_entries("kw", "male", "any", False)
                _cc_init_groups = _cc_group_choices(_cc_init_entries)
                _cc_init_group_val = _cc_init_groups[0][1] if _cc_init_groups else None
                _cc_init_libs = _cc_lib_choices(_cc_init_entries, _cc_init_group_val) if _cc_init_group_val else []
                _cc_init_lib_val = _cc_init_libs[0][1] if _cc_init_libs else None

                with gr.Accordion("🎭 Character creation", open=False):
                    cc_state = gr.State({"chr1": "", "chr2": "", "chr3": "", "chr4": ""})
                    cc_entries_state = gr.State(_cc_init_entries)

                    with gr.Row():
                        cc_slot = gr.Dropdown(
                            label="Character slot",
                            choices=[("chr1", "chr1"), ("chr2", "chr2"), ("chr3", "chr3"), ("chr4", "chr4")],
                            value="chr1",
                            scale=2,
                        )
                        cc_gender = gr.Dropdown(
                            label="Gender",
                            choices=[("male", "male"), ("female", "female")],
                            value="male",
                            scale=2,
                        )
                        cc_era = gr.Dropdown(
                            label="Era",
                            choices=[("any", "any"), ("modern", "modern"), ("fantasy", "fantasy"), ("futuristic", "futuristic")],
                            value="any",
                            scale=2,
                        )
                        cc_show_restricted = gr.Checkbox(
                            label="Show restricted libraries",
                            value=False,
                            scale=3,
                        )

                    with gr.Row():
                        cc_pick_kind = gr.Dropdown(
                            label="Pick from",
                            choices=[("Keywords", "kw"), ("Packs", "pack")],
                            value="kw",
                            scale=2,
                        )
                        cc_pick_mode = gr.Dropdown(
                            label="Pick behavior",
                            choices=[("Add", "add"), ("Replace", "replace")],
                            value="add",
                            scale=2,
                        )
                        cc_insert_target = gr.Dropdown(
                            label="Insert into Positive",
                            choices=[
                                ("Inside ((chrX)) block", "chrblock"),
                                ("After first BREAK", "after_first_break"),
                                ("After last BREAK", "after_last_break"),
                                ("Append at end", "append_end"),
                            ],
                            value="chrblock",
                            scale=4,
                        )

                    cc_buffer = gr.Textbox(label="Character text (current slot)", lines=2, value="")

                    with gr.Row():
                        cc_group = gr.Dropdown(label="Section", choices=_cc_init_groups, value=_cc_init_group_val, scale=3)
                        cc_library = gr.Dropdown(label="Library", choices=_cc_init_libs, value=_cc_init_lib_val, scale=3)

                    cc_search = gr.Textbox(label="Search in selected library", placeholder="type to filter…", lines=1)
                    cc_item_dd = gr.Dropdown(label="Pick item", choices=_cc_item_choices("kw", "", _cc_init_lib_val) if _cc_init_lib_val else [], value=None)

                    with gr.Row():
                        cc_add_btn = gr.Button("➕ Add to character", scale=3)
                        cc_clear_btn = gr.Button("🧼 Clear slot", scale=2)
                        cc_append_btn = gr.Button("➡️ Append to Positive", scale=3)

                    cc_status = gr.Markdown("")

                def _cc_refresh_all(gender: str, era: str, show_restricted: bool, pick_kind: str):
                    entries = _cc_entries(pick_kind, gender, era, show_restricted)
                    gch = _cc_group_choices(entries)
                    gval = gch[0][1] if gch else None
                    lch = _cc_lib_choices(entries, gval) if gval else []
                    lval = lch[0][1] if lch else None
                    items = _cc_item_choices(pick_kind, "", lval) if lval else []
                    return (
                        gr.update(choices=gch, value=gval),
                        gr.update(choices=lch, value=lval),
                        gr.update(choices=items, value=None),
                        entries,
                    )

                def _cc_refresh_libs(group: str, entries: List[dict], pick_kind: str):
                    lch = _cc_lib_choices(entries, group)
                    lval = lch[0][1] if lch else None
                    items = _cc_item_choices(pick_kind, "", lval) if lval else []
                    return gr.update(choices=lch, value=lval), gr.update(choices=items, value=None)

                def _cc_refresh_items(pick_kind: str, q: str, category: str):
                    ch = _cc_item_choices(pick_kind, q or "", category or "")
                    return gr.update(choices=ch, value=None)

                def _cc_on_slot_change(slot: str, state: dict):
                    slot = (slot or "chr1").strip().lower()
                    return state.get(slot, "") if isinstance(state, dict) else ""

                def _cc_on_clear(slot: str, state: dict):
                    slot = (slot or "chr1").strip().lower()
                    if not isinstance(state, dict):
                        state = {"chr1": "", "chr2": "", "chr3": "", "chr4": ""}
                    state[slot] = ""
                    return "", state, "🧼 Cleared."

                def _cc_on_add(pick_kind: str, pick_mode: str, item_id: str, slot: str, state: dict, current: str):
                    if not item_id:
                        return current, state, "⚠️ Pick something first."
                    slot = (slot or "chr1").strip().lower()
                    if not isinstance(state, dict):
                        state = {"chr1": "", "chr2": "", "chr3": "", "chr4": ""}
                    current = (current or "").strip().strip(",")

                    kind = (pick_kind or "kw").strip().lower()
                    mode = (pick_mode or "add").strip().lower()

                    text = ""
                    if kind in ("pack", "packs"):
                        tags = _vs.resolve_pack_tags(item_id)
                        names = [t.get("name","").strip() for t in tags if (t.get("name") or "").strip()]
                        names = [n for n in names if not _cc_is_explicit(n)]
                        text = ", ".join(names)
                    else:
                        k = _vs.get_tag(item_id) or {}
                        nm = (k.get("name") or "").strip()
                        if _cc_is_explicit(nm):
                            return current, state, "⚠️ That item is not available here."
                        text = nm

                    if not text:
                        return current, state, "⚠️ Nothing to add."

                    if mode == "replace":
                        newbuf = text
                    else:
                        newbuf = _cc_append_text(current, text)

                    state[slot] = newbuf
                    return newbuf, state, "✅ Added."

                def _cc_on_append(pos_prompt: str, slot: str, insert_target: str, state: dict, buf: str):
                    newp, msg = _cc_insert_into_positive(pos_prompt, slot, insert_target, buf)
                    return newp, state, msg

                # Wire refresh chain
                for _w in (cc_gender, cc_era, cc_show_restricted, cc_pick_kind):
                    _w.change(
                        fn=_cc_refresh_all,
                        inputs=[cc_gender, cc_era, cc_show_restricted, cc_pick_kind],
                        outputs=[cc_group, cc_library, cc_item_dd, cc_entries_state],
                    )

                cc_group.change(fn=_cc_refresh_libs, inputs=[cc_group, cc_entries_state, cc_pick_kind], outputs=[cc_library, cc_item_dd])
                for _w in (cc_search, cc_library, cc_pick_kind):
                    _w.change(fn=_cc_refresh_items, inputs=[cc_pick_kind, cc_search, cc_library], outputs=[cc_item_dd])

                cc_slot.change(fn=_cc_on_slot_change, inputs=[cc_slot, cc_state], outputs=[cc_buffer])
                cc_clear_btn.click(fn=_cc_on_clear, inputs=[cc_slot, cc_state], outputs=[cc_buffer, cc_state, cc_status])
                cc_add_btn.click(fn=_cc_on_add, inputs=[cc_pick_kind, cc_pick_mode, cc_item_dd, cc_slot, cc_state, cc_buffer], outputs=[cc_buffer, cc_state, cc_status])
                cc_append_btn.click(fn=_cc_on_append, inputs=[positive_out, cc_slot, cc_insert_target, cc_state, cc_buffer], outputs=[positive_out, cc_state, cc_status])

                # ✨ Recents + Favorites + Pack insert mode + Cleaner
                u0 = UsageStore()
                fav0 = u0.choices_favorites()
                rec0 = u0.choices_recents()

                with gr.Row():
                    pack_insert_mode = gr.Dropdown(
                        label="Pack insert",
                        choices=[
                            ("Insert all (max 40)", "all"),
                            ("Random 5", "random5"),
                            ("Random 10", "random10"),
                        ],
                        value="all",
                        interactive=True,
                        scale=2,
                    )
                    fav_btn = gr.Button("⭐ Favorite selected", scale=1)

                with gr.Row():
                    clean_pos_btn = gr.Button("🧹 Clean Positive", scale=1)
                    clean_neg_btn = gr.Button("🧹 Clean Negative", scale=1)

                with gr.Row():
                    favorites_dd = gr.Dropdown(
                        label="⭐ Favorites",
                        choices=fav0,
                        value=None,
                        interactive=True,
                        visible=bool(fav0),
                    )
                    recents_dd = gr.Dropdown(
                        label="🕘 Recents",
                        choices=rec0,
                        value=None,
                        interactive=True,
                        visible=bool(rec0),
                    )

                
                # -----------------------------
                # Base prompt templates (append)
                # -----------------------------
                with gr.Accordion("🧱 Base prompts (append)", open=False):
                    s_bp = VaultStore()
                    _bp_base_cats = ["all"] + s_bp.list_categories(kinds=["bases"])
                    _bp_slot_cats = ["all"] + s_bp.list_categories(kinds=["keywords", "packs"])

                    with gr.Row():
                        bp_filter = gr.Dropdown(label="Category (bases)", choices=_bp_base_cats, value="all")
                        bp_search = gr.Textbox(label="Search base prompts", lines=1, placeholder="ambulance / studio / rain…")
                        bp_dd = gr.Dropdown(label="Base prompt", choices=s_bp.list_base_choices(category="all"), value=None)

                    bp_template_preview = gr.Textbox(label="Template preview", lines=4, interactive=False)
                    bp_slots = gr.Slider(minimum=1, maximum=12, step=1, value=2, label="Slots")
                    bp_include_desc = gr.Checkbox(label="Include keyword descriptions", value=False)
                    bp_pick_mode = gr.Radio(label="On pick", choices=["add", "replace"], value="add")

                    with gr.Row():
                        slot_cat = gr.Dropdown(label="Slot pick category (keywords/packs)", choices=_bp_slot_cats, value="all")
                        slot_search = gr.Textbox(label="Slot search", lines=1, placeholder="piggyback / streetwear / soft light…")
                        slot_refresh = gr.Button("↻ Refresh slot list", scale=0)

                    slot_rows = []
                    slot_dds = []
                    slot_txts = []
                    for i in range(12):
                        with gr.Group(visible=(i < 2)) as rr:
                            with gr.Row():
                                dd = gr.Dropdown(label=f"Slot {i+1} (keyword/pack)", choices=[], value=None)
                                tx = gr.Textbox(label=f"Slot {i+1} content", lines=1, placeholder="type text or accumulate picks")
                        slot_rows.append(rr)
                        slot_dds.append(dd)
                        slot_txts.append(tx)


                    bp_preview = gr.Textbox(label="Built preview", lines=4, interactive=False)
                    with gr.Row():
                        bp_build_btn = gr.Button("🧱 Build preview", scale=0)
                        bp_append_btn = gr.Button("➕ Append to Positive", scale=0)

                    def _bp_refresh_bases(q: str, cat: str):
                        s = VaultStore()
                        return gr.update(choices=s.list_base_choices(q=q, category=cat), value=None)

                    def _bp_slot_choices(q: str, cat: str):
                        s = VaultStore()
                        tags = s.list_tag_choices(q=q, category=cat)
                        packs = s.list_pack_choices(q=q, category=cat)
                        out = []
                        for lab, kid in tags:
                            out.append((f"🏷️ {lab}", f"TAG:{kid}"))
                        for lab, pid in packs:
                            out.append((f"📦 {lab}", f"PACK:{pid}"))
                        return out

                    def _bp_refresh_slots(q: str, cat: str):
                        ch = _bp_slot_choices(q, cat)
                        # Reset values to avoid stale selection after filtering
                        return [gr.update(choices=ch, value=None) for _ in range(12)]

                    def _bp_set_slot_vis(n: int):
                        n = int(n or 0)
                        n = max(1, min(12, n))
                        return [gr.update(visible=(i < n)) for i in range(12)]

                    def _bp_load_base(sel: str):
                        if not sel:
                            return "", 2, *(_bp_set_slot_vis(2))
                        s = VaultStore()
                        b = s.get_base(sel) or {}
                        tmpl = (b.get("template") or "").strip()
                        slots = int(b.get("slots") or 0)
                        # auto infer if slots not set
                        if slots <= 0:
                            if "[tags]" in tmpl:
                                slots = 2
                            else:
                                ph = re.findall(r"\[tag\d+\]", tmpl)
                                slots = max(2, min(12, len(ph))) if ph else 2
                        slots = max(1, min(12, slots))
                        return tmpl, slots, *(_bp_set_slot_vis(slots))

                    def _bp_resolve_value(val: str, include_desc: bool, pack_mode: str) -> str:
                        if not val:
                            return ""
                        s = VaultStore()
                        if val.startswith("TAG:") or val.startswith("TAGD:"):
                            tid = val.split(":", 1)[1]
                            t = s.get_tag(tid) or {}
                            name = (t.get("name") or "").strip()
                            desc = (t.get("desc") or t.get("description") or "").strip()
                            if include_desc and desc:
                                return f"{name}, {desc}" if name else desc
                            return name
                        if val.startswith("PACK:"):
                            pid = val.split(":", 1)[1]
                            kws = s.resolve_pack_tags(pid)
                            names = [k.get("name") for k in kws if k.get("name")]
                            names = names[:40]
                            mode = (pack_mode or "all").strip().lower()
                            if mode == "random5" and names:
                                return ", ".join(random.sample(names, k=min(5, len(names))))
                            if mode == "random10" and names:
                                return ", ".join(random.sample(names, k=min(10, len(names))))
                            return ", ".join(names)
                        return ""


                    def _bp_apply_pick(pick_val: str, current: str, mode: str, include_desc: bool, pack_mode: str):
                        # Apply a dropdown pick into the slot content textbox.
                        if not pick_val:
                            return current, gr.update(value=None)
                        rep = _bp_resolve_value(pick_val, include_desc, pack_mode)
                        if not rep:
                            return current, gr.update(value=None)
                        cur = (current or "").strip()
                        mode = (mode or "add").strip().lower()
                        if mode == "replace" or not cur:
                            out = rep
                        else:
                            out = cur + ", " + rep
                        out = _bp_normalize_prompt(out)
                        return out, gr.update(value=None)
                    def _bp_normalize_prompt(s: str) -> str:
                        s = (s or "").replace("\n", ", ")
                        parts = [p.strip() for p in s.split(",")]
                        parts = [p for p in parts if p]
                        return ", ".join(parts)

                    # Apply picks into slot content (so changing category/search won't wipe previous selections)
                    for _dd, _tx in zip(slot_dds, slot_txts):
                        _dd.change(fn=_bp_apply_pick, inputs=[_dd, _tx, bp_pick_mode, bp_include_desc, pack_insert_mode], outputs=[_tx, _dd])


                    def _bp_build(template: str, slot_vals: List[str], slot_txts_in: List[str], include_desc: bool, pack_mode: str) -> str:
                        tmpl = (template or "").strip()
                        if not tmpl:
                            return ""
                        slots = []
                        for v, t in zip(slot_vals, slot_txts_in):
                            rep = _bp_resolve_value(v or "", include_desc, pack_mode)
                            extra = (t or "").strip()
                            if rep and extra:
                                slots.append(f"{rep}, {extra}")
                            elif rep:
                                slots.append(rep)
                            elif extra:
                                slots.append(extra)
                        slots = [x for x in slots if x]

                        out = tmpl
                        if "[tags]" in out:
                            out = out.replace("[tags]", ", ".join(slots))
                            slots = []
                        else:
                            ph = re.findall(r"\[tag\d+\]", out)
                            for i, tagph in enumerate(ph):
                                rep = slots[i] if i < len(slots) else ""
                                out = out.replace(tagph, rep, 1)
                            # extras beyond placeholders: append
                            extra = slots[len(ph):]
                            if extra:
                                out = out + ", " + ", ".join(extra)

                        # remove any leftovers
                        out = re.sub(r"\[tag\d+\]|\[tags\]", "", out)
                        return _bp_normalize_prompt(out)

                    def _bp_build_preview(template: str, include_desc: bool, pack_mode: str, *args):
                        # args = 12 dropdown values + 12 texts
                        vals = list(args[:12])
                        txts = list(args[12:24])
                        built = _bp_build(template, vals, txts, include_desc, pack_mode)
                        return built

                    def _bp_append_to_positive(prompt: str, built: str):
                        built = (built or "").strip()
                        if not built:
                            return prompt
                        prompt = (prompt or "").strip()
                        if not prompt:
                            return built
                        # comma-aware append
                        if prompt.endswith(","):
                            return prompt + " " + built
                        return prompt + ", " + built

                    # events
                    bp_filter.change(fn=_bp_refresh_bases, inputs=[bp_search, bp_filter], outputs=[bp_dd])
                    bp_search.change(fn=_bp_refresh_bases, inputs=[bp_search, bp_filter], outputs=[bp_dd])

                    slot_refresh.click(fn=_bp_refresh_slots, inputs=[slot_search, slot_cat], outputs=slot_dds)
                    slot_cat.change(fn=_bp_refresh_slots, inputs=[slot_search, slot_cat], outputs=slot_dds)
                    slot_search.change(fn=_bp_refresh_slots, inputs=[slot_search, slot_cat], outputs=slot_dds)

                    bp_dd.change(fn=_bp_load_base, inputs=[bp_dd], outputs=[bp_template_preview, bp_slots] + slot_rows)

                    bp_slots.change(fn=_bp_set_slot_vis, inputs=[bp_slots], outputs=slot_rows)

                    # build/append
                    bp_build_btn.click(
                        fn=_bp_build_preview,
                        inputs=[bp_template_preview, bp_include_desc, pack_insert_mode] + slot_dds + slot_txts,
                        outputs=[bp_preview],
                    )
                    bp_append_btn.click(
                        fn=lambda p, built: _bp_append_to_positive(p, built),
                        inputs=[positive_out, bp_preview],
                        outputs=[positive_out],
                    )



            fav_msg = gr.Markdown(visible=False)

            # Note: description insertion is selectable via the dropdown ("✨+desc" items)

            # -----------------------------
            # LoRA / TI quick insert (registry lives in Vault DB)
            # -----------------------------
            with gr.Accordion("🎛️ LoRA + TI Quick Insert", open=False):
                gr.Markdown("Pick a registered **LoRA/TI**, then insert the token + your saved triggers. (Manage registry in **Vault + Maps → LoRA / TI**) ")

                def _append_csv(base: str, addition: str) -> str:
                    base = base or ""
                    add = (addition or "").strip()
                    if not add:
                        return base
                    if not base.strip():
                        return add
                    if base.rstrip().endswith((",", ";", "\n")):
                        return base.rstrip() + " " + add
                    return base.rstrip() + ", " + add

                l_kind = gr.Radio(label="Type", choices=["lora", "ti"], value="lora")
                with gr.Row():
                    l_search = gr.Textbox(label="Search", placeholder="skin, anime, portrait, lighting…", lines=1)
                    l_dd = gr.Dropdown(label="Select", choices=_vs.list_lora_choices(kind="lora"), value=None)

                l_file = gr.Textbox(label="File", value="", interactive=False)
                l_rel = gr.Textbox(label="Prompt token", value="", interactive=False)
                l_strength = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Strength (LoRA)")
                l_triggers = gr.CheckboxGroup(label="Triggers (click)", choices=[], value=[])
                l_keywords_preview = gr.Markdown("")

                with gr.Row():
                    btn_insert_token = gr.Button("➕ Insert token")
                    btn_insert_triggers = gr.Button("🏷️ Insert selected triggers")
                    btn_insert_all_triggers = gr.Button("✨ Insert ALL triggers", variant="secondary")

                l_msg = gr.Markdown("")

                def _refresh_lora_dd(q: str, kind: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_lora_choices(q=q, kind=kind), value=None)

                def _kind_changed(kind: str, q: str):
                    s = VaultStore()
                    return (
                        gr.update(choices=s.list_lora_choices(q=q, kind=kind), value=None),
                        gr.update(visible=(kind == "lora")),
                        "",
                        "",
                        gr.update(choices=[], value=[]),
                        "",
                        "",
                    )

                def _load_lora(lid: str, kind: str):
                    s = VaultStore()
                    it = s.get_lora(lid) if lid else None
                    if not it:
                        return "", "", 0.8, gr.update(choices=[], value=[]), "", "⚠️ Select an item."
                    trig = it.get("triggers") or []
                    kw = it.get("keywords") or []
                    kline = "**Keywords:** " + ", ".join(kw) if kw else ""
                    # strength: only relevant for lora
                    strength = float(it.get("default_strength") or 1.0)
                    return it.get("file",""), it.get("rel",""), strength, gr.update(choices=trig, value=[]), kline, "✅ Loaded."

                def _insert_token(prompt: str, kind: str, lid: str, strength: float) -> Tuple[str, str]:
                    s = VaultStore()
                    it = s.get_lora(lid) if lid else None
                    if not it:
                        return prompt, "⚠️ Select a LoRA/TI first."
                    rel = (it.get("rel") or it.get("name") or "").strip()
                    if not rel:
                        return prompt, "⚠️ Missing token name."
                    if kind == "ti":
                        token = rel
                    else:
                        token = f"<lora:{rel}:{float(strength):.2f}>"
                    return _append_csv(prompt, token), "✅ Inserted token."

                def _insert_trigs_selected(prompt: str, lid: str, selected: List[str]) -> Tuple[str, str]:
                    s = VaultStore()
                    it = s.get_lora(lid) if lid else None
                    if not it:
                        return prompt, "⚠️ Select a LoRA/TI first."
                    trigs = it.get("triggers") or []
                    use = selected or []
                    if not use:
                        return prompt, "⚠️ No triggers selected."
                    add = ", ".join([t.strip() for t in use if t.strip()])
                    return _append_csv(prompt, add), "✅ Inserted triggers."

                def _insert_trigs_all(prompt: str, lid: str) -> Tuple[str, str]:
                    s = VaultStore()
                    it = s.get_lora(lid) if lid else None
                    if not it:
                        return prompt, "⚠️ Select a LoRA/TI first."
                    trigs = it.get("triggers") or []
                    if not trigs:
                        return prompt, "⚠️ No triggers saved for this item."
                    add = ", ".join([t.strip() for t in trigs if t.strip()])
                    return _append_csv(prompt, add), "✅ Inserted ALL triggers."

                l_search.change(_refresh_lora_dd, inputs=[l_search, l_kind], outputs=[l_dd])
                l_kind.change(_kind_changed, inputs=[l_kind, l_search], outputs=[l_dd, l_strength, l_file, l_rel, l_triggers, l_keywords_preview, l_msg])
                l_dd.change(_load_lora, inputs=[l_dd, l_kind], outputs=[l_file, l_rel, l_strength, l_triggers, l_keywords_preview, l_msg])
                btn_insert_token.click(_insert_token, inputs=[positive_out, l_kind, l_dd, l_strength], outputs=[positive_out, l_msg])
                btn_insert_triggers.click(_insert_trigs_selected, inputs=[positive_out, l_dd, l_triggers], outputs=[positive_out, l_msg])
                btn_insert_all_triggers.click(_insert_trigs_all, inputs=[positive_out, l_dd], outputs=[positive_out, l_msg])

            negative_out = gr.Textbox(label="Negative prompt", lines=4, elem_id="ps_neg_prompt")

            # -----------------------------
            # 💾 Quick save preset (near prompt boxes)
            # -----------------------------
            with gr.Row():
                quick_preset_title = gr.Textbox(
                    label='💾 Preset title',
                    placeholder='e.g., Ambulance tension — sweaty paramedics',
                    lines=1,
                    scale=6,
                )
                quick_preset_save = gr.Button('💾 Save preset', scale=2)
            quick_preset_status = gr.Markdown('')

            with gr.Row():
                send_txt2img = gr.Button("➡️ Send to txt2img")
                send_img2img = gr.Button("➡️ Send to img2img")


            # -----------------------------
            # Send to txt2img / img2img (JS)
            # -----------------------------
            send_txt2img.click(
                fn=lambda pos, neg: (pos, neg),
                inputs=[positive_out, negative_out],
                outputs=[positive_out, negative_out],
                js=r"""
(pos, neg) => {
  const app = (() => { try { return (typeof gradioApp === 'function') ? gradioApp() : document; } catch(e) { return document; } })();
  const q = (sel, root=app) => (root && root.querySelector) ? root.querySelector(sel) : null;
  const qa = (sel, root=app) => (root && root.querySelectorAll) ? Array.from(root.querySelectorAll(sel)) : [];

  const fire = (el) => {
    if (!el) return;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  };

  const setTA = (ta, v) => {
    if (!ta) return false;
    ta.value = v || '';
    fire(ta);
    return true;
  };

  const findTabPanel = (name) => {
    // direct ids
    let panel = q(`#tab_${name}`) || q(`#${name}`) || q(`[id*="tab_${name}"]`);
    if (panel) return panel;

    // role=tab buttons with aria-controls
    const btn = qa('[role="tab"]').find(b => {
      const txt = (b.textContent||'').trim().toLowerCase();
      const id = (b.id||'').toLowerCase();
      return txt === name || id.includes(name);
    });
    if (btn) {
      const ctl = btn.getAttribute('aria-controls');
      if (ctl) panel = q(`#${ctl}`);
      // try click to ensure tab rendered
      try { btn.click(); } catch(e) {}
      if (panel) return panel;
    }
    return null;
  };

  const findTextareaByLabel = (panel, labelExact) => {
    if (!panel) return null;
    const target = labelExact.trim().toLowerCase();
    const labels = qa('label', panel);
    const lab = labels.find(l => (l.textContent||'').trim().toLowerCase() === target)
             || labels.find(l => (l.textContent||'').trim().toLowerCase().startsWith(target));
    if (!lab) return null;

    // walk up until we find a textarea
    let el = lab;
    for (let i=0; i<8; i++) {
      el = el.parentElement;
      if (!el) break;
      const ta = q('textarea', el);
      if (ta) return ta;
    }
    // last resort: nearest textarea in panel
    return q('textarea', panel);
  };

  const panel = findTabPanel('txt2img');
  // common A1111 ids
  const ok1 = setTA(q('#txt2img_prompt textarea') || q('textarea#txt2img_prompt') || q('#txt2img_prompt'), pos);
  const ok2 = setTA(q('#txt2img_neg_prompt textarea') || q('textarea#txt2img_neg_prompt') || q('#txt2img_neg_prompt'), neg);

  // label fallback (works even if elem_id changed)
  if (!ok1) setTA(findTextareaByLabel(panel, 'Prompt'), pos);
  if (!ok2) setTA(findTextareaByLabel(panel, 'Negative prompt'), neg);
  // last resort: pick any textarea in the txt2img panel
  try {
    if (!ok1 && panel) {
      const tas = qa('textarea', panel);
      const pick = tas.find(t => ((t.id||'') + (t.name||'')).toLowerCase().includes('prompt')) || tas[0];
      if (pick) setTA(pick, pos);
    }
    if (!ok2 && panel) {
      const tas = qa('textarea', panel);
      const pick = tas.find(t => ((t.id||'') + (t.name||'')).toLowerCase().includes('neg')) || tas[1] || tas[0];
      if (pick) setTA(pick, neg);
    }
  } catch(e) {}


  // no return needed; python fn preserves outputs
}
"""
            )

            send_img2img.click(
                fn=lambda pos, neg: (pos, neg),
                inputs=[positive_out, negative_out],
                outputs=[positive_out, negative_out],
                js=r"""
(pos, neg) => {
  const app = (() => { try { return (typeof gradioApp === 'function') ? gradioApp() : document; } catch(e) { return document; } })();
  const q = (sel, root=app) => (root && root.querySelector) ? root.querySelector(sel) : null;
  const qa = (sel, root=app) => (root && root.querySelectorAll) ? Array.from(root.querySelectorAll(sel)) : [];

  const fire = (el) => {
    if (!el) return;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  };

  const setTA = (ta, v) => {
    if (!ta) return false;
    ta.value = v || '';
    fire(ta);
    return true;
  };

  const findTabPanel = (name) => {
    // direct ids
    let panel = q(`#tab_${name}`) || q(`#${name}`) || q(`[id*="tab_${name}"]`);
    if (panel) return panel;

    // role=tab buttons with aria-controls
    const btn = qa('[role="tab"]').find(b => {
      const txt = (b.textContent||'').trim().toLowerCase();
      const id = (b.id||'').toLowerCase();
      return txt === name || id.includes(name);
    });
    if (btn) {
      const ctl = btn.getAttribute('aria-controls');
      if (ctl) panel = q(`#${ctl}`);
      // try click to ensure tab rendered
      try { btn.click(); } catch(e) {}
      if (panel) return panel;
    }
    return null;
  };

  const findTextareaByLabel = (panel, labelExact) => {
    if (!panel) return null;
    const target = labelExact.trim().toLowerCase();
    const labels = qa('label', panel);
    const lab = labels.find(l => (l.textContent||'').trim().toLowerCase() === target)
             || labels.find(l => (l.textContent||'').trim().toLowerCase().startsWith(target));
    if (!lab) return null;

    // walk up until we find a textarea
    let el = lab;
    for (let i=0; i<8; i++) {
      el = el.parentElement;
      if (!el) break;
      const ta = q('textarea', el);
      if (ta) return ta;
    }
    // last resort: nearest textarea in panel
    return q('textarea', panel);
  };

  const panel = findTabPanel('img2img');
  // common A1111 ids
  const ok1 = setTA(q('#img2img_prompt textarea') || q('textarea#img2img_prompt') || q('#img2img_prompt'), pos);
  const ok2 = setTA(q('#img2img_neg_prompt textarea') || q('textarea#img2img_neg_prompt') || q('#img2img_neg_prompt'), neg);

  // label fallback (works even if elem_id changed)
  if (!ok1) setTA(findTextareaByLabel(panel, 'Prompt'), pos);
  if (!ok2) setTA(findTextareaByLabel(panel, 'Negative prompt'), neg);
  // last resort: pick any textarea in the img2img panel
  try {
    if (!ok1 && panel) {
      const tas = qa('textarea', panel);
      const pick = tas.find(t => ((t.id||'') + (t.name||'')).toLowerCase().includes('prompt')) || tas[0];
      if (pick) setTA(pick, pos);
    }
    if (!ok2 && panel) {
      const tas = qa('textarea', panel);
      const pick = tas.find(t => ((t.id||'') + (t.name||'')).toLowerCase().includes('neg')) || tas[1] || tas[0];
      if (pick) setTA(pick, neg);
    }
  } catch(e) {}


  // no return needed; python fn preserves outputs
}
"""
            )


            # -----------------------------
            # Local GGUF LLM Prompt Writer (optional)
            # -----------------------------
            _llms = _load_llm_settings()
            with gr.Accordion("🧠 Local GGUF LLM Prompt Writer (optional)", open=False):
                gr.Markdown(
                    "Generate or rewrite prompts using a **local GGUF text LLM** (runs via your separate venv python).\\n"
                    "This stays **SFW / non-explicit** by default (safer + avoids conflicts)."
                )

                llm_enable = gr.Checkbox(label="Enable LLM", value=False)
                llm_python = gr.Textbox(
                    label="LLM Python executable (venv python.exe)",
                    value=_llms.get("python_exe", ""),
                    placeholder=r"F:\\MyTools\\LLm Helper\\.venv\\Scripts\\python.exe",
                )
                llm_model = gr.Textbox(
                    label="GGUF model path (.gguf)",
                    value=_llms.get("model_path", ""),
                    placeholder=r"F:\\LLM\\Models\\GGUF\\Cydonia-24B-v4j-Q2_K.gguf",
                )

                with gr.Row():
                    llm_mode = gr.Dropdown(
                        label="Mode",
                        choices=["Generate from idea", "Rewrite current prompt"],
                        value="Rewrite current prompt",
                    )
                    llm_style = gr.Dropdown(
                        label="Output style",
                        choices=["Comma tags (SDXL)", "Paragraph (Photo SDXL)", "Bullet blocks"],
                        value="Comma tags (SDXL)",
                    )

                llm_idea = gr.Textbox(
                    label="Idea / direction (for Generate mode)",
                    lines=3,
                    placeholder="Example: two male models, rainy neon street, fashion editorial, romantic tension (non-graphic)",
                )

                llm_sfw = gr.Checkbox(label="Keep SFW / non-explicit", value=True)

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        llm_temp = gr.Slider(0.0, 1.5, value=float(_llms.get("temperature", 0.7)), step=0.05, label="Temperature")
                        llm_top_p = gr.Slider(0.1, 1.0, value=float(_llms.get("top_p", 0.95)), step=0.05, label="Top-p")
                        llm_max_tokens = gr.Slider(64, 2048, value=int(_llms.get("max_tokens", 256)), step=32, label="Max tokens")

                    with gr.Row():
                        llm_ctx = gr.Slider(1024, 8192, value=int(_llms.get("n_ctx", 4096)), step=256, label="Context (n_ctx)")
                        llm_gpu_layers = gr.Slider(0, 200, value=int(_llms.get("gpu_layers", 0)), step=1, label="GPU layers (0=CPU)")
                        llm_threads = gr.Slider(1, 32, value=int(_llms.get("threads", 8)), step=1, label="Threads")
                        llm_rp = gr.Slider(1.0, 2.0, value=float(_llms.get("repeat_penalty", 1.15)), step=0.05, label="Repeat penalty")

                with gr.Row():
                    llm_btn = gr.Button("🤖 Generate / Rewrite")
                    llm_save = gr.Button("💾 Save LLM settings", variant="secondary")
                llm_msg = gr.Markdown()

                def _llm_save_settings_cb(pyexe, model, n_ctx, max_tokens, temperature, top_p, rp, gpu_layers, threads):
                    d = _load_llm_settings()
                    d["python_exe"] = (pyexe or "").strip().strip('"')
                    d["model_path"] = (model or "").strip().strip('"')
                    d["n_ctx"] = int(n_ctx)
                    d["max_tokens"] = int(max_tokens)
                    d["temperature"] = float(temperature)
                    d["top_p"] = float(top_p)
                    d["repeat_penalty"] = float(rp)
                    d["gpu_layers"] = int(gpu_layers)
                    d["threads"] = int(threads)
                    _save_llm_settings(d)
                    return "✅ Saved LLM settings."

                def _llm_run_cb(
                    enable,
                    pyexe,
                    model_path,
                    mode,
                    out_style,
                    idea,
                    sfw,
                    temperature,
                    top_p,
                    max_tokens,
                    n_ctx,
                    n_gpu_layers,
                    n_threads,
                    repeat_penalty,
                    cur_pos,
                    cur_neg,
                ):
                    if not enable:
                        return cur_pos, cur_neg, "ℹ️ LLM is disabled."

                    pyexe = (pyexe or "").strip().strip('"')
                    model_path = (model_path or "").strip().strip('"')

                    if not pyexe:
                        return cur_pos, cur_neg, "⚠️ Set **LLM Python executable** (venv python.exe)."
                    if not os.path.exists(pyexe):
                        return cur_pos, cur_neg, f"❌ Python not found: `{pyexe}`"
                    if not model_path:
                        return cur_pos, cur_neg, "⚠️ Set **GGUF model path** (.gguf)."
                    if not os.path.exists(model_path):
                        return cur_pos, cur_neg, f"❌ GGUF not found: `{model_path}`"

                    if (mode or "").lower().startswith("generate") and not (idea or "").strip():
                        return cur_pos, cur_neg, "⚠️ Add an **idea/direction** for Generate mode."

                    payload = {
                        "mode": mode,
                        "idea": idea or "",
                        "current_positive": cur_pos or "",
                        "current_negative": cur_neg or "",
                        "tag_style": out_style or "Comma tags (SDXL)",
                        "family_safe": bool(sfw),
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "max_tokens": int(max_tokens),
                        "n_ctx": int(n_ctx),
                        "n_gpu_layers": int(n_gpu_layers),
                        "n_threads": int(n_threads),
                    }

                    # write payload to temp
                    tmp_fd, tmp_path = tempfile.mkstemp(prefix="ps_llm_payload_", suffix=".json")
                    os.close(tmp_fd)
                    try:
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)

                        cmd = [pyexe, str(LLM_CLI_PATH), "--model", model_path, "--payload", tmp_path]

                        p = subprocess.run(cmd, capture_output=True, text=True)
                        if p.returncode != 0:
                            err = (p.stderr or p.stdout or "").strip()
                            if not err:
                                err = f"LLM exited with code {p.returncode}"
                            return cur_pos, cur_neg, "❌ LLM error:\n\n```\n" + err[:2000] + "\n```"

                        raw = (p.stdout or "").strip()
                        data = {}
                        try:
                            data = json.loads(raw)
                        except Exception:
                            data = {"positive": raw}

                        new_pos = (data.get("positive") or "").strip() or cur_pos
                        new_neg = (data.get("negative") or "").strip() or cur_neg
                        return new_pos, new_neg, "✅ LLM updated prompts."
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                llm_save.click(
                    fn=_llm_save_settings_cb,
                    inputs=[llm_python, llm_model, llm_ctx, llm_max_tokens, llm_temp, llm_top_p, llm_rp, llm_gpu_layers, llm_threads],
                    outputs=[llm_msg],
                    queue=False,
                )

                llm_btn.click(
                    fn=_llm_run_cb,
                    inputs=[
                        llm_enable, llm_python, llm_model, llm_mode, llm_style, llm_idea, llm_sfw,
                        llm_temp, llm_top_p, llm_max_tokens, llm_ctx, llm_gpu_layers, llm_threads, llm_rp,
                        positive_out, negative_out,
                    ],
                    outputs=[positive_out, negative_out, llm_msg],
                    queue=False,
                )

            # -----------------------------
        # Vault Bridge (keywords/packs/mapsets + prompt presets)
        # -----------------------------
        with gr.Accordion("🗃️ Vault Bridge + Save/Load (keywords/packs/mapsets + assets + presets)", open=True):
            gr.Markdown("Load **keywords / packs / mapsets** you saved in **Vault + Maps**, then insert or queue them here.")
            gr.Markdown("⬇️ **Save your final prompt** + attached **composition/reference images** in the sections below. No restart needed.")

            def _append(base: str, addition: str) -> str:
                base = base or ""
                add = (addition or "").strip()
                if not add:
                    return base
                if not base.strip():
                    return add
                # keep formatting (comma-style prompts)
                if base.rstrip().endswith((",", ";", "\n")):
                    return base.rstrip() + " " + add
                return base.rstrip() + ", " + add

            with gr.Row():
                vault_refresh = gr.Button("🔄 Refresh Vault lists", variant="secondary")
                vault_status = gr.Markdown("")

            with gr.Tabs():
                # -------- Packs --------
                with gr.Tab("📦 Packs"):
                    pack_dd2 = gr.Dropdown(label="Pack", choices=_vs.list_pack_choices(), value=None)
                    pack_tags2 = gr.Dropdown(label="Tags in pack", choices=_vs.list_tag_choices(), multiselect=True, value=[])
                    pack_include_desc = gr.Checkbox(value=False, label="Include descriptions too")
                    btn_insert_pack = gr.Button("➕ Insert selected keywords → Positive")

                # -------- Tags --------
                with gr.Tab("🔑 Keywords"):
                    tag_search2 = gr.Textbox(label="Search keywords", placeholder="piggyback / hoodie / smirk…", lines=1)
                    tag_dd2 = gr.Dropdown(label="Keyword", choices=_vs.list_tag_choices(), value=None)
                    tag_include_desc = gr.Checkbox(value=True, label="Include description after keyword (if saved)")
                    tag_desc_preview = gr.Markdown("")
                    btn_insert_tag = gr.Button("➕ Insert keyword → Positive")

                # -------- MapSets --------
                with gr.Tab("🗺️ MapSets"):
                    mapset_dd2 = gr.Dropdown(label="Mapset", choices=_vs.list_mapset_choices(), value=None)
                    mapset_tags_preview = gr.Markdown("")
                    with gr.Row():
                        show_canny2 = gr.Checkbox(value=True, label="Show Canny")
                        show_depth2 = gr.Checkbox(value=True, label="Show Depth")
                        show_pose2  = gr.Checkbox(value=True, label="Show OpenPose")
                    canny_g2 = gr.Gallery(label="Canny", columns=4, rows=2, height="auto")
                    depth_g2 = gr.Gallery(label="Depth", columns=4, rows=2, height="auto")
                    pose_g2  = gr.Gallery(label="OpenPose", columns=4, rows=2, height="auto")

                    sel_canny_path = gr.State(value="")
                    sel_depth_path = gr.State(value="")
                    sel_pose_path  = gr.State(value="")

                    gr.Markdown("**Selected maps** (click thumbnails)")
                    with gr.Row():
                        sel_canny_img = gr.Image(label="Selected Canny", type="filepath", interactive=False)
                        sel_depth_img = gr.Image(label="Selected Depth", type="filepath", interactive=False)
                        sel_pose_img  = gr.Image(label="Selected OpenPose", type="filepath")

                    with gr.Row():
                        w_canny = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Canny weight (CN0)")
                        w_depth = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Depth weight (CN1)")
                        w_pose  = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="OpenPose weight (CN2)")

                    with gr.Row():
                        btn_q_canny2 = gr.Button("Queue Canny → CN Unit 0")
                        btn_q_depth2 = gr.Button("Queue Depth → CN Unit 1")
                        btn_q_pose2  = gr.Button("Queue OpenPose → CN Unit 2")
                        btn_clear_q2 = gr.Button("Clear queued maps", variant="secondary")

                    cn_status2 = gr.Markdown("")

                    
                    # -----------------------------
                    # Assets: Composition + Reference
                    # -----------------------------
                    with gr.Accordion("🖼️ Attach Assets (Composition / Reference) — saved with presets", open=True):
                        comp_state = gr.State([])
                        ref_state = gr.State([])

                        def _paths_from_gr_files(x):
                            if not x:
                                return []
                            if isinstance(x, str):
                                return [x]
                            if isinstance(x, dict):
                                p = x.get("name") or x.get("path")
                                return [p] if p else []
                            if isinstance(x, list):
                                out = []
                                for it in x:
                                    if isinstance(it, str):
                                        out.append(it)
                                    elif isinstance(it, dict):
                                        p = it.get("name") or it.get("path")
                                        if p:
                                            out.append(p)
                                return [p for p in out if p]
                            return []

                        def _dedupe_keep_order(items):
                            seen = set()
                            out = []
                            for p in (items or []):
                                if not p:
                                    continue
                                if p in seen:
                                    continue
                                seen.add(p)
                                out.append(p)
                            return out

                        def _add_assets(existing, new_files, include_desc=False):
                            cur = list(existing or [])
                            newp = _paths_from_gr_files(new_files)
                            cur.extend(newp)
                            cur = _dedupe_keep_order(cur)
                            return cur, cur, "\n".join(cur)

                        def _clear_assets():
                            return [], [], ""

                        with gr.Tabs():
                            with gr.Tab("🧩 Composition (IP-Adapter style)"):
                                comp_upload = gr.File(label="Add composition images", file_count="multiple")
                                with gr.Row():
                                    btn_comp_add = gr.Button("➕ Add")
                                    btn_comp_clear = gr.Button("🧹 Clear", variant="secondary")
                                comp_gallery = gr.Gallery(label="Composition images (paths)", show_label=True)
                                comp_paths = gr.Textbox(label="Composition paths", lines=3, interactive=False)

                            with gr.Tab("🎯 Reference (reference-only)"):
                                ref_upload = gr.File(label="Add reference images", file_count="multiple")
                                with gr.Row():
                                    btn_ref_add = gr.Button("➕ Add")
                                    btn_ref_clear = gr.Button("🧹 Clear", variant="secondary")
                                ref_gallery = gr.Gallery(label="Reference images (paths)", show_label=True)
                                ref_paths = gr.Textbox(label="Reference paths", lines=3, interactive=False)

                        
                        with gr.Row():
                            btn_comp_export = gr.Button("📤 Export composition → folder", variant="secondary")
                            btn_ref_export  = gr.Button("📤 Export reference → folder", variant="secondary")
                        export_status = gr.Markdown("", visible=False)

                        def _export_to_folder(kind: str, paths):
                            items = list(paths or [])
                            if not items:
                                return gr.update(value="⚠️ Nothing to export.", visible=True)
                            base = EXT_ROOT / "user_data" / "exports" / kind
                            base.mkdir(parents=True, exist_ok=True)
                            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                            out_dir = base / stamp
                            out_dir.mkdir(parents=True, exist_ok=True)
                            ok = 0
                            for p in items:
                                try:
                                    src = Path(p)
                                    if src.exists():
                                        shutil.copy2(src, out_dir / src.name)
                                        ok += 1
                                except Exception:
                                    pass
                            return gr.update(value=f"✅ Exported {ok}/{len(items)} file(s) → {out_dir}", visible=True)

                        btn_comp_add.click(fn=_add_assets, inputs=[comp_state, comp_upload], outputs=[comp_state, comp_gallery, comp_paths], queue=False)
                        btn_comp_clear.click(fn=_clear_assets, inputs=[], outputs=[comp_state, comp_gallery, comp_paths], queue=False)

                        btn_ref_add.click(fn=_add_assets, inputs=[ref_state, ref_upload], outputs=[ref_state, ref_gallery, ref_paths], queue=False)
                        btn_ref_clear.click(fn=_clear_assets, inputs=[], outputs=[ref_state, ref_gallery, ref_paths], queue=False)
                        btn_comp_export.click(fn=lambda paths: _export_to_folder('composition', paths), inputs=[comp_state], outputs=[export_status], queue=False)
                        btn_ref_export.click(fn=lambda paths: _export_to_folder('reference', paths), inputs=[ref_state], outputs=[export_status], queue=False)


                    with gr.Accordion("💾 Save & Load Final Prompts (includes maps + assets)", open=True):
                        preset_search = gr.Textbox(label="Search saved prompts", placeholder="hug / piggyback / temple…", lines=1)
                        preset_dd = gr.Dropdown(label="Saved prompts", choices=_ps.list_choices(), value=None)
                        preset_title = gr.Textbox(label="Preset title", placeholder="Piggyback ride — romantic streetwear")
                        gr.Markdown("Save the **current Output prompts** + optional **mapset + selected maps + weights**.")
                        with gr.Row():
                            btn_preset_new = gr.Button("💾 Save as NEW")
                            btn_preset_update = gr.Button("📝 Update selected")
                            btn_preset_load = gr.Button("📥 Load into Output")
                            btn_preset_del = gr.Button("🗑️ Delete", variant="stop")
                        preset_status = gr.Markdown("")

            # -----------------------------
            # Vault UI helpers / events
            # -----------------------------
            def _vault_refresh_all():
                vs = VaultStore()
                ps = PromptPresetStore()
                return (
                    gr.update(choices=vs.list_pack_choices(), value=None),
                    gr.update(choices=vs.list_tag_choices(), value=None),
                    gr.update(choices=vs.list_mapset_choices(), value=None),
                    gr.update(choices=vs.list_tag_choices(), value=[]),
                    gr.update(choices=ps.list_choices(), value=None),
                    "✅ Refreshed Vault lists."
                )

            def _load_pack(pid: str):
                vs = VaultStore()
                if not pid:
                    return gr.update(value=[]), "⚠️ Select a pack."
                kws = vs.resolve_pack_tags(pid)
                ids = [k.get("id") for k in (kws or []) if k.get("id")]
                return gr.update(value=ids), "✅ Pack loaded."

            def _tag_preview(tid: str):
                vs = VaultStore()
                t = vs.get_tag(tid) if tid else None
                if not t:
                    return ""
                name = t.get("name","")
                desc = t.get("desc","")
                aliases = ", ".join(t.get("aliases") or [])
                s = f"**{name}**"
                if aliases:
                    s += f"  \nAliases: `{aliases}`"
                if desc:
                    s += f"  \n> {desc}"
                return s

            def _search_tags(q: str):
                vs = VaultStore()
                return gr.update(choices=vs.list_tag_choices(q))

            def _insert_tag_into_positive(cur: str, tid: str, include_desc: bool):
                vs = VaultStore()
                t = vs.get_tag(tid) if tid else None
                if not t:
                    return cur
                add = t.get("name","")
                if include_desc and t.get("desc"):
                    add = f'{add}, {t.get("desc")}'
                return _append(cur, add)

            def _insert_pack_tags(cur: str, tag_ids: list, include_desc: bool):
                vs = VaultStore()
                adds = []
                for tid in (tag_ids or []):
                    t = vs.get_tag(tid)
                    if not t or not t.get("enabled", True):
                        continue
                    add = t.get("name","")
                    if include_desc and t.get("desc"):
                        add = f'{add}, {t.get("desc")}'
                    adds.append(add)
                return _append(cur, ", ".join(adds))

            def _mapset_select(mid: str):
                vs = VaultStore()
                m = vs.get_mapset(mid) if mid else None
                if not m:
                    return "", [], [], []
                tags = ", ".join(m.get("tags") or [])
                tag_line = f"Tags: `{tags}`" if tags else ""
                return tag_line, vs.list_map_paths(mid, "canny"), vs.list_map_paths(mid, "depth"), vs.list_map_paths(mid, "openpose")

            def _pick_from_gallery(items, evt: gr.SelectData):
                if not items:
                    return "", ""
                idx = getattr(evt, "index", 0)
                if isinstance(idx, (tuple, list)):
                    idx = idx[0]
                try:
                    it = items[int(idx)]
                except Exception:
                    it = items[0]
                path = it[0] if isinstance(it, (tuple, list)) else str(it)
                return path, path

            def _queue(kind: str, unit: int, path: str, w: float):
                if queue_cb is None:
                    return "⚠️ Queue function not available."
                return queue_cb(kind, unit, path, w)

            def _clear_q():
                if clear_queue_cb is None:
                    return "⚠️ Clear queue not available."
                return clear_queue_cb()

            def _search_presets(q: str):
                ps = PromptPresetStore()
                return gr.update(choices=ps.list_choices(q))

            def _save_new_preset(title: str, pos: str, neg: str, mid: str, p_canny: str, p_depth: str, p_pose: str, wc: float, wd: float, wp: float, comp_imgs, ref_imgs):
                ps = PromptPresetStore()
                pid = ps.upsert(
                    "",
                    title or "Untitled",
                    pos or "",
                    neg or "",
                    {"mapset_id": mid or "", "maps": {"canny": p_canny or "", "depth": p_depth or "", "openpose": p_pose or ""}},
                    {"unit0": "canny", "unit1": "depth", "unit2": "openpose"},
                    {"canny": float(wc or 1.0), "depth": float(wd or 1.0), "openpose": float(wp or 1.0)},
                    assets={"maps": {"canny": p_canny or "", "depth": p_depth or "", "openpose": p_pose or ""}, "composition": comp_imgs or [], "reference": ref_imgs or []},
                )
                if not pid:
                    return gr.update(choices=ps.list_choices(), value=None), "⚠️ Title required."
                return gr.update(choices=ps.list_choices(), value=pid), "✅ Saved new preset."

            def _update_preset(pid: str, title: str, pos: str, neg: str, mid: str, p_canny: str, p_depth: str, p_pose: str, wc: float, wd: float, wp: float, comp_imgs, ref_imgs):
                if not pid:
                    return gr.update(choices=_ps.list_choices(), value=None), "⚠️ Select a preset first."
                ps = PromptPresetStore()
                out = ps.upsert(
                    pid,
                    title or "Untitled",
                    pos or "",
                    neg or "",
                    {"mapset_id": mid or "", "maps": {"canny": p_canny or "", "depth": p_depth or "", "openpose": p_pose or ""}},
                    {"unit0": "canny", "unit1": "depth", "unit2": "openpose"},
                    {"canny": float(wc or 1.0), "depth": float(wd or 1.0), "openpose": float(wp or 1.0)},
                    assets={"maps": {"canny": p_canny or "", "depth": p_depth or "", "openpose": p_pose or ""}, "composition": comp_imgs or [], "reference": ref_imgs or []},
                )
                return gr.update(choices=ps.list_choices(), value=out), "✅ Updated."

            def _delete_preset(pid: str):
                if not pid:
                    return gr.update(choices=_ps.list_choices(), value=None), "⚠️ Select a preset first."
                ps = PromptPresetStore()
                ps.delete(pid)
                return gr.update(choices=ps.list_choices(), value=None), "🗑️ Deleted."

            def _load_preset(pid: str):
                ps = PromptPresetStore()
                it = ps.get(pid) if pid else None
                if not it:
                    return (
                        "", "", gr.update(value=None),
                        "", [], [], [],
                        "", "", "",
                        None, None, None,
                        1.0, 1.0, 1.0,
                        [], [], "", [], [], "",
                        "", "⚠️ Select a preset."
                    )

                linked = it.get("linked") or {}
                assets = it.get("assets") or {}
                maps = (assets.get("maps") or {}) or (linked.get("maps") or {})
                strengths = it.get("strengths") or {}
                mid = linked.get("mapset_id") or ""

                tag_line, cg, dg, pg = _mapset_select(mid)

                pc = maps.get("canny", "") or ""
                pd = maps.get("depth", "") or ""
                pp = maps.get("openpose", "") or ""

                comp = assets.get("composition") or []
                refs = assets.get("reference") or []

                # images update with filepath strings
                return (
                    it.get("positive",""),
                    it.get("negative",""),
                    gr.update(value=mid),
                    tag_line,
                    cg, dg, pg,
                    pc, pd, pp,
                    pc or None, pd or None, pp or None,
                    float(strengths.get("canny",1.0)),
                    float(strengths.get("depth",1.0)),
                    float(strengths.get("openpose",1.0)),
                    comp, comp, "\n".join(comp),
                    refs, refs, "\n".join(refs),
                    it.get("title",""),
                    "✅ Loaded preset."
                )
            pack_dd2.change(fn=_load_pack, inputs=[pack_dd2], outputs=[pack_tags2, vault_status], queue=False)

            tag_search2.change(fn=_search_tags, inputs=[tag_search2], outputs=[tag_dd2], queue=False)
            tag_dd2.change(fn=_tag_preview, inputs=[tag_dd2], outputs=[tag_desc_preview], queue=False)
            btn_insert_tag.click(fn=_insert_tag_into_positive, inputs=[positive_out, tag_dd2, tag_include_desc], outputs=[positive_out], queue=False)
            btn_insert_pack.click(fn=_insert_pack_tags, inputs=[positive_out, pack_tags2, pack_include_desc], outputs=[positive_out], queue=False)

            mapset_dd2.change(fn=_mapset_select, inputs=[mapset_dd2], outputs=[mapset_tags_preview, canny_g2, depth_g2, pose_g2], queue=False)

            show_canny2.change(lambda v: gr.update(visible=v), inputs=[show_canny2], outputs=[canny_g2])
            show_depth2.change(lambda v: gr.update(visible=v), inputs=[show_depth2], outputs=[depth_g2])
            show_pose2.change(lambda v: gr.update(visible=v), inputs=[show_pose2], outputs=[pose_g2])

            canny_g2.select(fn=_pick_from_gallery, inputs=[canny_g2], outputs=[sel_canny_path, sel_canny_img], queue=False)
            depth_g2.select(fn=_pick_from_gallery, inputs=[depth_g2], outputs=[sel_depth_path, sel_depth_img], queue=False)
            pose_g2.select(fn=_pick_from_gallery, inputs=[pose_g2], outputs=[sel_pose_path, sel_pose_img], queue=False)

            btn_q_canny2.click(fn=lambda p,w: _queue("canny", 0, p, w), inputs=[sel_canny_path, w_canny], outputs=[cn_status2], queue=False)
            btn_q_depth2.click(fn=lambda p,w: _queue("depth", 1, p, w), inputs=[sel_depth_path, w_depth], outputs=[cn_status2], queue=False)
            btn_q_pose2.click(fn=lambda p,w: _queue("openpose", 2, p, w), inputs=[sel_pose_path, w_pose], outputs=[cn_status2], queue=False)
            btn_clear_q2.click(fn=_clear_q, inputs=[], outputs=[cn_status2], queue=False)

            preset_search.change(fn=_search_presets, inputs=[preset_search], outputs=[preset_dd], queue=False)
            preset_dd.change(fn=lambda pid: ((PromptPresetStore().get(pid) or {}) .get("title","")), inputs=[preset_dd], outputs=[preset_title], queue=False)

            btn_preset_new.click(
                fn=_save_new_preset,
                inputs=[preset_title, positive_out, negative_out, mapset_dd2, sel_canny_path, sel_depth_path, sel_pose_path, w_canny, w_depth, w_pose, comp_state, ref_state],
                outputs=[preset_dd, preset_status],
                queue=False,
            )
            btn_preset_update.click(
                fn=_update_preset,
                inputs=[preset_dd, preset_title, positive_out, negative_out, mapset_dd2, sel_canny_path, sel_depth_path, sel_pose_path, w_canny, w_depth, w_pose, comp_state, ref_state],
                outputs=[preset_dd, preset_status],
                queue=False,
            )
            btn_preset_del.click(fn=_delete_preset, inputs=[preset_dd], outputs=[preset_dd, preset_status], queue=False)

            btn_preset_load.click(
                fn=_load_preset,
                inputs=[preset_dd],
                outputs=[positive_out, negative_out, mapset_dd2, mapset_tags_preview, canny_g2, depth_g2, pose_g2, sel_canny_path, sel_depth_path, sel_pose_path, sel_canny_img, sel_depth_img, sel_pose_img, w_canny, w_depth, w_pose, comp_state, comp_gallery, comp_paths, ref_state, ref_gallery, ref_paths, preset_title, preset_status],
                queue=False,
            )
        
        # -----------------------------
        # 💾 Quick save wiring (near prompt boxes)
        # -----------------------------
        def _save_new_preset_quick(title: str, pos: str, neg: str, mid: str, p_canny: str, p_depth: str, p_pose: str, wc: float, wd: float, wp: float, comp_imgs, ref_imgs):
            dd_upd, msg = _save_new_preset(title, pos, neg, mid, p_canny, p_depth, p_pose, wc, wd, wp, comp_imgs, ref_imgs)
            return dd_upd, msg, title or ''

        quick_preset_save.click(
            fn=_save_new_preset_quick,
            inputs=[quick_preset_title, positive_out, negative_out, mapset_dd2, sel_canny_path, sel_depth_path, sel_pose_path, w_canny, w_depth, w_pose, comp_state, ref_state],
            outputs=[preset_dd, quick_preset_status, preset_title],
            queue=False,
        )
# -----------------------------
        # 🔎 Vault typeahead wiring (tags / packs / loras / ti) for the panel prompt box
        # -----------------------------
        def _usage_updates():
            u = UsageStore()
            fav = u.choices_favorites()
            rec = u.choices_recents()
            return (
                gr.update(choices=fav, value=None, visible=bool(fav)),
                gr.update(choices=rec, value=None, visible=bool(rec)),
            )

        def _vault_suggest_ui(prompt: str):
            token = _ps_active_token(prompt)
            sugg = _ps_build_suggestions(token, limit=18)
            if not sugg:
                return gr.update(choices=[], value=None, visible=False)
            return gr.update(choices=sugg, value=None, visible=True)

        def _vault_apply_choice(prompt: str, val: str, pack_mode: str):
            if not val:
                fav_upd, rec_upd = _usage_updates()
                return prompt, gr.update(value=None, visible=False), fav_upd, rec_upd

            s = VaultStore()
            rep = ""

            if val.startswith("TAGD:") or val.startswith("TAG:"):
                tid = val.split(":", 1)[1]
                t = s.get_tag(tid) or {}
                name = (t.get("name") or "").strip()
                desc = (t.get("desc") or t.get("description") or "").strip()
                rep = name
                if val.startswith("TAGD:") and desc:
                    rep = f"{name}, {desc}"

            elif val.startswith("PACK:"):
                pid = val.split(":", 1)[1]
                # Packs are library-driven; resolve to keyword entries
                kws = s.resolve_pack_tags(pid)
                names = [k.get("name") for k in kws if k.get("name")]
                # hard cap to avoid huge insert
                names = names[:40]

                mode = (pack_mode or "all").strip().lower()
                if mode == "random5" and names:
                    rep = ", ".join(random.sample(names, k=min(5, len(names))))
                elif mode == "random10" and names:
                    rep = ", ".join(random.sample(names, k=min(10, len(names))))
                else:
                    rep = ", ".join(names)

            
            elif val.startswith("LORAT:") or val.startswith("LORA:"):
                lid = val.split(":", 1)[1]
                it = s.get_lora(lid) or {}
                rel = (it.get("rel") or it.get("name") or "").strip()
                w = it.get("default_strength", 0.8)
                try:
                    w = float(w)
                except Exception:
                    w = 0.8
                token = f"<lora:{rel}:{w:.2f}>" if rel else ""
                if val.startswith("LORAT:"):
                    tr = [x.strip() for x in (it.get("triggers") or []) if x.strip()]
                    rep = ", ".join([token] + tr) if token else ", ".join(tr)
                else:
                    rep = token

            elif val.startswith("TI:"):
                lid = val.split(":", 1)[1]
                it = s.get_lora(lid) or {}
                rep = (it.get("rel") or it.get("name") or "").strip()

            if not rep:
                fav_upd, rec_upd = _usage_updates()
                return prompt, gr.update(value=None, visible=False), fav_upd, rec_upd

            new_prompt = _ps_replace_active_token(prompt, rep)
            # record recent
            try:
                u = UsageStore()
                u.add_recent(_ps_label_for_value(val), val)
            except Exception:
                pass
            fav_upd, rec_upd = _usage_updates()
            return new_prompt, gr.update(value=None, visible=False), fav_upd, rec_upd

        def _clean_prompt_text(text: str) -> str:
            def _key(x: str) -> str:
                """Canonical token for de-dup, tolerant to case/spacing/punctuation/camelCase."""
                x = unicodedata.normalize("NFKC", (x or "").strip())
                # casefold is stronger than lower() for unicode
                x = x.casefold()
                # treat space/_/- as same (remove them entirely)
                x = re.sub(r"[\s_\-–—]+", "", x)
                # drop punctuation but keep word chars across languages
                x = re.sub(r"[^\w]+", "", x, flags=re.UNICODE)
                return x

            lines = (text or "").splitlines() or [""]
            cleaned_lines = []
            for ln in lines:
                parts = [p.strip() for p in (ln or "").split(",")]
                parts = [p for p in parts if p]
                seen = set()
                out = []
                for p in parts:
                    k = _key(p)
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(p)
                cleaned_lines.append(", ".join(out))
            return "\n".join(cleaned_lines).strip()

        def _fav_toggle_selected(val: str):
            if not val:
                fav_upd, rec_upd = _usage_updates()
                return fav_upd, rec_upd, gr.update(value="", visible=False)
            u = UsageStore()
            label = _ps_label_for_value(val)
            now = u.toggle_favorite(label, val)
            fav_upd, rec_upd = _usage_updates()
            msg = "⭐ Added to Favorites" if now else "🗑️ Removed from Favorites"
            return fav_upd, rec_upd, gr.update(value=msg, visible=True)

        # update suggestions as you type (deepbooru-ish)
        positive_out.input(
            fn=_vault_suggest_ui,
            inputs=[positive_out],
            outputs=[vault_suggest_dd],
        )
        vault_suggest_dd.change(
            fn=_vault_apply_choice,
            inputs=[positive_out, vault_suggest_dd, pack_insert_mode],
            outputs=[positive_out, vault_suggest_dd, favorites_dd, recents_dd],
        )

        # favorites/recents quick insert
        favorites_dd.change(
            fn=_vault_apply_choice,
            inputs=[positive_out, favorites_dd, pack_insert_mode],
            outputs=[positive_out, vault_suggest_dd, favorites_dd, recents_dd],
        )
        recents_dd.change(
            fn=_vault_apply_choice,
            inputs=[positive_out, recents_dd, pack_insert_mode],
            outputs=[positive_out, vault_suggest_dd, favorites_dd, recents_dd],
        )

        # toggle favorite for currently highlighted suggestion
        fav_btn.click(
            fn=_fav_toggle_selected,
            inputs=[vault_suggest_dd],
            outputs=[favorites_dd, recents_dd, fav_msg],
        )

        # cleaners
        clean_pos_btn.click(fn=_clean_prompt_text, inputs=[positive_out], outputs=[positive_out])
        clean_neg_btn.click(fn=_clean_prompt_text, inputs=[negative_out], outputs=[negative_out])

    return positive_out, negative_out
