
import json
import os
import re
import shutil
import uuid
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------------------------------------------------------
# VaultStore (Prompt Suite)
# - Stores tags / packs / mapsets (assets) ONLY
# - Persists to user_data/ so updates won't wipe data
# -------------------------------------------------------

EXT_ROOT = Path(__file__).resolve().parents[1]
USER_DATA = EXT_ROOT / "user_data"
VAULT_DB_PATH = USER_DATA / "vault_db.json"
ASSETS_DIR = USER_DATA / "assets"  # assets/<mapset_id>/<type>/*.png

# LoRA/TI registry storage lives in the same vault DB under the key "loras".
# Each entry example:
# {
#   "id": "...", "kind": "lora"|"ti",
#   "file": "C:/.../models/Lora/foo.safetensors",
#   "rel": "subfolder/foo", "name": "foo", "category": "subfolder",
#   "triggers": ["trigger1", "trigger2"],
#   "keywords": ["style", "subject"],
#   "default_strength": 0.8,
#   "notes": "...",
#   "created": "...", "updated": "...",
# }

_MAP_TYPES = ("canny", "depth", "openpose")
_LORA_EXTS = (".safetensors", ".pt", ".ckpt")
_EMBED_EXTS = (".pt", ".safetensors")

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_dirs():
    USER_DATA.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not VAULT_DB_PATH.exists():
        VAULT_DB_PATH.write_text(json.dumps({"tags": [], "packs": [], "mapsets": [], "loras": []}, indent=2), encoding="utf-8")


# ---------------- Built-in library import ----------------
_LIB_DIR = EXT_ROOT / "libraries"

_NUM_ITEM = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")
_BULLET_ITEM = re.compile(r"^\s*-\s+(.*\S)\s*$")
_H1 = re.compile(r"^\s*#\s+(.*\S)\s*$")
_H2 = re.compile(r"^\s*##\s+(.*\S)\s*$")
_TAG = re.compile(r"\[([A-Za-z0-9_+-]+)\]")

def _infer_category_from_filename(name: str) -> str:
    n = (name or "").lower()
    if "outfit" in n or "streetwear" in n:
        return "clothing"
    if "hair" in n:
        return "clothing"
    if "location" in n:
        return "location"
    if "pose" in n:
        return "pose"
    if "moment" in n:
        return "moment"
    return "misc"

def _clean_item_text(s: str) -> Tuple[str, List[str]]:
    """Return (main_text, bracket_tags)."""
    raw = (s or "").strip()
    tags = _TAG.findall(raw) if raw else []
    # strip bracket tags from the line
    main = _TAG.sub("", raw).strip()
    main = re.sub(r"\s{2,}", " ", main).strip()
    # remove trailing markdown line breaks
    main = main.rstrip(" -–—")
    return main, tags

def _maybe_import_builtin_libraries(data: Dict[str, Any]) -> bool:
    """
    One-time import:
    - Each .md file becomes one or more packs (per ## section if present)
    - List items become tag entries
    Returns True if it modified data.
    """
    try:
        if data.get("_builtin_import_v1") is True:
            return False
        if not _LIB_DIR.exists():
            data["_builtin_import_v1"] = True
            return False
    except Exception:
        return False

    tags = data.get("tags") or []
    packs = data.get("packs") or []

    # Index existing tags by normalized key to avoid duplicates
    seen = set()
    for t in tags:
        cat = (t.get("category") or "misc").strip()
        name = (t.get("name") or "").strip()
        if name:
            seen.add((_norm_token(cat), _norm_token(name)))

    def upsert_tag(cat: str, name: str, aliases: List[str], desc: str = "") -> str:
        c = (cat or "misc").strip()
        n = (name or "").strip()
        if not n:
            return ""
        key = (_norm_token(c), _norm_token(n))
        if key in seen:
            # find existing id
            for t in tags:
                if _norm_token(t.get("category") or "misc") == key[0] and _norm_token(t.get("name") or "") == key[1]:
                    return t.get("id") or ""
            return ""
        tid = str(uuid.uuid4())
        seen.add(key)
        # add underscore alias to help booru-style matching
        alias_set = set([a.strip() for a in (aliases or []) if a.strip()])
        alias_set.add(re.sub(r"\s+", "_", n.lower()))
        tags.append({
            "id": tid,
            "category": c,
            "name": n,
            "aliases": sorted(alias_set),
            "desc": desc or "",
            "enabled": True,
            "created": _now_iso(),
            "updated": _now_iso(),
            "_source": "builtin_md",
        })
        return tid

    def upsert_pack(cat: str, title: str, tag_ids: List[str]):
        if not title or not tag_ids:
            return
        # Avoid exact duplicate pack titles (same cat/title)
        for p in packs:
            if _norm_token(p.get("category") or "misc") == _norm_token(cat) and _norm_token(p.get("title") or "") == _norm_token(title):
                return
        packs.append({
            "id": str(uuid.uuid4()),
            "category": (cat or "misc").strip(),
            "title": title.strip(),
            "tag_ids": [tid for tid in tag_ids if tid],
            "created": _now_iso(),
            "updated": _now_iso(),
            "_source": "builtin_md",
        })

    # Parse every .md
    for md in sorted(_LIB_DIR.glob("*.md")):
        base = md.stem.strip()
        cat = _infer_category_from_filename(md.name)
        cur_section = ""
        section_items: Dict[str, List[str]] = {}
        try:
            text = md.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for line in text:
            if _H2.match(line):
                cur_section = _H2.match(line).group(1).strip()
                continue
            if _H1.match(line):
                # ignore H1 as section; we use filename
                continue
            m = _NUM_ITEM.match(line) or _BULLET_ITEM.match(line)
            if not m:
                continue
            item_raw = m.group(2) if isinstance(m, re.Match) and m.re == _NUM_ITEM else m.group(1)
            item_raw = (item_raw or "").strip()
            if not item_raw:
                continue
            main, bt = _clean_item_text(item_raw)
            if not main:
                continue
            sec_key = cur_section or base
            section_items.setdefault(sec_key, []).append((main, bt))

        # Create tags + packs
        for sec, items in section_items.items():
            tag_ids = []
            # Pack title: "File — Section" if section differs
            pack_title = base if sec == base else f"{base} — {sec}"
            for main, bt in items:
                tid = upsert_tag(cat, main, bt, "")
                if tid:
                    tag_ids.append(tid)
            if tag_ids:
                upsert_pack(cat, pack_title, tag_ids)

    data["tags"] = tags
    data["packs"] = packs
    data["_builtin_import_v1"] = True
    return True


def _load_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback

def _save_json(path: Path, data: Dict[str, Any]) -> None:
    """Atomic-ish save to avoid corrupting vault_db on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(str(tmp), str(path))
    except Exception:
        # Fallback (non-atomic) if replace fails on some FS.
        path.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")

def _safe_name(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9._ -]+", "_", s)
    return (s[:max_len] or "Untitled")

def _norm_token(s: str) -> str:
    """Canonical token for matching/lookup (case + spacing/punctuation tolerant)."""
    s = unicodedata.normalize("NFKC", (s or "").strip())
    s = s.casefold()
    # treat space/_/- as similar
    s = re.sub(r"[\s_\-–—]+", "", s)
    # drop punctuation but keep word chars across languages
    s = re.sub(r"[^\w]+", "", s, flags=re.UNICODE)
    return s

def _guess_map_type(filename: str) -> Optional[str]:
    n = filename.lower()
    if "canny" in n:
        return "canny"
    if "depth" in n:
        return "depth"
    if "openpose" in n or re.search(r"(^|[_-])pose([_-]|\.)", n):
        return "openpose"
    return None


def _try_get_default_dirs() -> Tuple[Optional[str], Optional[str]]:
    """Best-effort (Forge/A1111) default directories for LoRA and embeddings."""
    lora_dir = None
    emb_dir = None
    try:
        # A1111 style
        from modules import paths  # type: ignore
        models_path = getattr(paths, "models_path", None)
        if models_path and os.path.isdir(models_path):
            cand = os.path.join(models_path, "Lora")
            if os.path.isdir(cand):
                lora_dir = cand
        # Embeddings live at repo root typically
        sd_path = getattr(paths, "script_path", None)
        if sd_path and os.path.isdir(sd_path):
            cand = os.path.join(sd_path, "embeddings")
            if os.path.isdir(cand):
                emb_dir = cand
    except Exception:
        pass

    try:
        # cmd opts override
        from modules import shared  # type: ignore
        co = getattr(shared, "cmd_opts", None)
        if co is not None:
            ld = getattr(co, "lora_dir", None)
            if ld and os.path.isdir(ld):
                lora_dir = ld
            ed = getattr(co, "embeddings_dir", None)
            if ed and os.path.isdir(ed):
                emb_dir = ed
    except Exception:
        pass

    return lora_dir, emb_dir

def _ensure_suffix(stem: str, map_type: str) -> str:
    # enforce suffix for auto-detect in other tools
    st = stem
    st_low = st.lower()
    suf = f"_{map_type}"
    if st_low.endswith(suf):
        return st
    # avoid double suffixes like _canny_depth
    for t in _MAP_TYPES:
        if st_low.endswith(f"_{t}"):
            return st
    return f"{st}{suf}"

class VaultStore:
    def __init__(self):
        _ensure_dirs()
        # Vault DB is still used for MapSets + LoRA/TI meta.
        # Keywords/Packs are library (.md) driven in CLEAN mode.
        self.data = _load_json(VAULT_DB_PATH, {"mapsets": [], "loras": [], "tags": [], "packs": []})
        self.data.setdefault("mapsets", [])
        self.data.setdefault("loras", [])
        # keep keys for back-compat (not used for keywords/packs in CLEAN mode)
        self.data.setdefault("tags", [])
        self.data.setdefault("packs", [])
        self._lib_dir = _LIB_DIR

    def save(self):
        _save_json(VAULT_DB_PATH, self.data)


    # ------------------ Library (Keywords / Packs) ------------------
    def _slug_category(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            return ""
        name = unicodedata.normalize("NFKC", name).casefold()
        name = re.sub(r"[\s\-–—]+", "_", name)
        name = re.sub(r"[^a-z0-9_]+", "", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def _parse_lib_filename(self, filename: str) -> Optional[Tuple[str, str]]:
        # Expected: <category>__keywords*.md OR <category>__packs*.md
        stem = Path(filename).stem
        parts = stem.split("__")
        if len(parts) < 2:
            return None
        cat = (parts[0] or "").strip()
        kind = (parts[1] or "").strip().casefold()
        if kind not in ("keywords", "packs", "bases"):
            return None
        if not cat:
            return None
        return cat, kind

    def _iter_lib_files(self, kind: str) -> List[Path]:
        kind = (kind or "").strip().casefold()
        if not self._lib_dir.exists():
            return []
        out = []
        for p in sorted(self._lib_dir.glob("*.md")):
            info = self._parse_lib_filename(p.name)
            if not info:
                continue
            cat, k = info
            if k == kind:
                out.append(p)
        return out

    def _kw_id(self, category: str, canonical: str) -> str:
        return f"kw::{(category or 'misc')}::{_norm_token(canonical)}"

    # ------------------ Categories ------------------
    def list_categories(self, kinds: Optional[List[str]] = None) -> List[str]:
        """List categories present in library .md files.

        Args:
            kinds: Optional list of kinds to include. Supported: keywords, packs, bases.
                   If None/empty, includes all.
        """
        if kinds is None:
            kinds_list: List[str] = ["keywords", "packs", "bases"]
        elif isinstance(kinds, str):  # type: ignore
            kinds_list = [kinds]  # type: ignore
        else:
            kinds_list = list(kinds)

        # normalize + filter
        k_norm: List[str] = []
        for k in kinds_list:
            kk = (k or "").strip().casefold()
            if kk in ("keywords", "packs", "bases") and kk not in k_norm:
                k_norm.append(kk)
        if not k_norm:
            k_norm = ["keywords", "packs", "bases"]

        cats = set()
        for kk in k_norm:
            for p in self._iter_lib_files(kk):
                info = self._parse_lib_filename(p.name)
                if info:
                    cats.add(info[0])
        return sorted(cats)

    def add_category(self, name: str) -> bool:
        cat = self._slug_category(name)
        if not cat:
            return False
        self._lib_dir.mkdir(parents=True, exist_ok=True)
        kw_path = self._lib_dir / f"{cat}__keywords.md"
        pk_path = self._lib_dir / f"{cat}__packs.md"
        bs_path = self._lib_dir / f"{cat}__bases.md"
        created = False
        if not kw_path.exists():
            kw_path.write_text("", encoding="utf-8")
            created = True
        if not pk_path.exists():
            pk_path.write_text("", encoding="utf-8")
            created = True
        if not bs_path.exists():
            bs_path.write_text("", encoding="utf-8")
            created = True
        return created

    # ------------------ Keywords (compat: Tags) ------------------
    def _parse_keyword_line(self, line: str) -> Optional[Dict[str, Any]]:
        raw = (line or "").strip()
        if not raw:
            return None
        if raw.startswith("#") or raw.startswith("//"):
            return None
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if not parts:
            return None
        canonical = parts[0]
        meta = {"aliases": [], "desc": "", "enabled": True}
        for seg in parts[1:]:
            if ":" not in seg:
                continue
            k, v = seg.split(":", 1)
            k = (k or "").strip().casefold()
            v = (v or "").strip()
            if k in ("alias", "aliases"):
                meta["aliases"] = [a.strip() for a in v.split(",") if a.strip()]
            elif k in ("desc", "description"):
                meta["desc"] = v
            elif k in ("enabled", "enable"):
                vv = v.casefold()
                meta["enabled"] = vv not in ("0", "false", "no", "off")
        return {"name": canonical, **meta}

    def _load_keywords(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for p in self._iter_lib_files("keywords"):
            info = self._parse_lib_filename(p.name)
            if not info:
                continue
            cat, _ = info
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                rec = self._parse_keyword_line(line)
                if not rec:
                    continue
                kid = self._kw_id(cat, rec["name"])
                if kid in out:
                    continue
                out[kid] = {
                    "id": kid,
                    "category": cat,
                    "name": rec["name"],
                    "aliases": rec.get("aliases") or [],
                    "desc": rec.get("desc") or "",
                    "enabled": bool(rec.get("enabled", True)),
                    "source_file": str(p),
                }
        return out

    def list_tag_choices(self, q: str = "", category: str = "all") -> List[Tuple[str, str]]:
        # Back-compat name: "tags" = keywords
        qn = _norm_token(q)
        cn = _norm_token(category)
        kws = self._load_keywords()
        out: List[Tuple[str, str]] = []
        for kid, k in kws.items():
            cat = k.get("category") or "misc"
            name = k.get("name") or ""
            if not name:
                continue
            if cn and cn not in ("all", "*"):
                if _norm_token(cat) != cn:
                    continue
            label = f"{cat} › {name}"
            if qn:
                hay = _norm_token(label + " " + " ".join(k.get("aliases") or []) + " " + (k.get("desc") or ""))
                if qn not in hay:
                    continue
            out.append((label, kid))
        return out

    def get_tag(self, tid: str) -> Optional[Dict[str, Any]]:
        kws = self._load_keywords()
        return kws.get(tid)

    def _keywords_primary_file(self, category: str) -> Path:
        category = (category or "misc").strip()
        self._lib_dir.mkdir(parents=True, exist_ok=True)
        return self._lib_dir / f"{category}__keywords.md"

    def _write_keywords_for_category(self, category: str, keywords: List[Dict[str, Any]]) -> None:
        p = self._keywords_primary_file(category)
        lines = []
        for k in sorted(keywords, key=lambda x: (x.get("name") or "").casefold()):
            name = (k.get("name") or "").strip()
            if not name:
                continue
            segs = [name]
            aliases = [a.strip() for a in (k.get("aliases") or []) if a.strip()]
            if aliases:
                segs.append("alias:" + ",".join(aliases))
            desc = (k.get("desc") or "").strip()
            if desc:
                segs.append("desc:" + desc)
            enabled = bool(k.get("enabled", True))
            if not enabled:
                segs.append("enabled:false")
            lines.append(" | ".join(segs))
        txt = "\n".join(lines) + ("\n" if lines else "")
        bak = p.with_suffix(p.suffix + ".bak")
        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            if p.exists():
                shutil.copy2(p, bak)
        except Exception:
            pass
        tmp.write_text(txt, encoding="utf-8")
        os.replace(str(tmp), str(p))

    def _propagate_keyword_rename_in_packs(self, old_cat: str, old_name: str, new_cat: str, new_name: str):
        old_norm = _norm_token(old_name)
        for pf in self._iter_lib_files("packs"):
            try:
                txt = pf.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            blocks = self._parse_pack_blocks(txt, source_file=str(pf))
            changed = False
            for b in blocks:
                kws = b.get("keywords_raw") or []
                new_list = []
                for token in kws:
                    cat, nm = self._split_kw_token(token)
                    if _norm_token(nm) == old_norm and (not cat or _norm_token(cat) == _norm_token(old_cat)):
                        new_list.append(f"{new_cat}::{new_name}")
                        changed = True
                    else:
                        new_list.append(token)
                b["keywords_raw"] = new_list
            if changed:
                self._write_pack_blocks_to_file(pf, blocks)

    def upsert_tag(self, tid: str, category: str, name: str, aliases_csv: str, desc: str, enabled: bool) -> Optional[str]:
        name = (name or "").strip()
        if not name:
            return None
        category0 = (category or "").strip()
        category0 = self._slug_category(category0) or category0 or "misc"
        self.add_category(category0)

        aliases = [a.strip() for a in (aliases_csv or "").split(",") if a.strip()]

        old = self.get_tag(tid) if tid else None
        old_cat = old.get("category") if old else None
        old_name = old.get("name") if old else None

        def load_primary(cat: str) -> List[Dict[str, Any]]:
            p = self._keywords_primary_file(cat)
            if not p.exists():
                return []
            items = []
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                rec = self._parse_keyword_line(line)
                if not rec:
                    continue
                items.append({"name": rec["name"], "aliases": rec.get("aliases") or [], "desc": rec.get("desc") or "", "enabled": bool(rec.get("enabled", True))})
            return items

        cats_to_load = {category0}
        if old_cat:
            cats_to_load.add(old_cat)

        cat_items = {c: load_primary(c) for c in cats_to_load}

        if old_cat and old_name:
            old_norm = _norm_token(old_name)
            cat_items[old_cat] = [k for k in cat_items.get(old_cat, []) if _norm_token(k.get("name","")) != old_norm]

        new_norm = _norm_token(name)
        kept = [k for k in cat_items.get(category0, []) if _norm_token(k.get("name","")) != new_norm]
        kept.append({"name": name, "aliases": aliases, "desc": desc or "", "enabled": bool(enabled)})
        cat_items[category0] = kept

        for c, items in cat_items.items():
            self._write_keywords_for_category(c, items)

        if old_cat and old_name:
            if _norm_token(old_name) != _norm_token(name) or _norm_token(old_cat) != _norm_token(category0):
                self._propagate_keyword_rename_in_packs(old_cat, old_name, category0, name)

        return self._kw_id(category0, name)

    def delete_tag(self, tid: str):
        k = self.get_tag(tid)
        if not k:
            return
        cat = k.get("category") or "misc"
        name = k.get("name") or ""
        p = self._keywords_primary_file(cat)
        if p.exists():
            items = []
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                rec = self._parse_keyword_line(line)
                if not rec:
                    continue
                if _norm_token(rec["name"]) == _norm_token(name):
                    continue
                items.append({"name": rec["name"], "aliases": rec.get("aliases") or [], "desc": rec.get("desc") or "", "enabled": bool(rec.get("enabled", True))})
            self._write_keywords_for_category(cat, items)

        old_norm = _norm_token(name)
        for pf in self._iter_lib_files("packs"):
            try:
                txt = pf.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            blocks = self._parse_pack_blocks(txt, source_file=str(pf))
            changed = False
            for b in blocks:
                kws = b.get("keywords_raw") or []
                new_kws = []
                for token in kws:
                    c2, n2 = self._split_kw_token(token)
                    if _norm_token(n2) == old_norm and (not c2 or _norm_token(c2) == _norm_token(cat)):
                        changed = True
                        continue
                    new_kws.append(token)
                b["keywords_raw"] = new_kws
            if changed:
                self._write_pack_blocks_to_file(pf, blocks)

    # ------------------ Packs (library) ------------------
    def _split_kw_token(self, token: str) -> Tuple[str, str]:
        token = (token or "").strip()
        if "::" in token:
            c, n = token.split("::", 1)
            return (c.strip(), n.strip())
        return ("", token.strip())

    def _parse_pack_blocks(self, txt: str, source_file: str = "") -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        cur: List[str] = []
        for line in (txt or "").splitlines():
            if line.strip() == "---":
                if cur:
                    blocks.append(self._parse_pack_block(cur, source_file))
                cur = []
            else:
                cur.append(line)
        if cur:
            blocks.append(self._parse_pack_block(cur, source_file))
        return [b for b in blocks if b.get("title")]

    def _parse_pack_block(self, lines: List[str], source_file: str = "") -> Dict[str, Any]:
        pid = ""
        title = ""
        note = ""
        keywords_raw: List[str] = []
        for ln in lines:
            s = (ln or "").strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if s.startswith("@") and ":" in s:
                k, v = s[1:].split(":", 1)
                k = (k or "").strip().casefold()
                v = (v or "").strip()
                if k == "id":
                    pid = v
                elif k == "title":
                    title = v
                elif k in ("keywords", "tags"):
                    keywords_raw = [x.strip() for x in v.split(",") if x.strip()]
                elif k == "note":
                    note = v
        if not pid and title:
            pid = f"pk::{_safe_name(title)}::{abs(hash(source_file + '|' + title))}"
        return {
            "id": pid,
            "category": "",
            "title": title,
            "note": note,
            "keywords_raw": keywords_raw,
            "source_file": source_file,
        }

    def _packs_primary_file(self, category: str) -> Path:
        category = (category or "misc").strip()
        self._lib_dir.mkdir(parents=True, exist_ok=True)
        return self._lib_dir / f"{category}__packs.md"

    def _write_pack_blocks_to_file(self, path: Path, blocks: List[Dict[str, Any]]) -> None:
        out_lines: List[str] = []
        for b in blocks:
            title = (b.get("title") or "").strip()
            if not title:
                continue
            pid = (b.get("id") or "").strip() or str(uuid.uuid4())
            out_lines.append(f"@id: {pid}")
            out_lines.append(f"@title: {title}")
            kws = [x.strip() for x in (b.get("keywords_raw") or []) if x.strip()]
            if kws:
                out_lines.append("@keywords: " + ", ".join(kws))
            note = (b.get("note") or "").strip()
            if note:
                out_lines.append(f"@note: {note}")
            out_lines.append("")
            out_lines.append("---")
            out_lines.append("")
        txt = "\n".join(out_lines).strip() + ("\n" if out_lines else "")
        bak = path.with_suffix(path.suffix + ".bak")
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            if path.exists():
                shutil.copy2(path, bak)
        except Exception:
            pass
        tmp.write_text(txt, encoding="utf-8")
        os.replace(str(tmp), str(path))

    def _load_packs(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for p in self._iter_lib_files("packs"):
            info = self._parse_lib_filename(p.name)
            if not info:
                continue
            cat, _ = info
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            blocks = self._parse_pack_blocks(txt, source_file=str(p))
            for b in blocks:
                pid = b.get("id") or ""
                title = b.get("title") or ""
                if not pid or not title:
                    continue
                b["category"] = cat
                out[pid] = b
        return out


    # ------------------ Base Prompts (Templates) ------------------
    _BASE_SPLIT = re.compile(r"^\s*---\s*$", flags=re.M)

    def _slug_id(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            return ""
        name = unicodedata.normalize("NFKC", name).casefold()
        name = re.sub(r"[\s\-–—]+", "_", name)
        name = re.sub(r"[^a-z0-9_]+", "", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def _base_key(self, category: str, base_id: str) -> str:
        return f"base::{(category or 'misc')}::{(base_id or '').strip()}"

    def _parse_base_block(self, text: str) -> Optional[Dict[str, Any]]:
        lines = (text or "").splitlines()
        if not any((ln.strip() for ln in lines)):
            return None

        meta: Dict[str, Any] = {"id": "", "title": "", "slots": 0, "template": ""}
        in_template = False
        template_lines: List[str] = []

        for ln in lines:
            raw = ln.rstrip("\n")
            s = raw.strip()
            if not s and not in_template:
                continue

            if not in_template and s.startswith("@") and ":" in s:
                k, v = s[1:].split(":", 1)
                k = (k or "").strip().casefold()
                v = (v or "").strip()
                if k == "id":
                    meta["id"] = v
                    continue
                if k == "title":
                    meta["title"] = v
                    continue
                if k == "slots":
                    try:
                        meta["slots"] = int(v)
                    except Exception:
                        meta["slots"] = 0
                    continue
                if k == "template":
                    in_template = True
                    if v:
                        template_lines.append(v)
                    continue

            # also allow "template:" without "@"
            if not in_template and s.casefold().startswith("template:"):
                in_template = True
                v = s.split(":", 1)[1].strip()
                if v:
                    template_lines.append(v)
                continue

            if in_template:
                template_lines.append(raw)

        base_id = (meta.get("id") or "").strip()
        title = (meta.get("title") or "").strip()
        template = "\n".join(template_lines).strip()

        if not template:
            # fallback: treat entire block as template if no marker
            template = "\n".join([ln for ln in lines if ln.strip()]).strip()

        if not base_id:
            base_id = self._slug_id(title) if title else self._slug_id(template[:40])
        if not title:
            title = base_id or "Base Prompt"

        meta["id"] = base_id
        meta["title"] = title
        meta["template"] = template
        return meta

    def _load_bases(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for p in self._iter_lib_files("bases"):
            info = self._parse_lib_filename(p.name)
            if not info:
                continue
            cat, _ = info
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            blocks = re.split(r"^\s*---\s*$", txt, flags=re.M)
            for b in blocks:
                rec = self._parse_base_block(b)
                if not rec:
                    continue
                bid = (rec.get("id") or "").strip()
                if not bid:
                    continue
                key = self._base_key(cat, bid)
                if key in out:
                    continue
                out[key] = {
                    "id": key,
                    "base_id": bid,
                    "category": cat,
                    "title": rec.get("title") or bid,
                    "slots": int(rec.get("slots") or 0),
                    "template": rec.get("template") or "",
                    "source_file": str(p),
                }
        return out

    def list_base_choices(self, q: str = "", category: str = "all") -> List[Tuple[str, str]]:
        qn = _norm_token(q)
        cn = _norm_token(category)
        bases = self._load_bases()
        out: List[Tuple[str, str]] = []
        for key, b in bases.items():
            cat = (b.get("category") or "misc").strip()
            title = (b.get("title") or "").strip()
            if not key or not title:
                continue
            if cn and cn not in ("all", "*"):
                if _norm_token(cat) != cn:
                    continue
            label = f"{cat} › {title}"
            if qn:
                hay = _norm_token(label + " " + (b.get("template") or ""))
                if qn not in hay:
                    continue
            out.append((label, key))
        return out

    def get_base(self, base_key: str) -> Optional[Dict[str, Any]]:
        return self._load_bases().get(base_key)

    def _bases_primary_file(self, category: str) -> Path:
        category = (category or "misc").strip()
        self._lib_dir.mkdir(parents=True, exist_ok=True)
        return self._lib_dir / f"{category}__bases.md"

    def _read_bases_from_file(self, path: Path, category: str) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        blocks = re.split(r"^\s*---\s*$", txt, flags=re.M)
        out = []
        for b in blocks:
            rec = self._parse_base_block(b)
            if not rec:
                continue
            bid = (rec.get("id") or "").strip()
            if not bid:
                continue
            out.append({
                "base_id": bid,
                "category": category,
                "title": (rec.get("title") or bid).strip(),
                "slots": int(rec.get("slots") or 0),
                "template": (rec.get("template") or "").strip(),
            })
        return out

    def _write_bases_for_category(self, category: str, bases: List[Dict[str, Any]]) -> None:
        p = self._bases_primary_file(category)
        blocks = []
        for b in sorted(bases, key=lambda x: (x.get("title") or "").casefold()):
            bid = (b.get("base_id") or b.get("id") or "").strip()
            if bid.startswith("base::"):
                bid = bid.split("::")[-1]
            bid = self._slug_id(bid) or self._slug_id(b.get("title") or "") or str(uuid.uuid4())[:8]
            title = (b.get("title") or bid).strip()
            slots = int(b.get("slots") or 0)
            tmpl = (b.get("template") or "").strip()
            if not tmpl:
                continue
            block = [
                f"@id: {bid}",
                f"@title: {title}",
                f"@slots: {slots}",
                "@template:",
                tmpl,
                "---",
            ]
            blocks.append("\n".join(block))
        p.write_text("\n\n".join(blocks).strip() + ("\n" if blocks else ""), encoding="utf-8")

    def upsert_base(self, base_key: str, category: str, title: str, slots: int, template: str) -> str:
        cat = (category or "misc").strip()
        title = (title or "").strip()
        tmpl = (template or "").strip()
        if not cat or not title or not tmpl:
            return ""

        # Determine base_id + key
        base_id = ""
        if base_key and base_key.startswith("base::"):
            parts = base_key.split("::", 2)
            if len(parts) == 3:
                base_id = parts[2].strip()
                # if category changed, keep base_id but move file
        if not base_id:
            base_id = self._slug_id(title) or str(uuid.uuid4())[:8]
        key = self._base_key(cat, base_id)

        path = self._bases_primary_file(cat)
        existing = self._read_bases_from_file(path, cat)

        # replace by base_id
        updated = False
        for b in existing:
            if (b.get("base_id") or "").strip() == base_id:
                b["title"] = title
                b["slots"] = int(slots or 0)
                b["template"] = tmpl
                updated = True
                break
        if not updated:
            existing.append({"base_id": base_id, "category": cat, "title": title, "slots": int(slots or 0), "template": tmpl})

        self._write_bases_for_category(cat, existing)
        return key

    def delete_base(self, base_key: str) -> bool:
        if not base_key or not base_key.startswith("base::"):
            return False
        parts = base_key.split("::", 2)
        if len(parts) != 3:
            return False
        cat = (parts[1] or "misc").strip()
        base_id = (parts[2] or "").strip()
        path = self._bases_primary_file(cat)
        existing = self._read_bases_from_file(path, cat)
        new_list = [b for b in existing if (b.get("base_id") or "").strip() != base_id]
        if len(new_list) == len(existing):
            return False
        self._write_bases_for_category(cat, new_list)
        return True
    def list_pack_choices(self, q: str = "", category: str = "all") -> List[Tuple[str, str]]:
        qn = _norm_token(q)
        cn = _norm_token(category)
        packs = self._load_packs()
        out: List[Tuple[str, str]] = []
        for pid, p in packs.items():
            cat = p.get("category") or "misc"
            title = p.get("title") or ""
            if not title:
                continue
            if cn and cn not in ("all", "*"):
                if _norm_token(cat) != cn:
                    continue
            label = f"{cat} › {title}"
            if qn and qn not in _norm_token(label + " " + (p.get("note") or "")):
                continue
            out.append((label, pid))
        return out

    def get_pack(self, pid: str) -> Optional[Dict[str, Any]]:
        packs = self._load_packs()
        return packs.get(pid)

    def upsert_pack(self, pid: str, category: str, title: str, tag_ids: List[str]) -> Optional[str]:
        title = (title or "").strip()
        if not title:
            return None
        category0 = (category or "").strip()
        category0 = self._slug_category(category0) or category0 or "misc"
        self.add_category(category0)

        kws = self._load_keywords()
        kw_tokens: List[str] = []
        for kid in (tag_ids or []):
            k = kws.get(kid)
            if not k:
                continue
            nm = (k.get("name") or "").strip()
            if nm:
                kw_tokens.append(f"{k.get('category','misc')}::{nm}")

        path = self._packs_primary_file(category0)
        existing_blocks: List[Dict[str, Any]] = []
        if path.exists():
            try:
                existing_blocks = self._parse_pack_blocks(path.read_text(encoding="utf-8", errors="ignore"), source_file=str(path))
            except Exception:
                existing_blocks = []

        if pid:
            packs = self._load_packs()
            old = packs.get(pid)
            if old:
                old_file = Path(old.get("source_file") or "")
                try:
                    old_blocks = self._parse_pack_blocks(old_file.read_text(encoding="utf-8", errors="ignore"), source_file=str(old_file))
                    old_blocks = [b for b in old_blocks if b.get("id") != pid]
                    self._write_pack_blocks_to_file(old_file, old_blocks)
                except Exception:
                    pass

        if not pid:
            pid = str(uuid.uuid4())

        new_block = {"id": pid, "title": title, "keywords_raw": kw_tokens, "note": "", "source_file": str(path), "category": category0}
        existing_blocks = [b for b in existing_blocks if b.get("id") != pid]
        existing_blocks.append(new_block)
        existing_blocks = sorted(existing_blocks, key=lambda b: (b.get("title") or "").casefold())
        self._write_pack_blocks_to_file(path, existing_blocks)
        return pid

    def delete_pack(self, pid: str):
        packs = self._load_packs()
        p = packs.get(pid)
        if not p:
            return
        fp = Path(p.get("source_file") or "")
        if not fp.exists():
            return
        blocks = self._parse_pack_blocks(fp.read_text(encoding="utf-8", errors="ignore"), source_file=str(fp))
        blocks = [b for b in blocks if b.get("id") != pid]
        self._write_pack_blocks_to_file(fp, blocks)

    def resolve_pack_tags(self, pack_id: str) -> List[Dict[str, Any]]:
        p = self.get_pack(pack_id)
        if not p:
            return []
        kws = self._load_keywords()
        out = []
        for token in (p.get("keywords_raw") or []):
            cat, name = self._split_kw_token(token)
            if cat and name:
                kid = self._kw_id(cat, name)
                k = kws.get(kid)
                if k and k.get("enabled", True):
                    out.append(k)
                else:
                    out.append({"id": kid, "category": cat, "name": name, "aliases": [], "desc": "", "enabled": True})
            elif name:
                out.append({"id": "kw::misc::" + _norm_token(name), "category": "misc", "name": name, "aliases": [], "desc": "", "enabled": True})
        return out
    # ------------------ MapSets ------------------
    def list_mapset_choices(self, q: str = "") -> List[Tuple[str, str]]:
        qn = _norm_token(q)
        out: List[Tuple[str, str]] = []
        for m in self.data.get("mapsets", []):
            mid = m.get("id") or ""
            title = m.get("title") or ""
            label = title
            if not mid or not title:
                continue
            if qn and qn not in _norm_token(label + " " + " ".join(m.get("tags") or [])):
                continue
            out.append((label, mid))
        return out

    def get_mapset(self, mid: str) -> Optional[Dict[str, Any]]:
        for m in self.data.get("mapsets", []):
            if m.get("id") == mid:
                return m
        return None

    def create_mapset(self, title: str, tags_csv: str = "") -> str:
        mid = str(uuid.uuid4())
        tags = [t.strip() for t in (tags_csv or "").split(",") if t.strip()]
        item = {
            "id": mid,
            "title": _safe_name(title),
            "tags": tags,
            "created": _now_iso(),
            "updated": _now_iso(),
            "maps": {t: [] for t in _MAP_TYPES},
        }
        self.data.setdefault("mapsets", []).append(item)
        for t in _MAP_TYPES:
            (ASSETS_DIR / mid / t).mkdir(parents=True, exist_ok=True)
        self.save()
        return mid

    def update_mapset_meta(self, mid: str, title: str, tags_csv: str):
        m = self.get_mapset(mid)
        if not m:
            return
        m["title"] = _safe_name(title)
        m["tags"] = [t.strip() for t in (tags_csv or "").split(",") if t.strip()]
        m["updated"] = _now_iso()
        self.save()

    def delete_mapset(self, mid: str):
        self.data["mapsets"] = [m for m in self.data.get("mapsets", []) if m.get("id") != mid]
        folder = ASSETS_DIR / mid
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        self.save()

    def add_maps_to_mapset(
        self,
        mid: str,
        files: List[Any],
        map_type: str,
        auto_detect: bool,
        enforce_suffix: bool,
    ) -> int:
        m = self.get_mapset(mid)
        if not m:
            return 0
        count = 0
        for f in files or []:
            src = getattr(f, "name", None) or str(f)
            if not src or not os.path.exists(src):
                continue
            chosen = map_type
            if auto_detect:
                g = _guess_map_type(os.path.basename(src))
                if g:
                    chosen = g
            if chosen not in _MAP_TYPES:
                continue

            dst_dir = ASSETS_DIR / mid / chosen
            dst_dir.mkdir(parents=True, exist_ok=True)
            stem = _safe_name(Path(src).stem, max_len=90)
            if enforce_suffix:
                stem = _ensure_suffix(stem, chosen)
            suffix = Path(src).suffix.lower() or ".png"
            dst = dst_dir / f"{stem}{suffix}"
            if dst.exists():
                dst = dst_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
            shutil.copy2(src, dst)

            rel = str(dst.relative_to(USER_DATA)).replace("\\", "/")
            m["maps"][chosen].append(rel)
            count += 1

        m["updated"] = _now_iso()
        self.save()
        return count

    def list_map_paths(self, mid: str, map_type: str) -> List[Tuple[str, str]]:
        m = self.get_mapset(mid)
        if not m:
            return []
        rels = m.get("maps", {}).get(map_type, []) or []
        out: List[Tuple[str, str]] = []
        for r in rels:
            p = USER_DATA / r
            if p.exists():
                out.append((str(p), os.path.basename(str(p))))
        return out

    # ------------------ LoRA / TI Registry ------------------
    def _norm_path(self, p: str) -> str:
        try:
            return os.path.abspath(p).replace("\\", "/").lower()
        except Exception:
            return (p or "").replace("\\", "/").lower()

    def _default_lora_dir(self) -> str:
        """Best-effort guess of Forge's LoRA dir."""
        # Try A1111/Forge conventions
        try:
            from modules import shared
            d = getattr(getattr(shared, "cmd_opts", None), "lora_dir", None)
            if d and os.path.isdir(d):
                return d
        except Exception:
            pass
        try:
            from modules import paths
            d = os.path.join(paths.models_path, "Lora")
            if os.path.isdir(d):
                return d
        except Exception:
            pass
        # Fallback: relative to webui root
        try:
            here = str(EXT_ROOT)
            # EXT_ROOT/.../extensions/prompt_suite_bridge -> .../sd-webui-forge-neo
            root = Path(here).parents[2]
            d = str(root / "models" / "Lora")
            if os.path.isdir(d):
                return d
        except Exception:
            pass
        return ""

    def _default_embed_dir(self) -> str:
        """Best-effort guess of Forge's embeddings dir."""
        try:
            from modules import shared
            d = getattr(getattr(shared, "cmd_opts", None), "embeddings_dir", None)
            if d and os.path.isdir(d):
                return d
        except Exception:
            pass
        try:
            from modules import paths
            d = os.path.join(paths.script_path, "embeddings")
            if os.path.isdir(d):
                return d
        except Exception:
            pass
        try:
            here = str(EXT_ROOT)
            root = Path(here).parents[2]
            d = str(root / "embeddings")
            if os.path.isdir(d):
                return d
        except Exception:
            pass
        return ""

    def list_lora_choices(self, q: str = "", kind: str = "lora") -> List[Tuple[str, str]]:
        qn = _norm_token(q)
        out: List[Tuple[str, str]] = []
        for it in self.data.get("loras", []) or []:
            if (it.get("kind") or "lora") != kind:
                continue
            lid = it.get("id") or ""
            rel = (it.get("rel") or it.get("name") or "").strip()
            cat = (it.get("category") or "").strip()
            name = (it.get("name") or "").strip()
            label = f"{cat} › {rel}" if cat else rel
            if not lid or not name:
                continue
            if qn:
                hay = _norm_token(label + " " + " ".join(it.get("triggers") or []) + " " + " ".join(it.get("keywords") or []))
                if qn not in hay:
                    continue
            out.append((label, lid))
        return out

    def get_lora(self, lid: str) -> Optional[Dict[str, Any]]:
        for it in self.data.get("loras", []) or []:
            if it.get("id") == lid:
                return it
        return None

    def upsert_lora_meta(
        self,
        lid: str,
        triggers_csv: str,
        keywords_csv: str,
        default_strength: float,
        notes: str,
        enabled: bool = True,
    ) -> Optional[str]:
        it = self.get_lora(lid) if lid else None
        if it is None:
            return None
        triggers = [x.strip() for x in (triggers_csv or "").split(",") if x.strip()]
        keywords = [x.strip() for x in (keywords_csv or "").split(",") if x.strip()]
        it.update({
            "triggers": triggers,
            "keywords": keywords,
            "default_strength": float(default_strength) if default_strength is not None else float(it.get("default_strength") or 1.0),
            "notes": notes or "",
            "enabled": bool(enabled),
            "updated": _now_iso(),
        })
        self.save()
        return it.get("id")

    def delete_lora(self, lid: str):
        self.data["loras"] = [x for x in (self.data.get("loras") or []) if x.get("id") != lid]
        self.save()

    def scan_loras(self, lora_dir: str = "", embed_dir: str = "", include_ti: bool = True) -> Tuple[int, int]:
        """Scan LoRA/Embeddings folders and upsert missing entries. Returns (added, updated)."""
        lora_dir = (lora_dir or self._default_lora_dir()).strip()
        embed_dir = (embed_dir or self._default_embed_dir()).strip()

        existing = self.data.get("loras") or []
        by_file = {self._norm_path(x.get("file") or ""): x for x in existing if x.get("file")}

        added = 0
        updated = 0

        def _walk(base: str, exts: Tuple[str, ...]) -> List[str]:
            out: List[str] = []
            if not base or not os.path.isdir(base):
                return out
            for root, _, files in os.walk(base):
                for fn in files:
                    if fn.lower().endswith(tuple([e.lower() for e in exts])):
                        out.append(os.path.join(root, fn))
            return out

        # LoRAs
        for fp in _walk(lora_dir, _LORA_EXTS):
            nfp = self._norm_path(fp)
            rel = ""
            cat = ""
            try:
                relp = os.path.relpath(fp, lora_dir)
                rel = os.path.splitext(relp)[0].replace("\\", "/")
                cat = os.path.dirname(rel).replace("\\", "/").strip("/")
            except Exception:
                rel = os.path.splitext(os.path.basename(fp))[0]
                cat = ""
            name = os.path.splitext(os.path.basename(fp))[0]
            if nfp in by_file:
                it = by_file[nfp]
                # update discovered fields only
                it.update({
                    "kind": "lora",
                    "file": fp,
                    "rel": rel,
                    "name": name,
                    "category": cat,
                    "updated": _now_iso(),
                })
                updated += 1
                continue
            existing.append({
                "id": str(uuid.uuid4()),
                "kind": "lora",
                "file": fp,
                "rel": rel,
                "name": name,
                "category": cat,
                "triggers": [],
                "keywords": [],
                "default_strength": 0.8,
                "notes": "",
                "enabled": True,
                "created": _now_iso(),
                "updated": _now_iso(),
            })
            added += 1

        # Embeddings (TI)
        if include_ti:
            for fp in _walk(embed_dir, _EMBED_EXTS):
                nfp = self._norm_path(fp)
                name = os.path.splitext(os.path.basename(fp))[0]
                rel = name
                cat = ""
                if nfp in by_file:
                    it = by_file[nfp]
                    it.update({
                        "kind": "ti",
                        "file": fp,
                        "rel": rel,
                        "name": name,
                        "category": cat,
                        "updated": _now_iso(),
                    })
                    updated += 1
                    continue
                existing.append({
                    "id": str(uuid.uuid4()),
                    "kind": "ti",
                    "file": fp,
                    "rel": rel,
                    "name": name,
                    "category": cat,
                    "triggers": [],
                    "keywords": [],
                    "default_strength": 1.0,
                    "notes": "",
                    "enabled": True,
                    "created": _now_iso(),
                    "updated": _now_iso(),
                })
                added += 1

        self.data["loras"] = existing
        self.save()
        return added, updated

    # ------------------ Legacy migration ------------------
    def maybe_migrate_legacy_prompt_map_vault(self):
        """
        If an old data/library.json exists (prompt_map_vault style), migrate:
        - each legacy prompt -> new mapset with copied assets
        - (prompt text migration handled by PromptPresetStore)
        """
        legacy_lib = EXT_ROOT / "data" / "library.json"
        legacy_assets = EXT_ROOT / "data" / "assets"
        marker = USER_DATA / ".legacy_migrated_v1"
        if marker.exists():
            return
        if not legacy_lib.exists():
            marker.write_text("no legacy", encoding="utf-8")
            return
        try:
            legacy = json.loads(legacy_lib.read_text(encoding="utf-8"))
        except Exception:
            marker.write_text("bad legacy json", encoding="utf-8")
            return

        prompts = legacy.get("prompts", []) or []
        if not prompts:
            marker.write_text("empty legacy", encoding="utf-8")
            return

        # only migrate if vault is empty
        if self.data.get("mapsets"):
            marker.write_text("skipped - already has mapsets", encoding="utf-8")
            return

        for lp in prompts:
            pid = lp.get("id")
            title = lp.get("title") or "Legacy"
            if not pid:
                continue
            mid = self.create_mapset(f"{title} (legacy)", tags_csv="")
            # copy assets folder if exists
            if legacy_assets.exists():
                for t in _MAP_TYPES:
                    src_dir = legacy_assets / pid / t
                    if not src_dir.exists():
                        continue
                    dst_dir = ASSETS_DIR / mid / t
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    for fn in src_dir.iterdir():
                        if fn.is_file():
                            stem = _safe_name(fn.stem, max_len=90)
                            # keep legacy name
                            dst = dst_dir / f"{stem}{fn.suffix}"
                            if dst.exists():
                                dst = dst_dir / f"{stem}_{uuid.uuid4().hex[:6]}{fn.suffix}"
                            shutil.copy2(str(fn), str(dst))
                            rel = str(dst.relative_to(USER_DATA)).replace("\\", "/")
                            m = self.get_mapset(mid)
                            if m:
                                m["maps"][t].append(rel)
            self.save()

        marker.write_text("done", encoding="utf-8")
