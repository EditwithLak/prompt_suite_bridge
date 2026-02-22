
import gradio as gr
import os
import sys
import subprocess
import shutil
from pathlib import Path

from vault_store import VaultStore

EXT_NAME = "Vault + Maps (Assets)"
EXT_VER = "v0.2"

EXT_ROOT = Path(__file__).resolve().parents[1]
LIBRARIES_DIR = EXT_ROOT / "libraries"


ASSETS_DIR = EXT_ROOT / "user_data" / "assets"


def _open_folder(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return "⚠️ No folder path."
    try:
        if not os.path.exists(p):
            return f"⚠️ Folder not found: {p}"
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
        return f"✅ Opened: {p}"
    except Exception as e:
        return f"⚠️ Could not open folder: {e}"
def build_vault_ui():
    store = VaultStore()
    # migrate legacy assets once (safe)
    try:
        store.maybe_migrate_legacy_prompt_map_vault()
    except Exception:
        pass

    tag_id = gr.State(value="")
    pack_id = gr.State(value="")
    base_id = gr.State(value="")
    mapset_id = gr.State(value="")

    with gr.Column():
        gr.Markdown(f"### 🗃️ {EXT_NAME}")
        gr.Markdown("This tab is **asset management only**: tags/packs/mapsets. No sending to txt2img/img2img.")

        with gr.Tabs():

            # ---------------- Library (Keywords + Packs) ----------------
            with gr.Tab("🔑 Library"):
                gr.Markdown("Everything here is **library-driven** from the `libraries/` folder. No built-ins.")

                with gr.Row():
                    show_keywords = gr.Checkbox(label="Keywords", value=True)
                    show_packs = gr.Checkbox(label="Packs", value=True)
                    show_bases = gr.Checkbox(label="Base prompts", value=True)

                with gr.Row():
                    new_cat = gr.Textbox(label="Add new category", placeholder="e.g. pose_action, wardrobe, lighting…", lines=1)
                    btn_add_cat = gr.Button("➕ Add", scale=0)
                cat_msg = gr.Markdown("")

                # -------- Keywords --------
                kw_col = gr.Column(visible=True)
                with kw_col:
                    gr.Markdown("#### 🧩 Keywords")
                    with gr.Row():
                        kw_filter = gr.Dropdown(
                            label="Filter category",
                            choices=["all"] + store.list_categories(),
                            value="all"
                        )
                        kw_search = gr.Textbox(label="Search keywords", placeholder="piggyback / streetwear / soft light…", lines=1)
                        kw_dd = gr.Dropdown(label="Saved keywords", choices=store.list_tag_choices(category="all"), value=None)

                    kw_category = gr.Dropdown(
                        choices=store.list_categories(),
                        value=(store.list_categories()[0] if store.list_categories() else None),
                        label="Category"
                    )
                    kw_name = gr.Textbox(label="Keyword (canonical name)", placeholder="piggyback ride")
                    kw_aliases = gr.Textbox(label="Aliases (comma-separated)", placeholder="piggyBackride, piggy back ride, piggy-back ride")
                    kw_desc = gr.Textbox(label="Description (optional)", lines=4)
                    kw_enabled = gr.Checkbox(value=True, label="Enabled")

                    with gr.Row():
                        btn_kw_new = gr.Button("➕ New")
                        btn_kw_save = gr.Button("💾 Save")
                        btn_kw_del = gr.Button("🗑️ Delete", variant="stop")
                    kw_status = gr.Markdown("")

                # -------- Packs --------
                pk_col = gr.Column(visible=True)
                with pk_col:
                    gr.Markdown("#### 📦 Packs")
                    gr.Markdown("A pack is a curated list of keywords for fast insertion later in Prompt Builder.")

                    with gr.Row():
                        pk_filter = gr.Dropdown(
                            label="Filter category",
                            choices=["all"] + store.list_categories(),
                            value="all"
                        )
                        pk_search = gr.Textbox(label="Search packs", placeholder="romantic carries, streetwear sets…", lines=1)
                        pk_dd = gr.Dropdown(label="Saved packs", choices=store.list_pack_choices(category="all"), value=None)

                    pk_category = gr.Dropdown(
                        choices=store.list_categories(),
                        value=(store.list_categories()[0] if store.list_categories() else None),
                        label="Category"
                    )
                    pk_title = gr.Textbox(label="Pack title", placeholder="Romantic Carries")
                    pk_keywords = gr.Dropdown(
                        label="Keywords in this pack",
                        choices=store.list_tag_choices(category="all"),
                        multiselect=True,
                        value=[]
                    )

                    with gr.Row():
                        btn_pk_new = gr.Button("➕ New")
                        btn_pk_save = gr.Button("💾 Save")
                        btn_pk_del = gr.Button("🗑️ Delete", variant="stop")
                    pk_status = gr.Markdown("")
                    gr.Button("🔄 Refresh keyword list").click(fn=lambda: gr.update(choices=VaultStore().list_tag_choices(category="all")), inputs=[], outputs=[pk_keywords])

                # -------- Visibility toggles --------
                
                # -------- Base Prompts --------
                bp_col = gr.Column(visible=True)
                with bp_col:
                    gr.Markdown("#### 🧱 Base prompts")
                    gr.Markdown("Templates that include **[tag1]...[tagN]** or **[tags]** placeholders for fast building in Prompt Composer.")

                    with gr.Row():
                        bp_filter = gr.Dropdown(
                            label="Filter category (bases)",
                            choices=["all"] + store.list_categories(kinds=["bases"]),
                            value="all"
                        )
                        bp_search = gr.Textbox(label="Search base prompts", placeholder="ambulance / studio / rain…", lines=1)
                        bp_dd = gr.Dropdown(label="Saved base prompts", choices=store.list_base_choices(category="all"), value=None)

                    bp_category = gr.Dropdown(
                        choices=store.list_categories(),
                        value=(store.list_categories()[0] if store.list_categories() else None),
                        label="Category"
                    )
                    bp_title = gr.Textbox(label="Base prompt title", placeholder="Cramped Ambulance Tension")
                    bp_slots = gr.Slider(minimum=0, maximum=12, step=1, value=0, label="Slots (0 = auto)")
                    bp_template = gr.Textbox(
                        label="Template",
                        lines=6,
                        placeholder="cramped ambulance interior, ..., [tag1], [tag2], ...  (or use [tags])"
                    )

                    with gr.Row():
                        btn_bp_new = gr.Button("➕ New")
                        btn_bp_save = gr.Button("💾 Save")
                        btn_bp_del = gr.Button("🗑️ Delete", variant="stop")
                    bp_status = gr.Markdown("")

                # -------- Base prompt behaviors --------
                def _refresh_bases(q: str, cat_filter: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_base_choices(q=q, category=cat_filter), value=None)

                def _load_base(sel: str):
                    if not sel:
                        return "", None, "", 0, "", "⚠️ Select a base prompt."
                    s = VaultStore()
                    b = s.get_base(sel) or {}
                    return sel, b.get("category"), b.get("title"), int(b.get("slots") or 0), b.get("template"), "✅ Loaded."

                def _new_base():
                    s = VaultStore()
                    cats = s.list_categories()
                    return "", (cats[0] if cats else None), "", 0, "", "✅ New base prompt."

                def _save_base(bid: str, cat: str, title: str, slots: int, template: str, fcat: str, q: str):
                    s = VaultStore()
                    key = s.upsert_base(bid, cat, title, slots, template)
                    if not key:
                        return gr.update(), bid, "❌ Missing category/title/template."
                    dd = s.list_base_choices(q=q, category=fcat)
                    return gr.update(choices=dd, value=key), key, "✅ Saved."

                def _del_base(bid: str, fcat: str, q: str):
                    s = VaultStore()
                    ok = s.delete_base(bid)
                    dd = s.list_base_choices(q=q, category=fcat)
                    msg = "✅ Deleted." if ok else "ℹ️ Nothing to delete."
                    return gr.update(choices=dd, value=None), "", msg

                bp_filter.change(_refresh_bases, inputs=[bp_search, bp_filter], outputs=[bp_dd])
                bp_search.change(_refresh_bases, inputs=[bp_search, bp_filter], outputs=[bp_dd])
                bp_dd.change(_load_base, inputs=[bp_dd], outputs=[base_id, bp_category, bp_title, bp_slots, bp_template, bp_status])

                btn_bp_new.click(_new_base, inputs=[], outputs=[base_id, bp_category, bp_title, bp_slots, bp_template, bp_status])
                btn_bp_save.click(_save_base, inputs=[base_id, bp_category, bp_title, bp_slots, bp_template, bp_filter, bp_search], outputs=[bp_dd, base_id, bp_status])
                btn_bp_del.click(_del_base, inputs=[base_id, bp_filter, bp_search], outputs=[bp_dd, base_id, bp_status])


                show_keywords.change(lambda v: gr.update(visible=bool(v)), inputs=[show_keywords], outputs=[kw_col])
                show_packs.change(lambda v: gr.update(visible=bool(v)), inputs=[show_packs], outputs=[pk_col])
                show_bases.change(lambda v: gr.update(visible=bool(v)), inputs=[show_bases], outputs=[bp_col])

                # -------- Shared: add category --------
                def _add_category(cat_name: str):
                    s = VaultStore()
                    ok = s.add_category(cat_name)
                    cats = s.list_categories()
                    msg = "✅ Category added." if ok else "ℹ️ Category already exists (or empty)."
                    return (
                        gr.update(choices=cats, value=(cats[0] if cats else None)),
                        gr.update(choices=cats, value=(cats[0] if cats else None)),
                        gr.update(choices=cats, value=(cats[0] if cats else None)),
                        gr.update(choices=["all"] + cats, value="all"),
                        gr.update(choices=["all"] + cats, value="all"),
                        gr.update(choices=["all"] + cats, value="all"),
                        msg
                    )

                btn_add_cat.click(_add_category, inputs=[new_cat], outputs=[kw_category, pk_category, bp_category, kw_filter, pk_filter, bp_filter, cat_msg])

                # -------- Keyword behaviors --------
                def _refresh_keywords(q: str, cat_filter: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_tag_choices(q=q, category=cat_filter))

                def _filter_keywords(cat_filter: str, q: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_tag_choices(q=q, category=cat_filter), value=None)

                def _load_keyword(tid: str):
                    s = VaultStore()
                    t = s.get_tag(tid) if tid else None
                    if not t:
                        return "", (s.list_categories()[0] if s.list_categories() else None), "", "", "", True, "⚠️ Select a keyword."
                    return tid, t.get("category",""), t.get("name",""), ", ".join(t.get("aliases") or []), t.get("desc",""), bool(t.get("enabled",True)), "✅ Loaded."

                def _new_keyword():
                    s = VaultStore()
                    return "", (s.list_categories()[0] if s.list_categories() else None), "", "", "", True, "✅ New keyword."

                def _save_keyword(tid, cat, name, aliases, desc, enabled, cat_filter: str, q: str):
                    s = VaultStore()
                    out_id = s.upsert_tag(tid, cat, name, aliases, desc, enabled)
                    choices = s.list_tag_choices(q=q, category=cat_filter)
                    if not out_id:
                        return gr.update(choices=choices, value=None), "", "⚠️ Keyword name is required."
                    ok_val = out_id if any(v == out_id for (_, v) in choices) else None
                    pk_choices = s.list_tag_choices(category="all")
                    return gr.update(choices=choices, value=ok_val), out_id, "✅ Saved.", gr.update(choices=pk_choices)

                def _del_keyword(tid: str, cat_filter: str, q: str):
                    s = VaultStore()
                    if not tid:
                        return gr.update(choices=s.list_tag_choices(q=q, category=cat_filter), value=None), "", "⚠️ Nothing selected.", gr.update(choices=s.list_tag_choices(category="all"))
                    s.delete_tag(tid)
                    return gr.update(choices=s.list_tag_choices(q=q, category=cat_filter), value=None), "", "🗑️ Deleted.", gr.update(choices=s.list_tag_choices(category="all"))

                kw_search.change(_refresh_keywords, inputs=[kw_search, kw_filter], outputs=[kw_dd])
                kw_filter.change(_filter_keywords, inputs=[kw_filter, kw_search], outputs=[kw_dd])
                kw_dd.change(_load_keyword, inputs=[kw_dd], outputs=[tag_id, kw_category, kw_name, kw_aliases, kw_desc, kw_enabled, kw_status])
                btn_kw_new.click(_new_keyword, inputs=[], outputs=[tag_id, kw_category, kw_name, kw_aliases, kw_desc, kw_enabled, kw_status])
                btn_kw_save.click(_save_keyword, inputs=[tag_id, kw_category, kw_name, kw_aliases, kw_desc, kw_enabled, kw_filter, kw_search], outputs=[kw_dd, tag_id, kw_status, pk_keywords])
                btn_kw_del.click(_del_keyword, inputs=[tag_id, kw_filter, kw_search], outputs=[kw_dd, tag_id, kw_status, pk_keywords])

                # -------- Pack behaviors --------
                def _refresh_packs(q: str, cat_filter: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_pack_choices(q=q, category=cat_filter))

                def _filter_packs(cat_filter: str, q: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_pack_choices(q=q, category=cat_filter), value=None)

                def _load_pack(pid: str):
                    s = VaultStore()
                    p0 = s.get_pack(pid) if pid else None
                    if not p0:
                        return "", (s.list_categories()[0] if s.list_categories() else None), "", [], "⚠️ Select a pack."
                    kws = s._load_keywords()
                    selected = []
                    for token in (p0.get("keywords_raw") or []):
                        cat, nm = s._split_kw_token(token)
                        if not cat or not nm:
                            continue
                        kid = s._kw_id(cat, nm)
                        if kid in kws:
                            selected.append(kid)
                    return pid, p0.get("category",""), p0.get("title",""), selected, "✅ Loaded."

                def _new_pack():
                    s = VaultStore()
                    return "", (s.list_categories()[0] if s.list_categories() else None), "", [], "✅ New pack."

                def _save_pack(pid, cat, title, kw_ids, cat_filter: str, q: str):
                    s = VaultStore()
                    out_id = s.upsert_pack(pid, cat, title, kw_ids or [])
                    choices = s.list_pack_choices(q=q, category=cat_filter)
                    if not out_id:
                        return gr.update(choices=choices, value=None), "", "⚠️ Pack title is required."
                    ok_val = out_id if any(v == out_id for (_, v) in choices) else None
                    return gr.update(choices=choices, value=ok_val), out_id, "✅ Saved."

                def _del_pack(pid: str, cat_filter: str, q: str):
                    s = VaultStore()
                    if not pid:
                        return gr.update(choices=s.list_pack_choices(q=q, category=cat_filter), value=None), "", "⚠️ Nothing selected."
                    s.delete_pack(pid)
                    return gr.update(choices=s.list_pack_choices(q=q, category=cat_filter), value=None), "", "🗑️ Deleted."

                pk_search.change(_refresh_packs, inputs=[pk_search, pk_filter], outputs=[pk_dd])
                pk_filter.change(_filter_packs, inputs=[pk_filter, pk_search], outputs=[pk_dd])
                pk_dd.change(_load_pack, inputs=[pk_dd], outputs=[pack_id, pk_category, pk_title, pk_keywords, pk_status])
                btn_pk_new.click(_new_pack, inputs=[], outputs=[pack_id, pk_category, pk_title, pk_keywords, pk_status])
                btn_pk_save.click(_save_pack, inputs=[pack_id, pk_category, pk_title, pk_keywords, pk_filter, pk_search], outputs=[pk_dd, pack_id, pk_status])
                btn_pk_del.click(_del_pack, inputs=[pack_id, pk_filter, pk_search], outputs=[pk_dd, pack_id, pk_status])

            # ---------------- LoRA / TI ----------------
            with gr.Tab("🎛️ LoRA / TI"):
                gr.Markdown("Register your existing **LoRAs** (and optional **Textual Inversions**) so Prompt Builder can insert them + show your trigger tags.")

                # dirs + scan
                with gr.Row():
                    lora_dir = gr.Textbox(label="LoRA folder", value=store._default_lora_dir(), placeholder=r"F:\LLM\sd-webui-forge-neo\models\Lora")
                    embed_dir = gr.Textbox(label="Embeddings folder (TI)", value=store._default_embed_dir(), placeholder=r"F:\LLM\sd-webui-forge-neo\embeddings")

                include_ti = gr.Checkbox(label="Also scan embeddings (TI)", value=True)
                scan_btn = gr.Button("🔎 Scan folders")
                scan_msg = gr.Markdown("")

                with gr.Row():
                    lora_kind = gr.Radio(label="Kind", choices=["lora", "ti"], value="lora")
                    lora_search = gr.Textbox(label="Search", placeholder="portrait, skin, style, anime…", lines=1)

                lora_dd = gr.Dropdown(label="Registered", choices=store.list_lora_choices(kind="lora"), value=None)
                lora_id = gr.State(value="")

                lora_file = gr.Textbox(label="File (read-only)", value="", interactive=False)
                lora_rel = gr.Textbox(label="Prompt token (read-only)", value="", interactive=False)
                lora_cat = gr.Textbox(label="Category (folder)", value="", interactive=False)
                lora_name = gr.Textbox(label="Name (read-only)", value="", interactive=False)

                lora_strength = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Default strength")
                lora_triggers = gr.Textbox(label="Triggers (comma-separated)", placeholder="trigger_word, keyword2, style_tag")
                lora_keywords = gr.Textbox(label="Keywords (comma-separated)", placeholder="portrait, skin, hair, lighting")
                lora_notes = gr.Textbox(label="Notes", lines=3, placeholder="What this LoRA is good for…")
                lora_enabled = gr.Checkbox(label="Enabled", value=True)

                with gr.Row():
                    lora_save = gr.Button("💾 Save meta")
                    lora_del = gr.Button("🗑️ Delete", variant="stop")

                lora_status = gr.Markdown("")

                def _scan(l_dir: str, e_dir: str, inc_ti: bool, kind: str):
                    s = VaultStore()
                    added, updated = s.scan_loras(l_dir, e_dir, include_ti=bool(inc_ti))
                    # refresh dropdown for current kind
                    return gr.update(choices=s.list_lora_choices(kind=kind), value=None), f"✅ Scan complete. Added: **{added}**, updated: **{updated}**"

                def _refresh_loras(q: str, kind: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_lora_choices(q=q, kind=kind))

                def _kind_change(kind: str, q: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_lora_choices(q=q, kind=kind), value=None)

                def _load_lora(lid: str):
                    s = VaultStore()
                    it = s.get_lora(lid) if lid else None
                    if not it:
                        return "", "", "", "", "", 0.8, "", "", "", True, "⚠️ Select an item."
                    return (
                        lid,
                        it.get("file",""),
                        it.get("rel",""),
                        it.get("category",""),
                        it.get("name",""),
                        float(it.get("default_strength") or 1.0),
                        ", ".join(it.get("triggers") or []),
                        ", ".join(it.get("keywords") or []),
                        it.get("notes",""),
                        bool(it.get("enabled", True)),
                        "✅ Loaded."
                    )

                def _save_lora(lid: str, strength: float, triggers: str, keywords: str, notes: str, enabled: bool, kind: str, q: str):
                    s = VaultStore()
                    ok = s.upsert_lora_meta(lid, triggers, keywords, strength, notes, enabled=enabled)
                    if not ok:
                        return gr.update(choices=s.list_lora_choices(q=q, kind=kind)), "⚠️ Nothing to save (select a LoRA first)."
                    return gr.update(choices=s.list_lora_choices(q=q, kind=kind), value=lid), "✅ Saved."

                def _del_lora(lid: str, kind: str, q: str):
                    if not lid:
                        return gr.update(choices=store.list_lora_choices(q=q, kind=kind), value=None), "⚠️ Nothing selected."
                    s = VaultStore()
                    s.delete_lora(lid)
                    return gr.update(choices=s.list_lora_choices(q=q, kind=kind), value=None), "🗑️ Deleted."

                scan_btn.click(_scan, inputs=[lora_dir, embed_dir, include_ti, lora_kind], outputs=[lora_dd, scan_msg])
                lora_search.change(_refresh_loras, inputs=[lora_search, lora_kind], outputs=[lora_dd])
                lora_kind.change(_kind_change, inputs=[lora_kind, lora_search], outputs=[lora_dd])
                lora_dd.change(_load_lora, inputs=[lora_dd], outputs=[lora_id, lora_file, lora_rel, lora_cat, lora_name, lora_strength, lora_triggers, lora_keywords, lora_notes, lora_enabled, lora_status])
                lora_save.click(_save_lora, inputs=[lora_id, lora_strength, lora_triggers, lora_keywords, lora_notes, lora_enabled, lora_kind, lora_search], outputs=[lora_dd, lora_status])
                lora_del.click(_del_lora, inputs=[lora_id, lora_kind, lora_search], outputs=[lora_dd, lora_status])

            # ---------------- MapSets ----------------


            # ---------------- Libraries (built-in lists) ----------------
            
            with gr.Accordion("📚 Library Files (keywords / packs / legacy lists) — v0.4.11", open=True):
                gr.Markdown(
                    "Manage your **library `.md` files** inside the extension `libraries/` folder.\n\n"
                    "- Import new `.md` files here (no Forge restart).\n"
                    "- Edit existing files and save.\n"
                    "- The Prompt Composer typeahead reads the libraries live as you type."
                )

                def _list_lib_md():
                    if not LIBRARIES_DIR.exists():
                        return []
                    return [p.name for p in sorted(LIBRARIES_DIR.glob("*.md"))]

                with gr.Row():
                    lib_file = gr.Dropdown(label="Library file (.md)", choices=_list_lib_md(), value=None, scale=3)
                    btn_lib_refresh = gr.Button("🔄 Refresh list", scale=1)

                lib_text = gr.Textbox(label="File content", lines=18, placeholder="Select a library file to view/edit…")

                with gr.Row():
                    btn_lib_load = gr.Button("↻ Load")
                    btn_lib_save = gr.Button("💾 Save")

                with gr.Accordion("⬆️ Import .md into libraries/", open=True):
                    lib_upload = gr.File(label="Drop .md file(s) here", file_count="multiple", file_types=[".md"])
                    lib_overwrite = gr.Checkbox(value=False, label="Overwrite if same name exists")
                    btn_lib_import = gr.Button("Import file(s)")

                lib_msg = gr.Markdown("")

                def _load_lib(fn: str):
                    if not fn:
                        return "", "⚠️ Select a file."
                    path = LIBRARIES_DIR / fn
                    try:
                        txt = path.read_text(encoding="utf-8")
                    except Exception as e:
                        return "", f"❌ Read failed: `{e}`"
                    return txt, f"✅ Loaded `{fn}`"

                def _save_lib(fn: str, txt: str):
                    if not fn:
                        return "⚠️ Select a file."
                    path = LIBRARIES_DIR / fn
                    try:
                        # backup
                        if path.exists():
                            try:
                                (LIBRARIES_DIR / (fn + ".bak")).write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                            except Exception:
                                pass
                        path.write_text(txt or "", encoding="utf-8")
                    except Exception as e:
                        return f"❌ Save failed: `{e}`"
                    return f"✅ Saved `{fn}`."

                def _refresh_lib_list():
                    return gr.update(choices=_list_lib_md(), value=None), "✅ Refreshed list."

                def _import_lib(files, overwrite: bool):
                    LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)
                    if not files:
                        return gr.update(choices=_list_lib_md()), "⚠️ No files selected."
                    imported = 0
                    skipped = 0

                    def _fp(x):
                        if isinstance(x, str):
                            return x
                        if isinstance(x, dict):
                            return x.get("name") or ""
                        return getattr(x, "name", "") or ""

                    for f in (files or []):
                        fp = _fp(f)
                        if not fp:
                            skipped += 1
                            continue
                        src = Path(fp)
                        if src.suffix.lower() != ".md":
                            skipped += 1
                            continue
                        dst = LIBRARIES_DIR / src.name
                        if dst.exists() and not overwrite:
                            skipped += 1
                            continue
                        try:
                            shutil.copyfile(str(src), str(dst))
                            imported += 1
                        except Exception:
                            skipped += 1

                    msg = f"✅ Imported {imported} file(s)." + (f" Skipped {skipped}." if skipped else "")
                    return gr.update(choices=_list_lib_md()), msg

                btn_lib_load.click(fn=_load_lib, inputs=[lib_file], outputs=[lib_text, lib_msg])
                lib_file.change(fn=_load_lib, inputs=[lib_file], outputs=[lib_text, lib_msg])
                btn_lib_save.click(fn=_save_lib, inputs=[lib_file, lib_text], outputs=[lib_msg])

                btn_lib_refresh.click(fn=_refresh_lib_list, inputs=[], outputs=[lib_file, lib_msg])
                btn_lib_import.click(fn=_import_lib, inputs=[lib_upload, lib_overwrite], outputs=[lib_file, lib_msg])
            with gr.Tab("🗺️ MapSets"):

                gr.Markdown("A **mapset** is a saved set of maps (canny/depth/openpose) you can load later in Prompt Builder.")

                with gr.Row():
                    map_search = gr.Textbox(label="Search mapsets", placeholder="piggyback, balcony, hug…", lines=1)
                    map_dd = gr.Dropdown(label="Saved mapsets", choices=store.list_mapset_choices(), value=None)

                map_title = gr.Textbox(label="Mapset title", placeholder="Piggyback Pose A")
                map_tags = gr.Textbox(label="Tags (comma-separated)", placeholder="piggybackride, romantic, carry_pose")

                with gr.Row():
                    btn_map_new = gr.Button("➕ New mapset")
                    btn_map_save = gr.Button("💾 Save meta")
                    btn_map_del = gr.Button("🗑️ Delete", variant="stop")

                map_status = gr.Markdown("")
                with gr.Accordion("Preview maps", open=True):
                    with gr.Row():
                        show_canny = gr.Checkbox(value=True, label="Show Canny")
                        show_depth = gr.Checkbox(value=True, label="Show Depth")
                        show_pose  = gr.Checkbox(value=True, label="Show OpenPose")

                    canny_gallery = gr.Gallery(label="Canny", columns=4, rows=2, height="auto")
                    depth_gallery = gr.Gallery(label="Depth", columns=4, rows=2, height="auto")
                    pose_gallery  = gr.Gallery(label="OpenPose", columns=4, rows=2, height="auto")

                with gr.Accordion("Storage + Import", open=True):
                    gr.Markdown("Import maps into the selected mapset. **No Browse buttons** — paste paths or upload files.")
                    assets_root_tb = gr.Textbox(label="Mapsets root folder", value=str(ASSETS_DIR), interactive=False)
                    with gr.Row():
                        map_folder_tb = gr.Textbox(label="Selected mapset folder", value="", interactive=False)
                        btn_open_mapset_folder = gr.Button("📁 Open", scale=0)

                    with gr.Row():
                        btn_map_refresh = gr.Button("🔄 Refresh previews", scale=0)

                    with gr.Accordion("Batch upload (into selected mapset)", open=False):
                        enforce_suffix_up = gr.Checkbox(label="Enforce suffix (_canny/_depth/_openpose)", value=True)
                        with gr.Row():
                            up_canny = gr.File(label="Upload Canny map(s)", file_count="multiple", file_types=[".png",".jpg",".jpeg",".webp"])
                            btn_add_canny = gr.Button("➕ Add Canny", scale=0)
                        with gr.Row():
                            up_depth = gr.File(label="Upload Depth map(s)", file_count="multiple", file_types=[".png",".jpg",".jpeg",".webp"])
                            btn_add_depth = gr.Button("➕ Add Depth", scale=0)
                        with gr.Row():
                            up_pose = gr.File(label="Upload OpenPose map(s)", file_count="multiple", file_types=[".png",".jpg",".jpeg",".webp"])
                            btn_add_pose = gr.Button("➕ Add OpenPose", scale=0)

                    with gr.Accordion("Import from folder (paste path)", open=False):
                        folder_path = gr.Textbox(label="Folder path", placeholder=r"C:\maps\poseA  or  /workspace/maps/poseA", lines=1)
                        with gr.Row():
                            recursive = gr.Checkbox(label="Recursive", value=False)
                            enforce_suffix_imp = gr.Checkbox(label="Enforce suffix (_canny/_depth/_openpose)", value=True)
                        import_mode = gr.Dropdown(
                            label="How to classify files",
                            choices=["auto-detect by filename", "canny", "depth", "openpose"],
                            value="auto-detect by filename"
                        )
                        btn_import_folder = gr.Button("📂 Import ALL images from folder")

                def _refresh_mapsets(q: str):
                    s = VaultStore()
                    return gr.update(choices=s.list_mapset_choices(q))

                def _mapset_folder(mid: str) -> str:
                    if not mid:
                        return ""
                    try:
                        return str((ASSETS_DIR / mid).resolve())
                    except Exception:
                        return str(ASSETS_DIR / mid)

                def _select_mapset(mid: str):
                    s = VaultStore()
                    m = s.get_mapset(mid) if mid else None
                    if not m:
                        return "", "", "", "", [], [], [], "⚠️ Select a mapset."
                    return (
                        mid,
                        m.get("title",""),
                        ", ".join(m.get("tags") or []),
                        _mapset_folder(mid),
                        s.list_map_paths(mid, "canny"),
                        s.list_map_paths(mid, "depth"),
                        s.list_map_paths(mid, "openpose"),
                        "✅ Loaded."
                    )

                def _new_mapset(title: str, tags_csv: str):
                    s = VaultStore()
                    mid = s.create_mapset(title or "New Mapset", tags_csv or "")
                    m = s.get_mapset(mid) or {}
                    return (
                        gr.update(choices=s.list_mapset_choices(), value=mid),
                        mid,
                        m.get("title",""),
                        ", ".join(m.get("tags") or []),
                        _mapset_folder(mid),
                        [], [], [],
                        "✅ Created."
                    )

                def _save_mapset(mid: str, title: str, tags_csv: str):
                    if not mid:
                        return gr.update(choices=store.list_mapset_choices()), "⚠️ Select a mapset first."
                    s = VaultStore()
                    s.update_mapset_meta(mid, title, tags_csv)
                    return gr.update(choices=s.list_mapset_choices(), value=mid), "✅ Saved."

                def _del_mapset(mid: str):
                    if not mid:
                        return gr.update(choices=store.list_mapset_choices(), value=None), "", "", "", [], [], [], "⚠️ Nothing selected."
                    s = VaultStore()
                    s.delete_mapset(mid)
                    return gr.update(choices=s.list_mapset_choices(), value=None), "", "", "", [], [], [], "🗑️ Deleted."

                def _refresh_previews(mid: str):
                    if not mid:
                        return "", [], [], [], "⚠️ Select a mapset."
                    s = VaultStore()
                    return (
                        _mapset_folder(mid),
                        s.list_map_paths(mid, "canny"),
                        s.list_map_paths(mid, "depth"),
                        s.list_map_paths(mid, "openpose"),
                        "🔄 Refreshed."
                    )

                def _add_uploaded(mid: str, files, map_type: str, enforce_suffix: bool):
                    if not mid:
                        return "", [], [], [], "⚠️ Select a mapset."
                    s = VaultStore()
                    n = s.add_maps_to_mapset(mid, files or [], map_type=map_type, auto_detect=False, enforce_suffix=bool(enforce_suffix))
                    return (
                        _mapset_folder(mid),
                        s.list_map_paths(mid, "canny"),
                        s.list_map_paths(mid, "depth"),
                        s.list_map_paths(mid, "openpose"),
                        f"✅ Added {n} file(s) to **{map_type}**."
                    )

                def _import_from_folder(mid: str, folder: str, rec: bool, mode: str, enforce_suffix: bool):
                    if not mid:
                        return "", [], [], [], "⚠️ Select a mapset."
                    folder = (folder or "").strip().strip('"').strip("'")
                    if not folder or not os.path.isdir(folder):
                        # keep current preview, but show warning
                        p = _refresh_previews(mid)
                        return p[0], p[1], p[2], p[3], "⚠️ Folder not found."
                    exts = {".png",".jpg",".jpeg",".webp"}
                    paths = []
                    if rec:
                        for root, _, files in os.walk(folder):
                            for fn in files:
                                if os.path.splitext(fn)[1].lower() in exts:
                                    paths.append(os.path.join(root, fn))
                    else:
                        for fn in os.listdir(folder):
                            pth = os.path.join(folder, fn)
                            if os.path.isfile(pth) and os.path.splitext(fn)[1].lower() in exts:
                                paths.append(pth)
                    if not paths:
                        p = _refresh_previews(mid)
                        return p[0], p[1], p[2], p[3], "⚠️ No images found in that folder."
                    s = VaultStore()
                    auto_detect = (mode == "auto-detect by filename")
                    map_type = "canny" if auto_detect else mode
                    n = s.add_maps_to_mapset(mid, paths, map_type=map_type, auto_detect=auto_detect, enforce_suffix=bool(enforce_suffix))
                    return (
                        _mapset_folder(mid),
                        s.list_map_paths(mid, "canny"),
                        s.list_map_paths(mid, "depth"),
                        s.list_map_paths(mid, "openpose"),
                        f"✅ Imported {n} file(s) from folder."
                    )

                map_search.change(_refresh_mapsets, inputs=[map_search], outputs=[map_dd])
                map_dd.change(_select_mapset, inputs=[map_dd], outputs=[mapset_id, map_title, map_tags, map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])

                btn_map_new.click(_new_mapset, inputs=[map_title, map_tags], outputs=[map_dd, mapset_id, map_title, map_tags, map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])
                btn_map_save.click(_save_mapset, inputs=[mapset_id, map_title, map_tags], outputs=[map_dd, map_status])
                btn_map_del.click(_del_mapset, inputs=[mapset_id], outputs=[map_dd, mapset_id, map_title, map_tags, map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])

                btn_map_refresh.click(_refresh_previews, inputs=[mapset_id], outputs=[map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])

                btn_open_mapset_folder.click(_open_folder, inputs=[map_folder_tb], outputs=[map_status])

                btn_add_canny.click(_add_uploaded, inputs=[mapset_id, up_canny, gr.State("canny"), enforce_suffix_up], outputs=[map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])
                btn_add_depth.click(_add_uploaded, inputs=[mapset_id, up_depth, gr.State("depth"), enforce_suffix_up], outputs=[map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])
                btn_add_pose.click(_add_uploaded, inputs=[mapset_id, up_pose, gr.State("openpose"), enforce_suffix_up], outputs=[map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])

                btn_import_folder.click(_import_from_folder, inputs=[mapset_id, folder_path, recursive, import_mode, enforce_suffix_imp], outputs=[map_folder_tb, canny_gallery, depth_gallery, pose_gallery, map_status])

                show_canny.change(lambda v: gr.update(visible=v), inputs=[show_canny], outputs=[canny_gallery])
                show_depth.change(lambda v: gr.update(visible=v), inputs=[show_depth], outputs=[depth_gallery])
                show_pose.change(lambda v: gr.update(visible=v), inputs=[show_pose], outputs=[pose_gallery])

    return
