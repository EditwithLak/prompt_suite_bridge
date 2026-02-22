
import os
import sys
import gradio as gr
from modules import script_callbacks

# Add extension lib/ to path (so we can keep big modules out of scripts/)
EXT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(EXT_ROOT, "lib")
if LIB_DIR not in sys.path:
    sys.path.append(LIB_DIR)

# Add scripts/ to path so sibling modules can be imported reliably in Forge
SCRIPTS_DIR = os.path.dirname(__file__)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from prompt_builder_embed import build_prompt_builder_ui
from vault_embed import build_vault_ui
from caption_supporter_embed import build_caption_supporter_ui
from map_generator_embed import build_map_generator_ui

from prompt_suite_injector import queue_cn, clear_queue, cn_ext


EXT_NAME = "Prompt Suite (Vault-first) v0.4.20"
EXT_SLUG = "prompt_suite"


def _queue_ui(kind: str, unit_idx: int, path: str, weight: float = 1.0) -> str:
    path = (path or "").strip()
    if not path:
        return f"⚠️ No {kind} selected."
    queue_cn(unit_idx, kind, path, weight)
    if cn_ext is None:
        return f"✅ Queued {kind} → CN Unit {unit_idx}. (ControlNet bridge not found)"
    return f"✅ Queued {kind} → CN Unit {unit_idx}. Now hit **Generate**."


def _clear_queue_ui() -> str:
    clear_queue()
    return "🧹 Cleared queued maps."


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("## 🧩 Prompt Suite (Vault-first) v0.4.20")
        gr.Markdown(
            "- **Prompt Builder** = main workstation (compose prompts, load vault assets, send to txt2img/img2img, queue ControlNet)\n"
            "- **Vault + Maps** = asset management only (tags/packs/mapsets CRUD)\n"
            "- **Caption Tools** = utilities (optional, can run via separate python.exe)\n"
        )

        with gr.Tabs():
            with gr.Tab("🧱 Prompt Composer"):
                # Builds full builder UI and returns the shared editor fields (pos/neg)
                build_prompt_builder_ui(
                    queue_cb=_queue_ui,
                    clear_queue_cb=_clear_queue_ui,
                )

            with gr.Tab("🗃️ Vault + Maps"):
                build_vault_ui()

            with gr.Tab("🧪 Map Generator"):
                build_map_generator_ui(queue_cb=_queue_ui, clear_queue_cb=_clear_queue_ui)

            with gr.Tab("🏷️ Caption Tools"):
                build_caption_supporter_ui()

    return [(ui, EXT_NAME, EXT_SLUG)]


script_callbacks.on_ui_tabs(on_ui_tabs)
