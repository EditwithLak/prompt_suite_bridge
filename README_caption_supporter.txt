Caption Supporter (Forge extension)

This extension adds a "Caption Supporter" tab in Forge Neo.

Backend implemented:
- JoyCaption GGUF (llama.cpp via llama-cpp-python)

Recommended: run the captioner in a separate Python venv, then point the tab to that python.exe.

Install llama-cpp-python (CPU):
  pip install llama-cpp-python

CUDA build (source build):
  set CMAKE_ARGS=-DGGML_CUDA=on
  pip install llama-cpp-python

See llama-cpp-python docs for backends & multi-modal handlers (Llava15ChatHandler).
