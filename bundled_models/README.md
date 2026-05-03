# Radz Human Skin details Bundled Models

This folder is the single-place model home for the `Radz Nodes` bundle.

Subfolders:

- `clip_vision`
- `ipadapter`
- `loras`
- `insightface`

ComfyUI can be pointed here with `extra_model_paths.yaml`.

Notes:

- `clip_vision`, `ipadapter`, and `loras` can be loaded through ComfyUI extra model paths.
- `insightface` is used directly by the bundled Radz Insight Face loader and is checked here first.
- The node code is bundled already, but large model weights are still separate assets and are not embedded in source control here.
