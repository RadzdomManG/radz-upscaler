# Radz Nodes

Bundled ComfyUI custom nodes for:

- `Radz Ultimate Upscaler`
- `Radz Insight Face ...` nodes based on IPAdapter Plus
- `Radz Real Human Detail`

This package is designed to live as a single folder inside `ComfyUI/custom_nodes/`.

## Included Modules

### 1. Radz Ultimate Upscaler

Ultimate SD Upscale style tiled upscaling with SDXL-first defaults, source-preservation controls, optional preserve masks, optional second detail model pass, and realism finishing controls.

### 2. Radz Insight Face

Bundled IPAdapter Plus code re-exported under `Radz Insight Face ...` node names so the nodes appear under one Radz package instead of as a separate custom node repo.

### 3. Radz Real Human Detail

Skin, eye, baby hair, and realism enhancement node bundled from your existing Radz human detailer setup.

## Installation

Place this folder into:

```text
ComfyUI/custom_nodes/Radz Nodes
```

Then restart ComfyUI.

For a fresh Runpod CUDA 13 pod, use:

```text
Radz Nodes/install_runpod_cuda13.sh
```

Additional Runpod notes are in:

```text
Radz Nodes/RUNPOD.md
```

## What Is Already Bundled

The code for these node packs is already included inside this single folder:

- Ultimate SD Upscale integration
- IPAdapter Plus integration
- Radz Real Human Detail

There is also a bundled single-place model layout inside:

```text
Radz Nodes/bundled_models
```

and a helper setup script:

```text
Radz Nodes/setup_radz_nodes.ps1
```

That means you do not need to separately clone those repositories just to get the node code itself.

## What Is Not Auto-Downloaded

Model weights are not bundled in this folder. They still need to exist in your ComfyUI models directories.

This is especially important for IPAdapter / InsightFace workflows.

## IPAdapter Model Layout

Recommended directories:

```text
Radz Nodes/bundled_models/clip_vision
Radz Nodes/bundled_models/ipadapter
Radz Nodes/bundled_models/loras
Radz Nodes/bundled_models/insightface
```

For the Unified Loader, filenames must match the expected names exactly.

`clip_vision`, `ipadapter`, and `loras` can be exposed to ComfyUI through `extra_model_paths.yaml`.

`insightface` is checked directly from `Radz Nodes/bundled_models/insightface` first by the bundled Radz Insight Face loader.

### `ComfyUI/models/clip_vision`

- `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
- `CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors`
- `clip-vit-large-patch14-336.bin` for Kolors only

### `ComfyUI/models/ipadapter`

- `ip-adapter_sd15.safetensors`
- `ip-adapter_sd15_light_v11.bin`
- `ip-adapter-plus_sd15.safetensors`
- `ip-adapter-plus-face_sd15.safetensors`
- `ip-adapter-full-face_sd15.safetensors`
- `ip-adapter_sd15_vit-G.safetensors`
- `ip-adapter_sdxl_vit-h.safetensors`
- `ip-adapter-plus_sdxl_vit-h.safetensors`
- `ip-adapter-plus-face_sdxl_vit-h.safetensors`
- `ip-adapter_sdxl.safetensors`
- `ip-adapter-faceid_sd15.bin`
- `ip-adapter-faceid-plusv2_sd15.bin`
- `ip-adapter-faceid-portrait-v11_sd15.bin`
- `ip-adapter-faceid_sdxl.bin`
- `ip-adapter-faceid-plusv2_sdxl.bin`
- `ip-adapter-faceid-portrait_sdxl.bin`
- `ip-adapter-faceid-portrait_sdxl_unnorm.bin`
- `ip_plus_composition_sd15.safetensors`
- `ip_plus_composition_sdxl.safetensors`
- `Kolors-IP-Adapter-Plus.bin`
- `Kolors-IP-Adapter-FaceID-Plus.bin`

### `ComfyUI/models/loras`

- `ip-adapter-faceid_sd15_lora.safetensors`
- `ip-adapter-faceid-plusv2_sd15_lora.safetensors`
- `ip-adapter-faceid_sdxl_lora.safetensors`
- `ip-adapter-faceid-plusv2_sdxl_lora.safetensors`

## InsightFace Note

FaceID workflows require `insightface` in your ComfyUI Python environment and the correct FaceID models/LoRAs. The node code can be bundled here, but those runtime assets are still external requirements.

## Notes

- IPAdapter works best on a current ComfyUI version.
- If a unified loader preset does not work, the most common causes are missing models, incorrect filenames, or missing InsightFace dependencies.
- Your manual settings in `Radz Ultimate Upscaler` are preserved even when using presets.
