# Radz Nodes on Runpod `runpod/comfyui:cuda13.0`

This bundle is prepared for a fresh official Runpod ComfyUI pod using the CUDA 13 line.

Verified alignment targets from current docs:

- Runpod official ComfyUI pods/templates
- Python 3.13
- PyTorch CUDA 13.x
- modern ComfyUI install layout

References:

- [Runpod ComfyUI docs](https://docs.runpod.io/tutorials/pods/comfyui)
- [ComfyUI system requirements](https://docs.comfy.org/installation/system_requirements/)
- [ComfyUI admin guide example install](https://doccompiler.ai/api/v1/jobs/shared/job_1776340060827_2f9250e5/download/Comfy-Org__ComfyUI__AdminGuide.pdf)

## What this setup assumes

- Python `3.13`
- Torch / torchvision / torchaudio already installed by the Runpod image with CUDA 13 support
- ComfyUI already present in the container

## What this bundle adds

- `Radz Ultimate Upscaler`
- `Radz Insight Face ...` IPAdapter nodes
- `Radz Real Human Detail`
- bundled model directory layout under `Radz Nodes/bundled_models`
- optional FaceID / InsightFace dependency install path

## Recommended install flow on Runpod

1. Put `Radz Nodes` in `ComfyUI/custom_nodes/`
2. Run `install_runpod_cuda13.sh`
3. Restart ComfyUI
4. Add model files into `Radz Nodes/bundled_models`

## Important note about FaceID

The non-FaceID IPAdapter nodes should work as long as your ComfyUI, Torch, and models are correct.

The FaceID / InsightFace nodes also require:

- `insightface`
- `onnxruntime-gpu`
- InsightFace model assets inside `bundled_models/insightface`

## Model location strategy

This bundle is designed so the following can live inside one package folder:

- `bundled_models/clip_vision`
- `bundled_models/ipadapter`
- `bundled_models/loras`
- `bundled_models/insightface`

The installer writes `extra_model_paths.yaml` so ComfyUI can find the first three.
