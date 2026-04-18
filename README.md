# Radz Human Skin Details

`Radz Human Skin Details` is a ComfyUI custom node for realistic human portrait enhancement.

It is designed to sit after `VAE Decode` and gently improve:

- skin realism
- human micro-texture
- subtle eye clarity
- restrained baby hair / peach fuzz feeling
- soft cinematic film grain

The visual target is premium portrait realism:

- believable skin
- natural detail
- preserved identity
- tasteful photographic finish

It intentionally avoids:

- crunchy oversharpening
- plastic AI skin
- fake pores
- ugly digital noise
- bright eye halos

## Current Version

This first version is fully working and uses a handcrafted placeholder enhancement pipeline.

It does **not** require a trained realism model yet. The architecture is already prepared for later custom-model integration through the model hook inside:

- `real_human_detail_node.py`

Look for:

- `_run_custom_model_if_available(...)`
- `run_model_or_placeholder(...)`
- `_placeholder_skin_enhance(...)`
- `_placeholder_eye_enhance(...)`
- `_placeholder_baby_hair_detail(...)`
- `_add_film_grain(...)`

## Folder Structure

Place the package here:

```text
ComfyUI/custom_nodes/radz_human_skin_details/
```

Files:

```text
custom_nodes/radz_human_skin_details/__init__.py
custom_nodes/radz_human_skin_details/real_human_detail_node.py
custom_nodes/radz_human_skin_details/requirements.txt
custom_nodes/radz_human_skin_details/README.md
```

## Installation

1. Copy the folder `radz_human_skin_details` into:

   ```text
   ComfyUI/custom_nodes/
   ```

2. Install dependencies if needed.

   For the Windows portable build, from the portable root:

   ```powershell
   python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\radz_human_skin_details\requirements.txt
   ```

3. Restart ComfyUI.

4. Search for this node in ComfyUI:

   ```text
   Radz Human Skin Details
   ```

Category:

```text
Radz/Skin
```

## Recommended Workflow Placement

Typical connection:

```text
Checkpoint Loader
-> KSampler
-> VAE Decode
-> Radz Human Skin Details
-> Save Image
```

This node expects an `IMAGE` input and outputs one `IMAGE`.

## Inputs

### Required

- `image`
  ComfyUI `IMAGE` tensor, usually connected after `VAE Decode`.

- `skin_detail`
  Controls gentle skin micro-contrast and photographic texture recovery.

- `eye_detail`
  Adds restrained crispness in high-detail eye regions without glowing or haloing.

- `baby_hair`
  Enhances extremely fine facial-edge texture for a soft baby-hair / peach-fuzz feeling.
  It does not draw fake hairs.

- `film_grain`
  Adds subtle luminance-aware monochromatic grain for a more expensive camera-like finish.

- `naturalness`
  Protection slider. Higher values keep the image more believable and reduce artificial-looking processing.

- `strength`
  Overall blend amount of the enhancement.

- `mode`
  Options:
  - `detail_only`
  - `both`

  `detail_only` skips film grain.
  `both` runs detail enhancement plus grain.

### Optional

- `enable_tiling`
  Enables tiled processing for larger images.

- `tile_size`
  Tile width and height used when tiling is enabled.

- `tile_overlap`
  Feathered overlap between tiles to avoid seams.

## Suggested Starting Settings

For realistic portraits:

- `upscale_model = RealESRGAN_x2.pth`
- `upscale_model_2 = none`
- `upscale_model_3 = none`
- `skin_enhancer_model = x1_ITF_SkinDiffDetail_Lite_v1.pth`
- `skin_detail = 0.68`
- `eye_detail = 0.24`
- `baby_hair = 0.10`
- `film_grain = 0.01`
- `naturalness = 0.84`
- `strength = 0.58`
- `mode = detail_only`

For softer premium realism:

- lower `strength`
- lower `skin_detail`
- keep `naturalness` high

Best skin-focused upscaler pairing:

- `RealESRGAN_x2.pth` as the main upscaler
- `x1_ITF_SkinDiffDetail_Lite_v1.pth` as the skin enhancer

This pairing is intentionally more natural than stacking multiple aggressive upscalers.

For a slightly richer photographic finish:

- set `mode = both`
- keep `film_grain` very low, usually `0.01` to `0.03`
- keep `naturalness` at `0.80` to `0.92`

## Implementation Notes

The placeholder pipeline is intentionally conservative and built around:

1. preprocessing and validation
2. optional tile split
3. placeholder realism enhancement
4. eye-detail refinement
5. baby-hair / micro-texture refinement
6. film grain
7. naturalness-aware blending with the original
8. tile merge
9. safe clamp back to `0..1`

The skin pass is the main focus. It uses gentle micro-contrast recovery and edge-aware masking so the result aims for:

- believable skin texture
- subtle tonal transitions
- improved human detail

without:

- crunchy pores
- halos
- rough sharpened skin
- fake detail

## Future Model Integration

You can add a trained model later by placing model files inside:

```text
ComfyUI/custom_nodes/radz_human_skin_details/models/
```

Then wire your model into:

- `_run_custom_model_if_available(...)`

The node is already structured so the placeholder pipeline remains the fallback when no model is present.

## Dependencies

- `torch`
- `numpy`

Most ComfyUI installs already include PyTorch. `numpy` is also typically available. The included `requirements.txt` is mainly there to make the package self-contained and explicit.

## Limits Of The Placeholder Version

This first version is a premium placeholder, not a learned portrait-restoration model.

That means:

- it enhances existing detail rather than hallucinating truly new biological detail
- it cannot detect facial anatomy semantically like a dedicated trained model
- eye targeting is heuristic, not landmark-based
- baby hair is interpreted as restrained micro-edge refinement, not literal hair synthesis

Even with those limits, it is designed to be useful now and easy to upgrade later.
