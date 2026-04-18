# SDXL Skin Enhancer Setup

This repo's `Radz SDXL Skin Realism` node is designed to sit after an SDXL or diffusion-based upscale stage.

## Recommended SDXL / Diffusion Skin-Realism Chain

Best practical chain for realistic portrait skin:

1. `SUPIR-v0Q.ckpt`
   Use as the diffusion restoration / enhancement stage before final skin cleanup.

2. `4xFFHQDAT.pth`
   Use as the face-focused upscale model when you need stronger facial structure and cleaner pores.

3. `1x-ITF-SkinDiffDetail-Lite-v1.pth`
   Use as the final skin-detail enhancer.

4. `Radz SDXL Skin Realism`
   Use after the upscale/refiner output for restrained, natural-looking skin cleanup.

## Why This Chain

- `SUPIR-v0Q` is commonly used in portrait restoration workflows for high-detail facial cleanup.
- `4xFFHQDAT` is face-trained and tends to preserve realistic facial texture better than generic sharpness-first upscalers.
- `1x-ITF-SkinDiffDetail-Lite-v1` is a lightweight skin-detail enhancer that works well as a final realism pass.
- `Radz SDXL Skin Realism` is tuned to avoid the crunchy or over-grained look that can happen after aggressive upscaling.

## ComfyUI Placement

```text
Load Image / SDXL Output
-> SUPIR / SDXL Upscale Stage
-> VAE Decode
-> 4xFFHQDAT or face-focused upscale stage
-> 1x-ITF-SkinDiffDetail-Lite-v1
-> Radz SDXL Skin Realism
-> Save Image
```

If your upstream SDXL workflow already did the heavy upscale, use:

```text
SDXL Upscale Output
-> VAE Decode
-> Radz SDXL Skin Realism
-> Save Image
```

## Recommended `Radz SDXL Skin Realism` Settings

- `upscale_model = 4xFFHQDAT.pth`
- `upscale_model_2 = 1xSkinContrast-High-SuperUltraCompact.pth`
- `upscale_model_3 = none`
- `skin_enhancer_model = x1_ITF_SkinDiffDetail_Lite_v1.pth`
- `skin_detail = 0.72`
- `eye_detail = 0.22`
- `baby_hair = 0.08`
- `film_grain = 0.00`
- `naturalness = 0.88`
- `strength = 0.52`
- `upscale_factor = 1x`
- `mode = detail_only`

## Notes

- If the result looks too polished, lower `strength` first.
- If the result lacks pores, raise `skin_detail` slightly.
- If the result looks fake, raise `naturalness` and keep `film_grain` at `0.00`.
