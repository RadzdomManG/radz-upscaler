# ComfyUI node for Radz Ultimate Upscaler.

import logging
from contextlib import contextmanager

import comfy
import modules.shared as shared
import numpy as np
import torch
from PIL import Image, ImageFilter
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from modules.processing import StableDiffusionProcessing
from modules.upscaler import UpscalerData

from usdu_patch import usdu
from usdu_utils import pil_to_tensor, tensor_to_pil

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    root_logger = logging.getLogger()
    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)


MAX_RESOLUTION = 8192
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}
PRESETS = {
    "SDXL Balanced Preserve": {
        "steps": 24,
        "cfg": 5.5,
        "denoise": 0.12,
        "tile_width": 1024,
        "tile_height": 1024,
        "mask_blur": 6,
        "tile_padding": 64,
        "seam_fix_mode": "Band Pass",
        "seam_fix_denoise": 0.2,
        "seam_fix_width": 96,
        "seam_fix_mask_blur": 6,
        "seam_fix_padding": 32,
        "force_uniform_tiles": True,
        "tiled_decode": True,
        "batch_size": 1,
        "similarity_strength": 0.30,
        "color_preservation": 0.20,
    },
    "SDXL Face Protect": {
        "steps": 22,
        "cfg": 5.0,
        "denoise": 0.10,
        "tile_width": 1024,
        "tile_height": 1024,
        "mask_blur": 5,
        "tile_padding": 72,
        "seam_fix_mode": "Band Pass",
        "seam_fix_denoise": 0.15,
        "seam_fix_width": 96,
        "seam_fix_mask_blur": 5,
        "seam_fix_padding": 40,
        "force_uniform_tiles": True,
        "tiled_decode": True,
        "batch_size": 1,
        "similarity_strength": 0.38,
        "color_preservation": 0.25,
    },
    "SDXL High Detail": {
        "steps": 28,
        "cfg": 6.0,
        "denoise": 0.16,
        "tile_width": 1024,
        "tile_height": 1024,
        "mask_blur": 8,
        "tile_padding": 64,
        "seam_fix_mode": "Half Tile + Intersections",
        "seam_fix_denoise": 0.25,
        "seam_fix_width": 96,
        "seam_fix_mask_blur": 8,
        "seam_fix_padding": 32,
        "force_uniform_tiles": True,
        "tiled_decode": True,
        "batch_size": 1,
        "similarity_strength": 0.18,
        "color_preservation": 0.12,
    },
    "Custom": {},
}
BASE_DEFAULTS = {
    "steps": 24,
    "cfg": 5.5,
    "denoise": 0.12,
    "tile_width": 1024,
    "tile_height": 1024,
    "mask_blur": 6,
    "tile_padding": 64,
    "seam_fix_mode": "Band Pass",
    "seam_fix_denoise": 0.2,
    "seam_fix_width": 96,
    "seam_fix_mask_blur": 6,
    "seam_fix_padding": 32,
    "force_uniform_tiles": True,
    "tiled_decode": True,
    "batch_size": 1,
    "similarity_strength": 0.30,
    "color_preservation": 0.20,
}


def usdu_base_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
        ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
        ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2.0, "min": 0.05, "max": 4.0, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "The seed to use for image-to-image."})),
        ("steps", ("INT", {"default": 24, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of sampling steps for each tile."})),
        ("cfg", ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
        ("denoise", ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The primary upscaler model for the base upscale stage."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 6, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 96, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 6, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the configured tile size so tile batches stay consistent."})),
        ("tiled_decode", ("BOOLEAN", {"default": True, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in one batch."})),
    ]
    optional = [
        ("preset", (list(PRESETS.keys()), {"default": "SDXL Balanced Preserve", "tooltip": "Applies SDXL-first defaults unless set to Custom."})),
        ("preserve_mask", ("MASK", {"tooltip": "Optional mask for regions that must stay closest to the source image, such as a face."})),
        ("preserve_mask_strength", ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strongly to keep preserve_mask regions close to the source."})),
        ("preserve_mask_blur", ("INT", {"default": 12, "min": 0, "max": 128, "step": 1, "tooltip": "Blur applied to the preserve mask before blending."})),
        ("similarity_strength", ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Global blend back toward the source to reduce identity drift."})),
        ("color_preservation", ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Preserves the original color and lighting palette."})),
        ("detail_boost", ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Final detail/sharpness boost after upscaling."})),
        ("detail_boost_radius", ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Unsharp mask radius for detail boost."})),
        ("post_detail_model", ("UPSCALE_MODEL", {"tooltip": "Optional second upscale model used as a detail polish stage."})),
        ("post_detail_strength", ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend strength for the second model polish pass."})),
    ]
    return required, optional


def prepare_inputs(required, optional=None):
    inputs = {}
    if required:
        inputs["required"] = {name: node_type for name, node_type in required}
    if optional:
        inputs["optional"] = {name: node_type for name, node_type in optional}
    return inputs


def remove_input(inputs, input_name):
    for index, (name, _) in enumerate(inputs):
        if name == input_name:
            del inputs[index]
            break


def rename_input(inputs, old_name, new_name):
    for index, (name, node_type) in enumerate(inputs):
        if name == old_name:
            inputs[index] = (new_name, node_type)
            break


def apply_preset(values, preset_name):
    if preset_name in PRESETS:
        for key, preset_value in PRESETS[preset_name].items():
            base_value = BASE_DEFAULTS.get(key)
            current_value = values.get(key)
            # Presets act like default-fill behavior. If the user changed a widget
            # away from the base default, keep the user's value.
            if current_value == base_value:
                values[key] = preset_value
    return values


def resize_tensor_image(image_tensor, width, height):
    nchw = image_tensor.movedim(-1, 1)
    resized = torch.nn.functional.interpolate(
        nchw,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def blur_mask(mask_tensor, radius):
    if mask_tensor is None:
        return None
    safe_mask = mask_tensor.float().clamp(0.0, 1.0)
    if safe_mask.ndim == 2:
        safe_mask = safe_mask.unsqueeze(0)
    pil_mask = Image.fromarray((safe_mask[0].cpu().numpy() * 255.0).astype(np.uint8), mode="L")
    if radius > 0:
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius))
    blurred = torch.from_numpy(np.array(pil_mask).astype(np.float32) / 255.0).unsqueeze(0)
    return blurred.clamp(0.0, 1.0)


def blend_with_mask(base_tensor, overlay_tensor, mask_tensor, strength):
    if mask_tensor is None or strength <= 0.0:
        return base_tensor
    mask = mask_tensor
    if mask.shape[-2:] != base_tensor.shape[1:3]:
        mask = resize_tensor_image(mask.unsqueeze(-1), base_tensor.shape[2], base_tensor.shape[1]).squeeze(-1)
    mask = mask.unsqueeze(-1).to(base_tensor.device, dtype=base_tensor.dtype)
    mask = (mask * strength).clamp(0.0, 1.0)
    return (base_tensor * (1.0 - mask) + overlay_tensor * mask).clamp(0.0, 1.0)


def apply_global_similarity(result_tensor, source_tensor, strength):
    if strength <= 0.0:
        return result_tensor
    return torch.lerp(result_tensor, source_tensor, float(strength)).clamp(0.0, 1.0)


def apply_detail_boost(image_tensor, amount, radius):
    if amount <= 0.0:
        return image_tensor
    boosted = []
    percent = int(100 + amount * 200)
    threshold = max(1, int(6 - amount * 5))
    for batch_index in range(len(image_tensor)):
        image = tensor_to_pil(image_tensor, batch_index)
        sharpened = image.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
        boosted.append(pil_to_tensor(sharpened))
    return torch.cat(boosted, dim=0).clamp(0.0, 1.0)


def apply_post_detail_model(image_tensor, detail_model, blend_strength):
    if detail_model is None or blend_strength <= 0.0:
        return image_tensor
    if "execute" in dir(ImageUpscaleWithModel):
        (polished,) = ImageUpscaleWithModel.execute(detail_model, image_tensor)
    else:
        (polished,) = ImageUpscaleWithModel().upscale(detail_model, image_tensor)
    polished = resize_tensor_image(polished, image_tensor.shape[2], image_tensor.shape[1])
    return torch.lerp(image_tensor, polished, float(blend_strength)).clamp(0.0, 1.0)


class RadzUltimateUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        required, optional = usdu_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = (
        "Ultimate SD Upscale with SDXL-first defaults, source preservation controls, "
        "mask-based face protection, and an optional second detail model polish pass."
    )

    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_model,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        batch_size=1,
        preset="SDXL Balanced Preserve",
        preserve_mask=None,
        preserve_mask_strength=0.65,
        preserve_mask_blur=12,
        similarity_strength=0.30,
        color_preservation=0.20,
        detail_boost=0.10,
        detail_boost_radius=2,
        post_detail_model=None,
        post_detail_strength=0.20,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        values = {
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "mask_blur": mask_blur,
            "tile_padding": tile_padding,
            "seam_fix_mode": seam_fix_mode,
            "seam_fix_denoise": seam_fix_denoise,
            "seam_fix_width": seam_fix_width,
            "seam_fix_mask_blur": seam_fix_mask_blur,
            "seam_fix_padding": seam_fix_padding,
            "force_uniform_tiles": force_uniform_tiles,
            "tiled_decode": tiled_decode,
            "batch_size": batch_size,
            "similarity_strength": similarity_strength,
            "color_preservation": color_preservation,
        }
        apply_preset(values, preset)

        redraw_mode = MODES[mode_type]
        seam_fix_value = SEAM_FIX_MODES[values["seam_fix_mode"]]

        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        assert values["batch_size"] == 1 or values["force_uniform_tiles"], (
            "batch_size greater than 1 requires force_uniform_tiles to be True; "
            "all tiles in a batch must share the same size."
        )

        sdprocessing = StableDiffusionProcessing(
            shared.batch[0],
            model,
            positive,
            negative,
            vae,
            seed,
            values["steps"],
            values["cfg"],
            sampler_name,
            scheduler,
            values["denoise"],
            upscale_by,
            values["force_uniform_tiles"],
            values["tiled_decode"],
            values["tile_width"],
            values["tile_height"],
            redraw_mode,
            seam_fix_value,
            custom_sampler,
            custom_sigmas,
            values["batch_size"],
        )

        with suppress_logging():
            script = usdu.Script()
            script.run(
                p=sdprocessing,
                _=None,
                tile_width=values["tile_width"],
                tile_height=values["tile_height"],
                mask_blur=values["mask_blur"],
                padding=values["tile_padding"],
                seams_fix_width=values["seam_fix_width"],
                seams_fix_denoise=values["seam_fix_denoise"],
                seams_fix_padding=values["seam_fix_padding"],
                upscaler_index=0,
                save_upscaled_image=False,
                redraw_mode=redraw_mode,
                save_seams_fix_image=False,
                seams_fix_mask_blur=values["seam_fix_mask_blur"],
                seams_fix_type=seam_fix_value,
                target_size_type=2,
                custom_width=None,
                custom_height=None,
                custom_scale=upscale_by,
            )

        result_tensor = torch.cat([pil_to_tensor(img) for img in shared.batch], dim=0)
        source_upscaled = resize_tensor_image(image, result_tensor.shape[2], result_tensor.shape[1])

        result_tensor = apply_global_similarity(result_tensor, source_upscaled, values["similarity_strength"])
        result_tensor = apply_global_similarity(result_tensor, source_upscaled, values["color_preservation"] * 0.5)

        blurred_preserve_mask = blur_mask(preserve_mask, preserve_mask_blur)
        result_tensor = blend_with_mask(
            result_tensor,
            source_upscaled,
            blurred_preserve_mask,
            preserve_mask_strength,
        )
        result_tensor = apply_post_detail_model(result_tensor, post_detail_model, post_detail_strength)
        result_tensor = apply_detail_boost(result_tensor, detail_boost, detail_boost_radius)
        return (result_tensor.clamp(0.0, 1.0),)


class RadzUltimateUpscalerNoUpscale(RadzUltimateUpscaler):
    @classmethod
    def INPUT_TYPES(cls):
        required, optional = usdu_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final refined image.",)
    DESCRIPTION = "Refines an already upscaled image tile-by-tile without another base upscale pass."

    def upscale(
        self,
        upscaled_image,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        batch_size=1,
        preset="SDXL Balanced Preserve",
        preserve_mask=None,
        preserve_mask_strength=0.65,
        preserve_mask_blur=12,
        similarity_strength=0.30,
        color_preservation=0.20,
        detail_boost=0.10,
        detail_boost_radius=2,
        post_detail_model=None,
        post_detail_strength=0.20,
    ):
        return super().upscale(
            upscaled_image,
            model,
            positive,
            negative,
            vae,
            1.0,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            None,
            mode_type,
            tile_width,
            tile_height,
            mask_blur,
            tile_padding,
            seam_fix_mode,
            seam_fix_denoise,
            seam_fix_mask_blur,
            seam_fix_width,
            seam_fix_padding,
            force_uniform_tiles,
            tiled_decode,
            batch_size,
            preset,
            preserve_mask,
            preserve_mask_strength,
            preserve_mask_blur,
            similarity_strength,
            color_preservation,
            detail_boost,
            detail_boost_radius,
            post_detail_model,
            post_detail_strength,
        )


class RadzUltimateUpscalerCustomSample(RadzUltimateUpscaler):
    @classmethod
    def INPUT_TYPES(cls):
        required, optional = usdu_base_inputs()
        remove_input(required, "upscale_model")
        optional.append(
            (
                "upscale_model",
                (
                    "UPSCALE_MODEL",
                    {
                        "tooltip": "Optional primary upscaler model. If omitted, the node falls back to Lanczos scaling."
                    },
                ),
            )
        )
        optional.append(
            (
                "custom_sampler",
                (
                    "SAMPLER",
                    {
                        "tooltip": "Optional custom sampler. Used only when custom sigmas are also provided."
                    },
                ),
            )
        )
        optional.append(
            (
                "custom_sigmas",
                (
                    "SIGMAS",
                    {
                        "tooltip": "Optional custom sigma schedule. Used only when a custom sampler is also provided."
                    },
                ),
            )
        )
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Runs the Radz upscaler with optional custom sampler and sigma inputs."

    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        batch_size=1,
        preset="SDXL Balanced Preserve",
        preserve_mask=None,
        preserve_mask_strength=0.65,
        preserve_mask_blur=12,
        similarity_strength=0.30,
        color_preservation=0.20,
        detail_boost=0.10,
        detail_boost_radius=2,
        post_detail_model=None,
        post_detail_strength=0.20,
        upscale_model=None,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        return super().upscale(
            image,
            model,
            positive,
            negative,
            vae,
            upscale_by,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            upscale_model,
            mode_type,
            tile_width,
            tile_height,
            mask_blur,
            tile_padding,
            seam_fix_mode,
            seam_fix_denoise,
            seam_fix_mask_blur,
            seam_fix_width,
            seam_fix_padding,
            force_uniform_tiles,
            tiled_decode,
            batch_size,
            preset,
            preserve_mask,
            preserve_mask_strength,
            preserve_mask_blur,
            similarity_strength,
            color_preservation,
            detail_boost,
            detail_boost_radius,
            post_detail_model,
            post_detail_strength,
            custom_sampler,
            custom_sigmas,
        )


NODE_CLASS_MAPPINGS = {
    "RadzUltimateUpscaler": RadzUltimateUpscaler,
    "RadzUltimateUpscalerNoUpscale": RadzUltimateUpscalerNoUpscale,
    "RadzUltimateUpscalerCustomSample": RadzUltimateUpscalerCustomSample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadzUltimateUpscaler": "Radz Ultimate Upscaler",
    "RadzUltimateUpscalerNoUpscale": "Radz Ultimate Upscaler (No Upscale)",
    "RadzUltimateUpscalerCustomSample": "Radz Ultimate Upscaler (Custom Sample)",
}
