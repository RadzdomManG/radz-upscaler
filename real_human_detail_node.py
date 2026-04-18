from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import comfy.utils
import folder_paths
from comfy import model_management
from comfy_extras.nodes_upscale_model import UpscaleModelLoader


@dataclass
class TileInfo:
    y0: int
    y1: int
    x0: int
    x1: int


class RealHumanDetailEngine:
    """
    Placeholder realism engine with a clean hook for a future trained model.

    IMAGE tensors are expected in ComfyUI layout: (B, H, W, C), float32, 0..1.
    """

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or (Path(__file__).resolve().parent / "models")
        self._cached_model = None
        self._cached_model_path = None
        self._cached_upscale_model = None
        self._cached_upscale_name = None

    def enhance(
        self,
        image: torch.Tensor,
        upscale_model_name: str,
        upscale_model_name_2: str,
        upscale_model_name_3: str,
        skin_enhancer_model_name: str,
        skin_detail: float,
        eye_detail: float,
        baby_hair: float,
        film_grain: float,
        naturalness: float,
        strength: float,
        mode: str,
        upscale_factor: float = 1.0,
    ) -> torch.Tensor:
        original = image
        current = image
        for model_name in (upscale_model_name, upscale_model_name_2, upscale_model_name_3):
            if model_name != "none":
                current = self._run_upscale_model(current, model_name)
                original = current

        if skin_enhancer_model_name != "none":
            current = self._run_upscale_model(current, skin_enhancer_model_name)
            original = current

        enhanced = self.run_model_or_placeholder(
            image=current,
            skin_detail=skin_detail,
            eye_detail=eye_detail,
            baby_hair=baby_hair,
            film_grain=film_grain,
            naturalness=naturalness,
            strength=strength,
            mode=mode,
        )

        # When the user asks for very high skin detail, push a second detail pass so
        # the result changes visibly instead of staying too close to the source.
        detail_boost = torch.clamp(
            torch.tensor((skin_detail - 0.9) / 0.1, device=current.device, dtype=current.dtype),
            0.0,
            1.0,
        )
        if float(detail_boost.max()) > 0.0:
            boosted = self._placeholder_skin_enhance(
                enhanced,
                min(1.0, skin_detail + 0.08),
                max(0.01, naturalness * 0.9),
            )
            boosted = self._synthetic_skin_texture(
                boosted,
                current,
                min(1.0, skin_detail + 0.1),
                max(0.01, naturalness * 0.92),
            )
            enhanced = torch.lerp(enhanced, boosted, detail_boost * 0.35)

        protected_strength = strength * (0.86 + 0.08 * (1.0 - naturalness))
        blended = torch.lerp(original, enhanced, protected_strength)
        identity_guard = 0.26 + 0.28 * naturalness
        output = torch.lerp(blended, original * 0.12 + blended * 0.88, identity_guard)
        if upscale_factor > 1.0:
            output = self._upscale_and_refine(output, upscale_factor, skin_detail, naturalness)
        return output.clamp(0.0, 1.0)

    def _run_upscale_model(self, image: torch.Tensor, upscale_model_name: str) -> torch.Tensor:
        upscale_model = self._load_upscale_model(upscale_model_name)
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 768
        overlap = 32
        try:
            while True:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                        in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                    )
                    pbar = comfy.utils.ProgressBar(steps)
                    out = comfy.utils.tiled_scale(
                        in_img,
                        lambda a: upscale_model(a),
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                        upscale_amount=upscale_model.scale,
                        pbar=pbar,
                    )
                    break
                except Exception as e:
                    model_management.raise_non_oom(e)
                    tile //= 2
                    if tile < 128:
                        raise
        finally:
            upscale_model.to("cpu")

        return out.movedim(-3, -1).clamp(0.0, 1.0)

    def _load_upscale_model(self, upscale_model_name: str):
        if self._cached_upscale_name != upscale_model_name or self._cached_upscale_model is None:
            self._cached_upscale_model = UpscaleModelLoader.load_model(upscale_model_name)[0]
            self._cached_upscale_name = upscale_model_name
        return self._cached_upscale_model

    def run_model_or_placeholder(
        self,
        image: torch.Tensor,
        skin_detail: float,
        eye_detail: float,
        baby_hair: float,
        film_grain: float,
        naturalness: float,
        strength: float,
        mode: str,
    ) -> torch.Tensor:
        model_output = self._run_custom_model_if_available(
            image=image,
            skin_detail=skin_detail,
            eye_detail=eye_detail,
            baby_hair=baby_hair,
            film_grain=film_grain,
            naturalness=naturalness,
            strength=strength,
            mode=mode,
        )
        if model_output is not None:
            return model_output

        base = image.clone()
        realism = self._placeholder_skin_enhance(base, skin_detail, naturalness)
        realism = self._synthetic_skin_texture(realism, base, skin_detail, naturalness)
        realism = self._placeholder_eye_enhance(realism, base, eye_detail, naturalness)
        realism = self._placeholder_baby_hair_detail(realism, base, baby_hair, naturalness)

        if mode == "both":
            realism = self._add_film_grain(realism, film_grain, naturalness)

        return realism.clamp(0.0, 1.0)

    def _run_custom_model_if_available(
        self,
        image: torch.Tensor,
        skin_detail: float,
        eye_detail: float,
        baby_hair: float,
        film_grain: float,
        naturalness: float,
        strength: float,
        mode: str,
    ) -> torch.Tensor | None:
        """
        Future model hook.

        Drop a TorchScript or PyTorch checkpoint in:
        custom_nodes/radz_human_skin_details/models/

        Replace this stub with your own loading and inference logic later.
        """
        model_path = self._find_model_file()
        if model_path is None:
            return None

        # Placeholder: keep architecture ready without forcing a model format today.
        # Return None to use the handcrafted realism pipeline until a real model is wired in.
        if self._cached_model_path != model_path:
            self._cached_model_path = model_path
            self._cached_model = None
        return None

    def _find_model_file(self) -> Path | None:
        if not self.model_dir.exists():
            return None
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.ts"):
            matches = sorted(self.model_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _placeholder_skin_enhance(
        self,
        image: torch.Tensor,
        skin_detail: float,
        naturalness: float,
    ) -> torch.Tensor:
        luminance = self._luminance(image)
        fine = luminance - self._gaussian_blur(luminance, radius=1, sigma=0.85)
        medium = self._gaussian_blur(luminance, radius=2, sigma=1.35) - self._gaussian_blur(
            luminance, radius=5, sigma=3.2
        )
        broad = luminance - self._gaussian_blur(luminance, radius=7, sigma=4.8)

        # Protect extreme contrast zones to avoid halos around strong edges.
        edge_energy = self._edge_energy(luminance)
        skin_gate = torch.exp(-edge_energy * (3.7 + 1.2 * naturalness))
        midtone_gate = torch.clamp(1.0 - ((luminance - 0.56).abs() / 0.58), 0.0, 1.0)
        texture_gate = skin_gate * (0.82 + 0.38 * midtone_gate)

        fine_rgb = fine.repeat(1, 1, 1, 3)
        medium_rgb = medium.repeat(1, 1, 1, 3)
        broad_rgb = broad.repeat(1, 1, 1, 3)

        micro_amount = 0.08 + 0.18 * skin_detail
        tonal_amount = 0.06 + 0.10 * skin_detail
        broad_amount = 0.02 + 0.04 * skin_detail
        natural_protection = 0.74 + 0.10 * (1.0 - naturalness)

        texture_gate_rgb = texture_gate.repeat(1, 1, 1, 3)
        enhanced = image + texture_gate_rgb * (
            fine_rgb * micro_amount * natural_protection
            + medium_rgb * tonal_amount
            + broad_rgb * broad_amount
        )

        # Smooth flatter skin regions while keeping edges protected.
        skin_smooth = self._gaussian_blur(image, radius=2, sigma=1.55)
        flat_gate = texture_gate * torch.clamp(1.0 - edge_energy * 2.5, 0.0, 1.0)
        smooth_amount = 0.05 + 0.08 * naturalness + 0.03 * (1.0 - skin_detail)
        enhanced = torch.lerp(enhanced, skin_smooth, flat_gate.repeat(1, 1, 1, 3) * smooth_amount)
        enhanced = enhanced + fine_rgb * texture_gate_rgb * (0.03 + 0.08 * skin_detail)

        # Add a very soft edge-aware local contrast lift without turning crunchy.
        low = self._gaussian_blur(image, radius=3, sigma=2.2)
        local_contrast = image - low
        contrast_gate = texture_gate_rgb * (0.35 + 0.65 * midtone_gate.repeat(1, 1, 1, 3))
        enhanced = enhanced + local_contrast * contrast_gate * (0.025 + 0.05 * skin_detail)

        # Recover gentle color separation in low-contrast skin so the pass feels less like plain sharpening.
        chroma_base = self._gaussian_blur(image, radius=2, sigma=1.4)
        chroma_residual = image - chroma_base
        enhanced = enhanced + chroma_residual * texture_gate_rgb * (0.018 + 0.035 * skin_detail)

        # Gentle tonal rolloff keeps skin from looking oily or over-etched.
        smooth_base = self._gaussian_blur(enhanced, radius=1, sigma=0.9)
        polish = 0.04 + 0.05 * naturalness
        enhanced = torch.lerp(enhanced, smooth_base * 0.10 + enhanced * 0.90, polish)
        return enhanced.clamp(0.0, 1.0)

    def _placeholder_eye_enhance(
        self,
        image: torch.Tensor,
        original: torch.Tensor,
        eye_detail: float,
        naturalness: float,
    ) -> torch.Tensor:
        luminance = self._luminance(original)
        edge = self._edge_energy(luminance)
        dark_focus = torch.clamp((0.45 - luminance) / 0.45, 0.0, 1.0)
        highlight_protect = 1.0 - torch.clamp((luminance - 0.72) / 0.28, 0.0, 1.0)
        eye_mask = torch.clamp(edge * 5.0, 0.0, 1.0) * (0.6 * dark_focus + 0.4 * highlight_protect)

        detail = original - self._gaussian_blur(original, radius=1, sigma=0.8)
        amount = (0.01 + 0.035 * eye_detail) * (0.72 - 0.28 * naturalness)
        refined = image + detail * eye_mask.repeat(1, 1, 1, 3) * amount
        return refined.clamp(0.0, 1.0)

    def _synthetic_skin_texture(
        self,
        image: torch.Tensor,
        original: torch.Tensor,
        skin_detail: float,
        naturalness: float,
    ) -> torch.Tensor:
        batch, height, width, _ = image.shape
        device = image.device
        dtype = image.dtype

        luma = self._luminance(original)
        edge_energy = self._edge_energy(luma)
        midtone_gate = torch.clamp(1.0 - ((luma - 0.58).abs() / 0.42), 0.0, 1.0)
        skin_mask = torch.exp(-edge_energy * (4.6 + 1.1 * naturalness)) * (0.68 + 0.22 * midtone_gate)

        noise_small = torch.randn((batch, 1, max(16, height // 2), max(16, width // 2)), device=device, dtype=dtype)
        noise_micro = torch.randn((batch, 1, max(16, height), max(16, width)), device=device, dtype=dtype)
        noise_small = F.interpolate(noise_small, size=(height, width), mode="bicubic", align_corners=False)
        noise_micro = F.interpolate(noise_micro, size=(height, width), mode="bilinear", align_corners=False)

        pores = noise_micro - self._gaussian_blur_chw(noise_micro, radius=1, sigma=0.75)
        pores = pores + 0.45 * (noise_small - self._gaussian_blur_chw(noise_small, radius=2, sigma=1.2))
        pores = pores.permute(0, 2, 3, 1)
        pores = self._gaussian_blur(pores.repeat(1, 1, 1, 3), radius=1, sigma=0.65)[:, :, :, :1]

        chroma_bias = original - self._gaussian_blur(original, radius=3, sigma=2.2)
        luminance_detail = original - self._gaussian_blur(original, radius=2, sigma=1.3)
        texture_rgb = pores.repeat(1, 1, 1, 3) * (0.65 + 0.35 * chroma_bias.sign())
        texture_rgb = texture_rgb + luminance_detail * 0.14
        texture_rgb = self._gaussian_blur(texture_rgb, radius=1, sigma=0.75)
        amount = 0.025 + 0.1 * skin_detail + 0.03 * (1.0 - naturalness)
        textured = image + texture_rgb * skin_mask.repeat(1, 1, 1, 3) * amount
        polished = self._gaussian_blur(textured, radius=1, sigma=0.85)
        textured = torch.lerp(textured, polished, 0.03 + 0.08 * naturalness)
        return textured.clamp(0.0, 1.0)

    def _placeholder_baby_hair_detail(
        self,
        image: torch.Tensor,
        original: torch.Tensor,
        baby_hair: float,
        naturalness: float,
    ) -> torch.Tensor:
        luminance = self._luminance(original)
        gradient_x = torch.abs(luminance[:, :, 1:, :] - luminance[:, :, :-1, :])
        gradient_y = torch.abs(luminance[:, 1:, :, :] - luminance[:, :-1, :, :])
        gradient_x = torch.cat([gradient_x, gradient_x[:, :, -1:, :]], dim=2)
        gradient_y = torch.cat([gradient_y, gradient_y[:, -1:, :, :]], dim=1)
        oriented_edges = gradient_x * 0.6 + gradient_y * 0.4

        broad_edges = self._edge_energy(luminance)
        hair_mask = torch.clamp(oriented_edges * 8.0, 0.0, 1.0) * torch.exp(-broad_edges * 4.5)

        high_freq = original - self._gaussian_blur(original, radius=1, sigma=0.65)
        amount = (0.006 + 0.02 * baby_hair) * (0.7 - 0.25 * naturalness)
        result = image + high_freq * hair_mask.repeat(1, 1, 1, 3) * amount
        return result.clamp(0.0, 1.0)

    def _add_film_grain(
        self,
        image: torch.Tensor,
        film_grain: float,
        naturalness: float,
    ) -> torch.Tensor:
        if film_grain <= 0.0:
            return image

        batch, height, width, _ = image.shape
        device = image.device
        dtype = image.dtype

        grain_h = max(16, height // 2)
        grain_w = max(16, width // 2)

        mono = torch.randn((batch, 1, grain_h, grain_w), device=device, dtype=dtype)
        soft = self._gaussian_blur_chw(mono, radius=2, sigma=1.15)
        fine = mono - soft * 0.72
        grain = F.interpolate(fine, size=(height, width), mode="bilinear", align_corners=False)
        grain = grain.permute(0, 2, 3, 1)
        grain = self._gaussian_blur(grain.repeat(1, 1, 1, 3), radius=1, sigma=0.8)[:, :, :, :1]

        luma = self._luminance(image)
        shadow_weight = torch.clamp((0.48 - luma) / 0.48, 0.0, 1.0)
        mid_weight = torch.clamp(1.0 - torch.abs(luma - 0.52) / 0.42, 0.0, 1.0)
        highlight_weight = torch.clamp((0.85 - luma) / 0.85, 0.0, 1.0)
        grain_mask = 0.22 * shadow_weight + 0.26 * mid_weight + 0.06 * highlight_weight

        amount = film_grain * (0.0015 + 0.008 * (1.0 - naturalness))
        grain_rgb = grain.repeat(1, 1, 1, 3)
        output = image + grain_rgb * grain_mask.repeat(1, 1, 1, 3) * amount
        return output.clamp(0.0, 1.0)

    def _upscale_and_refine(
        self,
        image: torch.Tensor,
        upscale_factor: float,
        skin_detail: float,
        naturalness: float,
    ) -> torch.Tensor:
        batch, height, width, _ = image.shape
        target_h = max(height, int(round(height * upscale_factor)))
        target_w = max(width, int(round(width * upscale_factor)))
        chw = image.permute(0, 3, 1, 2)
        upscaled = F.interpolate(chw, size=(target_h, target_w), mode="bicubic", align_corners=False)
        upscaled = upscaled.permute(0, 2, 3, 1).clamp(0.0, 1.0)

        sharpen_base = self._gaussian_blur(upscaled, radius=1, sigma=0.8)
        medium_base = self._gaussian_blur(upscaled, radius=2, sigma=1.6)
        hi = upscaled - sharpen_base
        med = upscaled - medium_base
        sharpen_amount = 0.12 + 0.12 * skin_detail - 0.02 * naturalness
        refined = upscaled + hi * sharpen_amount + med * (0.05 + 0.08 * skin_detail)
        refined = self._synthetic_skin_texture(refined, upscaled, min(1.0, skin_detail + 0.08), min(1.0, naturalness * 0.95))
        refined = torch.lerp(refined, self._gaussian_blur(refined, radius=1, sigma=0.7), 0.025 + 0.05 * naturalness)
        return refined.clamp(0.0, 1.0)

    def _luminance(self, image: torch.Tensor) -> torch.Tensor:
        coeffs = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
        return (image * coeffs).sum(dim=-1, keepdim=True)

    def _edge_energy(self, luminance: torch.Tensor) -> torch.Tensor:
        tensor = luminance.permute(0, 3, 1, 2)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=tensor.device,
            dtype=tensor.dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=tensor.device,
            dtype=tensor.dtype,
        ).view(1, 1, 3, 3)
        padded = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(padded, sobel_x)
        gy = F.conv2d(padded, sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return edge.permute(0, 2, 3, 1)

    def _gaussian_blur(self, image: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
        chw = image.permute(0, 3, 1, 2)
        blurred = self._gaussian_blur_chw(chw, radius, sigma)
        return blurred.permute(0, 2, 3, 1)

    def _gaussian_blur_chw(self, image: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
        if radius <= 0:
            return image

        kernel_size = radius * 2 + 1
        coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - radius
        kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)

        padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(padded, kernel, groups=image.shape[1])


class RadzHumanSkinDetailsNode:
    CATEGORY = "Radz/Skin"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"

    def __init__(self):
        self.engine = RealHumanDetailEngine()

    @classmethod
    def INPUT_TYPES(cls):
        return cls._build_input_types(
            default_upscale_preference="4xFaceUpDAT.pth",
            default_upscale_2_preference="1xSkinContrast-High-SuperUltraCompact.pth",
            default_upscale_3_preference="none",
            default_skin_detail=0.74,
            default_eye_detail=0.24,
            default_baby_hair=0.10,
            default_film_grain=0.01,
            default_naturalness=0.80,
            default_strength=0.64,
            default_upscale_factor="2x",
            default_mode="detail_only",
        )

    @classmethod
    def _build_input_types(
        cls,
        default_upscale_preference: str,
        default_upscale_2_preference: str,
        default_upscale_3_preference: str,
        default_skin_detail: float,
        default_eye_detail: float,
        default_baby_hair: float,
        default_film_grain: float,
        default_naturalness: float,
        default_strength: float,
        default_upscale_factor: str,
        default_mode: str,
    ):
        upscale_models = ["none"] + folder_paths.get_filename_list("upscale_models")
        fallback_upscale = (
            "RealESRGAN_x2.pth"
            if "RealESRGAN_x2.pth" in upscale_models
            else ("RealESRGAN_x4plus.safetensors" if "RealESRGAN_x4plus.safetensors" in upscale_models else upscale_models[0])
        )
        default_upscale = default_upscale_preference if default_upscale_preference in upscale_models else fallback_upscale
        default_skin = (
            "x1_ITF_SkinDiffDetail_Lite_v1.pth"
            if "x1_ITF_SkinDiffDetail_Lite_v1.pth" in upscale_models
            else upscale_models[0]
        )
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (upscale_models, {"default": default_upscale}),
                "upscale_model_2": (
                    upscale_models,
                    {
                        "default": (
                            default_upscale_2_preference
                            if default_upscale_2_preference in upscale_models
                            else "none"
                        )
                    },
                ),
                "upscale_model_3": (
                    upscale_models,
                    {
                        "default": (
                            default_upscale_3_preference
                            if default_upscale_3_preference in upscale_models
                            else "none"
                        )
                    },
                ),
                "skin_enhancer_model": (upscale_models, {"default": default_skin}),
                "skin_detail": ("FLOAT", {"default": default_skin_detail, "min": 0.01, "max": 1.0, "step": 0.01}),
                "eye_detail": ("FLOAT", {"default": default_eye_detail, "min": 0.01, "max": 1.0, "step": 0.01}),
                "baby_hair": ("FLOAT", {"default": default_baby_hair, "min": 0.01, "max": 1.0, "step": 0.01}),
                "film_grain": ("FLOAT", {"default": default_film_grain, "min": 0.0, "max": 1.0, "step": 0.01}),
                "naturalness": ("FLOAT", {"default": default_naturalness, "min": 0.01, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": default_strength, "min": 0.01, "max": 2.0, "step": 0.01}),
                "upscale_factor": (["1x", "2x", "4x", "6x"], {"default": default_upscale_factor}),
                "mode": (["detail_only", "both"], {"default": default_mode}),
            },
            "optional": {
                "enable_tiling": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "tile_overlap": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
            },
        }

    def process(
        self,
        image: torch.Tensor,
        upscale_model: str,
        upscale_model_2: str,
        upscale_model_3: str,
        skin_enhancer_model: str,
        skin_detail: float,
        eye_detail: float,
        baby_hair: float,
        film_grain: float,
        naturalness: float,
        strength: float,
        upscale_factor: str,
        mode: str,
        enable_tiling: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 64,
    ) -> Tuple[torch.Tensor]:
        source = self._prepare_image_tensor(image)
        upscale_factor_value = self._resolve_upscale_factor(upscale_factor)

        if enable_tiling:
            result = self._process_tiled(
                source=source,
                upscale_model=upscale_model,
                upscale_model_2=upscale_model_2,
                upscale_model_3=upscale_model_3,
                skin_enhancer_model=skin_enhancer_model,
                skin_detail=skin_detail,
                eye_detail=eye_detail,
                baby_hair=baby_hair,
                film_grain=film_grain,
                naturalness=naturalness,
                strength=strength,
                upscale_factor=upscale_factor_value,
                mode=mode,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
        else:
            result = self.engine.enhance(
                image=source,
                upscale_model_name=upscale_model,
                upscale_model_name_2=upscale_model_2,
                upscale_model_name_3=upscale_model_3,
                skin_enhancer_model_name=skin_enhancer_model,
                skin_detail=skin_detail,
                eye_detail=eye_detail,
                baby_hair=baby_hair,
                film_grain=film_grain,
                naturalness=naturalness,
                strength=strength,
                mode=mode,
                upscale_factor=upscale_factor_value,
            )

        return (result.clamp(0.0, 1.0).to(image.device),)

    def _prepare_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Radz Human Skin Details expects an IMAGE tensor with shape (B, H, W, 3).")
        if not torch.is_floating_point(image):
            image = image.float()
        return image.clamp(0.0, 1.0)

    def _process_tiled(
        self,
        source: torch.Tensor,
        upscale_model: str,
        upscale_model_2: str,
        upscale_model_3: str,
        skin_enhancer_model: str,
        skin_detail: float,
        eye_detail: float,
        baby_hair: float,
        film_grain: float,
        naturalness: float,
        strength: float,
        mode: str,
        upscale_factor: float,
        tile_size: int,
        tile_overlap: int,
    ) -> torch.Tensor:
        batch, height, width, channels = source.shape
        if (
            upscale_model != "none"
            or upscale_model_2 != "none"
            or upscale_model_3 != "none"
            or skin_enhancer_model != "none"
            or (tile_size >= height and tile_size >= width)
        ):
            return self.engine.enhance(
                image=source,
                upscale_model_name=upscale_model,
                upscale_model_name_2=upscale_model_2,
                upscale_model_name_3=upscale_model_3,
                skin_enhancer_model_name=skin_enhancer_model,
                skin_detail=skin_detail,
                eye_detail=eye_detail,
                baby_hair=baby_hair,
                film_grain=film_grain,
                naturalness=naturalness,
                strength=strength,
                mode=mode,
                upscale_factor=upscale_factor,
            )

        output = torch.zeros_like(source)
        weight_sum = torch.zeros((batch, height, width, 1), device=source.device, dtype=source.dtype)
        window_cache = {}

        for tile in self._generate_tiles(height, width, tile_size, tile_overlap):
            patch = source[:, tile.y0:tile.y1, tile.x0:tile.x1, :]
            patch_out = self.engine.enhance(
                image=patch,
                upscale_model_name="none",
                upscale_model_name_2="none",
                upscale_model_name_3="none",
                skin_enhancer_model_name="none",
                skin_detail=skin_detail,
                eye_detail=eye_detail,
                baby_hair=baby_hair,
                film_grain=film_grain,
                naturalness=naturalness,
                strength=strength,
                mode=mode,
                upscale_factor=1.0,
            )

            key = (tile.y1 - tile.y0, tile.x1 - tile.x0)
            if key not in window_cache:
                window_cache[key] = self._feather_window(key[0], key[1], source.device, source.dtype)
            window = window_cache[key]

            output[:, tile.y0:tile.y1, tile.x0:tile.x1, :] += patch_out * window
            weight_sum[:, tile.y0:tile.y1, tile.x0:tile.x1, :] += window

        merged = (output / weight_sum.clamp_min(1e-6)).clamp(0.0, 1.0)
        if upscale_factor > 1.0:
            return self.engine._upscale_and_refine(merged, upscale_factor, skin_detail, naturalness)
        return merged

    def _resolve_upscale_factor(self, upscale_factor: str) -> float:
        mapping = {
            "1x": 1.0,
            "2x": 2.0,
            "4x": 4.0,
            "6x": 6.0,
        }
        return mapping.get(upscale_factor, 2.0)

    def _generate_tiles(self, height: int, width: int, tile_size: int, tile_overlap: int) -> List[TileInfo]:
        step = max(1, tile_size - tile_overlap)

        def positions(length: int) -> List[int]:
            coords = [0]
            while coords[-1] + tile_size < length:
                next_pos = coords[-1] + step
                if next_pos + tile_size >= length:
                    coords.append(max(0, length - tile_size))
                    break
                coords.append(next_pos)
            return sorted(set(coords))

        ys = positions(height)
        xs = positions(width)
        return [
            TileInfo(y0=y, y1=min(y + tile_size, height), x0=x, x1=min(x + tile_size, width))
            for y in ys
            for x in xs
        ]

    def _feather_window(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.from_numpy(np.hanning(height)).to(device=device, dtype=dtype).clamp_min(1e-3)
        x = torch.from_numpy(np.hanning(width)).to(device=device, dtype=dtype).clamp_min(1e-3)
        window = torch.outer(y, x).view(1, height, width, 1)
        return window


class RadzSDXLSkinRealismNode(RadzHumanSkinDetailsNode):
    CATEGORY = "Radz/SDXL"

    @classmethod
    def INPUT_TYPES(cls):
        return cls._build_input_types(
            default_upscale_preference="none",
            default_upscale_2_preference="1xSkinContrast-High-SuperUltraCompact.pth",
            default_upscale_3_preference="none",
            default_skin_detail=0.72,
            default_eye_detail=0.22,
            default_baby_hair=0.08,
            default_film_grain=0.00,
            default_naturalness=0.88,
            default_strength=0.52,
            default_upscale_factor="1x",
            default_mode="detail_only",
        )


NODE_CLASS_MAPPINGS = {
    "RodzRealHumanDetail": RadzHumanSkinDetailsNode,
    "RadzHumanSkinDetails": RadzHumanSkinDetailsNode,
    "RadzSDXLSkinRealism": RadzSDXLSkinRealismNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "RodzRealHumanDetail": "Radz Human Skin Details",
    "RadzHumanSkinDetails": "Radz Human Skin Details",
    "RadzSDXLSkinRealism": "Radz SDXL Skin Realism",
}
