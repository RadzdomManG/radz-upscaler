$ErrorActionPreference = "Stop"

$comfyRoot = "D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
$radzRoot = Join-Path $comfyRoot "custom_nodes\Radz Human Skin details"
$bundledModels = Join-Path $radzRoot "bundled_models"
$extraModelPaths = Join-Path $comfyRoot "extra_model_paths.yaml"

New-Item -ItemType Directory -Force (Join-Path $bundledModels "clip_vision") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $bundledModels "ipadapter") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $bundledModels "loras") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $bundledModels "insightface") | Out-Null

$yaml = @"
radz_nodes:
  base_path: custom_nodes/Radz Human Skin details/bundled_models
  is_default: false
  clip_vision: clip_vision
  ipadapter: ipadapter
  loras: loras
"@

Set-Content -LiteralPath $extraModelPaths -Value $yaml -Encoding UTF8

Write-Host "Radz Human Skin details model paths configured:"
Write-Host "  $bundledModels"
Write-Host "ComfyUI extra model paths written to:"
Write-Host "  $extraModelPaths"
Write-Host ""
Write-Host "Next step: place IPAdapter, clip vision, LoRA, and InsightFace files into the bundled_models subfolders."
