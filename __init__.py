import logging
import os
import sys

logger = logging.getLogger(__name__)

# Check for the bundled original USDU script.
current_dir = os.path.dirname(os.path.realpath(__file__))
repos_dir = os.path.join(current_dir, "repositories")
usdu_dir = os.path.join(repos_dir, "ultimate_sd_upscale")
if not len(os.listdir(usdu_dir)):
    logger.info(
        "Original USDU script not found, downloading it from "
        "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111"
    )
    import shutil
    import urllib.request
    import zipfile

    url = "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/archive/master.zip"
    zip_path = os.path.join(current_dir, "usdu_temp.zip")

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        top_folder = zip_ref.namelist()[0].split("/")[0] + "/"
        for member in zip_ref.namelist():
            if member.startswith(top_folder) and not member.endswith("/"):
                relative_path = member[len(top_folder):]
                target = os.path.join(usdu_dir, relative_path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zip_ref.open(member) as source, open(target, "wb") as target_file:
                    shutil.copyfileobj(source, target_file)

    os.remove(zip_path)
    logger.info("Original USDU script downloaded successfully")

# Remove other custom node paths from sys.path to avoid conflicts.
custom_node_paths = [path for path in sys.path if "custom_node" in path]
original_sys_path = sys.path.copy()
for path in custom_node_paths:
    sys.path.remove(path)

# Add this repository's path to sys.path for third-party imports.
repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)
original_modules = sys.modules.copy()

# Place aside potentially conflicting modules.
modules_used = [
    "modules",
    "modules.devices",
    "modules.images",
    "modules.processing",
    "modules.scripts",
    "modules.shared",
    "modules.upscaler",
    "utils",
]
original_imported_modules = {}
for module in modules_used:
    if module in sys.modules:
        original_imported_modules[module] = sys.modules.pop(module)

from .radz_nodes import (
    NODE_CLASS_MAPPINGS as RADZ_UPSCALER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RADZ_UPSCALER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .radz_ipadapter_nodes import (
    NODE_CLASS_MAPPINGS as RADZ_IPADAPTER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RADZ_IPADAPTER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .radz_human_skin_detailer import (
    NODE_CLASS_MAPPINGS as RADZ_HUMAN_DETAIL_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RADZ_HUMAN_DETAIL_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(RADZ_UPSCALER_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(RADZ_IPADAPTER_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(RADZ_HUMAN_DETAIL_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(RADZ_UPSCALER_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(RADZ_IPADAPTER_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(RADZ_HUMAN_DETAIL_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Clean up imports.
modules_to_remove = []
for module in sys.modules:
    if module not in original_modules:
        modules_to_remove.append(module)
for module in modules_to_remove:
    del sys.modules[module]

# Restore original modules and sys.path.
sys.modules.update(original_imported_modules)
sys.path = original_sys_path
