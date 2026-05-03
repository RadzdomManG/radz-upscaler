from .radz_insight_face import (
    NODE_CLASS_MAPPINGS as IPADAPTER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IPADAPTER_NODE_DISPLAY_NAME_MAPPINGS,
)


def _radz_node_id(node_id: str) -> str:
    if node_id.startswith("IPAdapter"):
        suffix = node_id[len("IPAdapter"):]
        return f"RadzInsightFace{suffix}"
    if node_id.startswith("IPAAdapter"):
        suffix = node_id[len("IPAAdapter"):]
        return f"RadzInsightFace{suffix}"
    if node_id.startswith("PrepImageForClipVision"):
        return "RadzInsightFacePrepImageForClipVision"
    return f"RadzInsightFace{node_id}"


def _radz_display_name(name: str) -> str:
    return f"Radz Insight Face {name}"


NODE_CLASS_MAPPINGS = {
    _radz_node_id(node_id): node_class
    for node_id, node_class in IPADAPTER_NODE_CLASS_MAPPINGS.items()
}

NODE_DISPLAY_NAME_MAPPINGS = {
    _radz_node_id(node_id): _radz_display_name(
        IPADAPTER_NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
    )
    for node_id in IPADAPTER_NODE_CLASS_MAPPINGS
}
