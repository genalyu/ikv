"""Supported input contract: full native RGB, with optional IKV retention."""
def validate_dense_rgb_config(config):
    for key in ("use_rgb_motion_tokens", "rgb_motion_online_preprocess"):
        value = config.get(key, False) if isinstance(config, dict) else getattr(config, key, False)
        if bool(value):
            raise ValueError(f"{key} was removed; use full-RGB baseline or IKV")

def validate_dense_rgb_payload(payload):
    if payload is None:
        return
    removed = {"rgb_motion_patch_indices", "rgb_motion_indices", "rgb_motion_valid_mask",
        "motion_indices", "motion_valid_mask", "motion_scores", "condition_motion",
        "rgb_motion", "rgb_motion_inputs"}
    if any(key in payload for key in removed):
        raise ValueError("Motion payloads were removed; provide full native RGB")
