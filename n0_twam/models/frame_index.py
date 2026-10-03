"""Position-free frame annotations shared by training and serving."""
import torch


def broadcast_frame_contacts(rows, frame_features=None, frame_times=None):
    """Broadcast a frame descriptor to visual rows; preserve tactile descriptors.

    External frame features must already be contact-gated (zero = no contact).
    Without them, pool active tactile descriptors once per time, never per pixel.
    """
    rows = dict(rows)
    times = rows["world_time_id"]
    explicit = frame_features is not None
    presence = rows.get("contact_present", rows["neoforce"].ne(0).any(-1)).clone()
    if not explicit:
        features = rows["neoforce"]
        active = (rows["kind"] == 2) & rows["observation_flag"] & presence
        if not active.any():
            return rows
        frame_times = torch.unique(times[active], sorted=True)
        frame_features = torch.stack([features[active & (times == t)].mean(0) for t in frame_times])
    if frame_features.ndim != 2 or len(frame_features) != len(frame_times):
        raise ValueError("frame NeoForce requires [F,D] features and [F] times")
    if not torch.isfinite(frame_features).all() or not torch.isfinite(frame_times).all():
        raise ValueError("frame NeoForce must be finite")
    width = frame_features.shape[-1]
    if rows["neoforce"].shape[-1] not in (0, width):
        raise ValueError("frame/tactile NeoForce widths disagree")
    neo = rows["neoforce"].clone() if rows["neoforce"].shape[-1] else frame_features.new_zeros(len(times), width)
    for t, descriptor in zip(frame_times, frame_features):
        selected = (times == t) & rows["observation_flag"] & (rows["kind"] != 1)
        if not explicit:
            selected &= rows["kind"] == 0
        neo[selected] = descriptor
        presence[selected] = bool(descriptor.ne(0).any()) if explicit else True
    rows["neoforce"] = neo
    rows["contact_present"] = presence
    return rows


def dense_observations(index, grid, *, required=False):
    """Extract real full-grid DINO rows, independently of sparse selection."""
    if index is None or not index.get("dino", torch.empty(0)).numel():
        if required:
            raise ValueError("v2 content scores require full-grid DINO before motion selection")
        return None
    features = index["dino"]
    if features.ndim == 3:
        if not torch.equal(features, features[:1].expand_as(features)):
            raise ValueError("CFG DINO metadata must agree")
        features = features[0]
    times = grid[0, 0].float()
    if features.ndim != 2 or len(features) != len(times):
        raise ValueError("full-grid DINO must align with original video grid")
    flags = torch.as_tensor(index.get("observation_flag", 1), device=features.device)
    if flags.ndim == 2:
        if not torch.equal(flags, flags[:1].expand_as(flags)):
            raise ValueError("CFG observation flags must agree")
        flags = flags[0]
    if not ((flags == 0) | (flags == 1)).all():
        raise ValueError("observation flags must be binary")
    flags = flags.reshape(-1)
    if len(flags) not in (1, len(times)):
        raise ValueError("full-grid observation flags mismatch")
    real = flags.expand(len(times)).bool()
    duration = torch.as_tensor(index.get("duration", 1.), device=features.device)
    if duration.ndim == 2:
        if not torch.equal(duration, duration[:1].expand_as(duration)):
            raise ValueError("CFG observation durations must agree")
        duration = duration[0]
    duration = duration.reshape(-1).expand(len(times))
    if not torch.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("observation duration must be finite and nonnegative")
    return features[real].detach(), times[real].detach(), duration[real].detach()

