"""NeoSim force observations -> frozen NeoForce -> experimental RGB pairing.

No depth/object pose is read. Force fields are simulator sensor-frame samples;
their sim-to-real distribution is an explicit evaluation limitation. RGB pairing
is a mutual-feature heuristic, not ground-truth visibility or correspondence.
"""
import torch
from .neoforce import pool_contact_tokens
from .rgb_contact_matching import mutual_contact_matches, pool_visual_tokens


@torch.no_grad()
def build_online_contact_pairs(obs, *, encoder, anchors, camera_keys,
                               tactile_keys, visual_grid, tactile_grid,
                               paired_camera, device, min_similarity=.7,
                               min_margin=.15):
    rgb_frames = obs['obs'] if isinstance(obs['obs'], list) else [obs['obs']]
    tactile_frames = obs['tactile'] if isinstance(obs['tactile'], list) else [obs['tactile']]
    if len(rgb_frames) != len(tactile_frames) or not rgb_frames:
        raise ValueError('online NeoForce requires synchronized RGB/tactile frame histories')
    sensors = len(tactile_keys)
    if sensors not in (2, 4) or paired_camera not in camera_keys:
        raise ValueError('online NeoForce requires 2/4 ordered sensors and a configured third-person camera')
    ids = torch.as_tensor(anchors)
    if (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not ids.numel()
            or (ids < 0).any() or (ids >= len(rgb_frames)).any()
            or (ids[1:] <= ids[:-1]).any()):
        raise ValueError('online NeoForce anchors must be increasing raw frame IDs')
    rgb, force = [], []
    for frame, touch in zip(rgb_frames, tactile_frames):
        image = torch.as_tensor(frame[paired_camera], device=device)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
            raise ValueError('online NeoForce RGB must be uint8 HWC')
        if touch.get('__ikv_force_schema__') != 'neosim_sensor_xyz_v1':
            raise ValueError('missing/unknown NeoSim sensor force schema; do not silently zero contact')
        if list(touch.get('__ikv_force_keys__', [])) != list(tactile_keys):
            raise ValueError('NeoForce sensor order does not match tactile expert order')
        field = torch.as_tensor(touch['__ikv_force__'], device=device).float()
        if field.ndim != 4 or field.shape[0] != sensors or field.shape[-1] != 3 or not torch.isfinite(field).all():
            raise ValueError('force fields must be finite [sensors,H,W,3]')
        rgb.append(image.permute(2, 0, 1).float() / 255.)
        force.append(field.permute(0, 3, 1, 2))
    rgb = torch.stack(rgb)[None]
    force = torch.stack(force)  # T,S,3,H,W
    camera_id = list(camera_keys).index(paired_camera)
    vh, vw = visual_grid
    all_features, all_response, all_rows = [], [], []
    for arm in range(sensors // 2):
        arm_features, arm_response, arm_rows = [], [], []
        for output_frame, anchor in enumerate(ids.tolist()):
            start = max(0, anchor - 3)
            local_force = force[start:anchor+1, arm*2:arm*2+2].reshape(1, anchor-start+1, 6, *force.shape[-2:])
            # UIPC force samples are zero away from contact. Retain separate
            # local responses: opposite fingers must not cancel one another.
            mask = local_force.ne(0).float()
            encoded = encoder(rgb[:, start:anchor+1], local_force, mask)
            if 'visual_tokens' not in encoded:
                raise ValueError('NeoForce encoder must expose visual patch tokens for RGB matching')
            pooled = pool_contact_tokens({'tokens': encoded['tokens'][:, -1:],
                                          'response': encoded['response'][:, -1:]}, tactile_grid)
            v = pool_visual_tokens(encoded['visual_tokens'][:, -1:], visual_grid).reshape(vh*vw, -1)
            f = pooled['neoforce'][0, :, 0]
            r = pooled['response'][0, :, 0]
            sensor_rows = []
            for sensor in range(2):
                matched = mutual_contact_matches(v, f[sensor].flatten(0, 1), r[sensor].flatten(),
                                                 min_similarity=min_similarity, min_margin=min_margin)
                rows = matched['visual_rows']
                found = rows >= 0
                packed = rows.clone()
                packed[found] = ((output_frame * vh + rows[found] // vw) * len(camera_keys) * vw
                                 + camera_id * vw + rows[found] % vw)
                sensor_rows.append(packed.reshape(*tactile_grid))
            arm_features.append(f)
            arm_response.append(r)
            arm_rows.append(torch.stack(sensor_rows))
        all_features.append(torch.stack(arm_features, dim=1))
        all_response.append(torch.stack(arm_response, dim=1))
        all_rows.append(torch.stack(arm_rows, dim=1))
    features = torch.cat(all_features).flatten(0, 3)
    response = torch.cat(all_response).flatten()
    rows = torch.cat(all_rows).flatten()
    return {'neoforce': features, 'response': response, 'visual_rows': rows}
