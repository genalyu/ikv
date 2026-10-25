"""Reuse N0 Trainer preparation, schedulers, forward and compute_loss."""
from copy import deepcopy
import torch


def final_frame_loss_inputs(data, predictions, patch_size):
    """Slice predictions AND targets; retain original FM weighting/reduction."""
    if patch_size[0] != 1:
        raise ValueError("Temporal patching is not supported by final-frame rewards")
    if any("motion" in key for key in data["latent_dict"]):
        raise ValueError("Initial scorer reward path requires dense RGB tokens")
    result = dict(data, latent_dict=dict(data["latent_dict"]), action_dict=dict(data["action_dict"]))
    v, a = result["latent_dict"], result["action_dict"]
    frames = v["targets"].shape[2]
    if frames < 2 or a["targets"].shape[2] != frames:
        raise ValueError("Aligned video/action history plus target frame required")
    outputs = []
    for pred, source in zip(predictions[:2], (v, a)):
        if pred.shape[1] % frames:
            raise ValueError("Prediction sequence does not align with frames")
        outputs.append(pred[:, -(pred.shape[1] // frames):])
        source["targets"] = source["targets"][:, :, -1:]
        source["timesteps"] = source["timesteps"][:, -1:]
    a["actions_mask"] = a["actions_mask"][:, :, -1:]
    if not a["actions_mask"].bool().any():
        raise ValueError("Final frame has no valid expert action target")
    if len(predictions) == 3:
        tactile = predictions[2]
        target = a.get("tactile_global_targets")
        if target is not None and tactile is not None:
            B,S,C,F,H,W = target.shape
            if F != frames:
                raise ValueError("Tactile/video frame alignment mismatch")
            # Preserve native sensor-major patch ordering.
            tactile = tactile.reshape(B, S, F, -1, tactile.shape[-1])[:, :, -1:]
            outputs.append(tactile.flatten(1, 3))
            a["tactile_global_targets"] = target[:, :, :, -1:]
        else:
            outputs.append(tactile)
    return result, tuple(outputs)


class FMReward:
    def __init__(self, trainer, model, weights=(1.,1.,1.), scales=(1.,1.,1.)):
        self.trainer, self.model = trainer, model.eval().requires_grad_(False)
        if not hasattr(trainer, "device"):
            trainer.device = next(model.parameters()).device
        self.weights, self.scales = weights, scales
        if len(weights) != 3 or len(scales) != 3 or any(x <= 0 for x in scales):
            raise ValueError("Three positive calibrated modality scales required")
        trainer.gradient_accumulation_steps = 1
        trainer.transformer = model

    @torch.no_grad()
    def prepare(self, raw_batch, seed, time_index):
        from unittest.mock import patch
        # The Trainer uses module-level sampling. Scope this replacement to this
        # process/call, restore it even on failure; other training jobs unaffected.
        module = __import__(type(self.trainer).__module__, fromlist=["sample_timestep_id"])
        original_sampler = module.sample_timestep_id
        def fixed_time(batch_size, num_train_timesteps, **kwargs):
            if kwargs:
                return original_sampler(batch_size=batch_size,
                                        num_train_timesteps=num_train_timesteps, **kwargs)
            if not 0 <= time_index < num_train_timesteps:
                raise ValueError("FM time index out of range")
            return torch.full((batch_size,), time_index, dtype=torch.long)
        device = self.trainer.device
        devices = [] if device.type == "cpu" else [device.index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            with patch.object(module, "sample_timestep_id", fixed_time):
                batch = self.trainer.convert_input_format(deepcopy(raw_batch))
                prepared = self.trainer._prepare_input_dict(batch)
        prepared["chunk_size"] = 1
        prepared["window_size"] = 2 * prepared["latent_dict"]["targets"].shape[2] + 2
        config = prepared.get("ikv_training")
        if config is None:
            raise ValueError("Trainer config must enable IKV training")
        config.update(execution="masked", compact_attention=False, sample_capacity=False)
        config.pop("history_cache_key", None)
        return prepared

    @torch.no_grad()
    def evaluate(self, prepared, video_scores, budgets, tactile_scores=None):
        case = deepcopy(prepared)
        frames = case["latent_dict"]["targets"].shape[2]
        config = case["ikv_training"]
        config["capacity"] = sum(budgets)
        if config.get("retention", {}).get("version", 1) == 2:
            config["retention"].update(zip(
                ("video_capacity","action_capacity","tactile_capacity"), budgets))
        config["scorer_query"] = dict(target_phase=2*(frames-1),
                                     video_scores=video_scores, budgets=list(budgets))
        if tactile_scores is not None:
            config["scorer_query"]["tactile_scores"] = tactile_scores
        # Capacity belongs to the historical pool; simultaneous noisy tokens are
        # required inputs and are not counted as historical retained slots.
        with torch.autocast(device_type=self.trainer.device.type,
                            dtype=torch.bfloat16, enabled=self.trainer.device.type == "cuda"):
            outputs = self.model(case, train_mode=True)
        sliced, predictions = final_frame_loss_inputs(case, outputs, self.trainer.patch_size)
        losses = self.trainer.compute_loss(sliced, predictions)
        values = [float(losses[k]) for k in ("latent_loss","action_loss","tactile_loss")]
        if not torch.isfinite(torch.tensor(values)).all():
            raise FloatingPointError("Nonfinite frozen FM reward")
        reward = -sum(w*v/s for w,v,s in zip(self.weights,values,self.scales))
        selected = config.get("scorer_selected_indices")
        if selected is None:
            raise RuntimeError("Scorer mask hook did not execute")
        return reward, dict(zip(("video","action","tactile"),values)), selected


def map_scores(scores, coordinates):
    """Explicit provenance map: each clean video token -> raw RGB [f,c,r,col].

    No implicit latent-time to raw-time conversion. Target tokens use -1 rows.
    """
    mapping = torch.as_tensor(coordinates, dtype=torch.long)
    if mapping.ndim != 2 or mapping.shape[1] != 4:
        raise ValueError("kv_coordinates must have shape [clean_video_tokens,4]")
    array = torch.as_tensor(scores)
    valid = (mapping >= 0).all(1)
    if ((mapping < 0).any(1) & ~(mapping == -1).all(1)).any():
        raise ValueError("Unmapped rows must be all -1")
    for dim, bound in enumerate(array.shape):
        if (mapping[valid, dim] >= bound).any():
            raise ValueError("KV map references an unobserved/future RGB patch")
    result = torch.zeros(len(mapping), dtype=torch.float32)
    if valid.any():
        result[valid] = array[tuple(mapping[valid].T)].float()
    return result
