"""Completion-token GRPO and independent-frame gradient accumulation."""
import torch


def advantages(rewards, floor=1e-4):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 1 or len(rewards) < 2 or not torch.isfinite(rewards).all():
        raise ValueError("Finite group rewards required")
    if floor <= 0:
        raise ValueError("std floor must be positive")
    return ((rewards - rewards.mean()) /
            rewards.std(unbiased=False).clamp_min(floor)).detach()


def grpo_loss(logp, old_logp, ref_logp, advantage, clip=0.2, beta=0.01):
    if logp.ndim != 1 or not len(logp):
        raise ValueError("One nonempty completion per call")
    if old_logp.shape != logp.shape or ref_logp.shape != logp.shape:
        raise ValueError("Completion log-probability shapes differ")
    if not all(torch.isfinite(x).all() for x in (logp, old_logp, ref_logp)):
        raise FloatingPointError("Nonfinite completion log probabilities")
    ratio = (logp - old_logp.detach()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    delta = ref_logp.detach() - logp
    kl = delta.exp() - delta - 1
    return (-surrogate + beta * kl).mean()


def backward_group(policy, prompt, completions, old, reference, reward, frame_scale):
    adv = advantages(reward, policy.std_floor).to(policy.device)
    values = []
    for i, completion in enumerate(completions):
        current = policy.logprobs(prompt, completion)
        loss = grpo_loss(current, old[i].to(current.device), reference[i].to(current.device),
                         adv[i], policy.clip, policy.beta)
        (loss / (len(completions) * frame_scale)).backward()
        values.append(float(loss.detach()))
    return sum(values) / len(values)
