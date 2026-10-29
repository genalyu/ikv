# Explicit observed-current context in native retention

FrozenMemoryRetention supports optional paired score_history(history_prefix, observed_prefix)
and observed_prefix() callbacks. Prefixes are exclusive native video-frame counts.
The callback scores only historical candidates but sees the real observed current.
During a prediction insertion the latest real RGB frame stays protected; its192slots
count inside the video budget, alongside reserved incoming tokens. During grounding
the newly observed frames are incoming and already reserved. Action retention stays
recency-based. No deleted KV ID can be resurrected by scoring.

The serving hooks wire this protocol only when memory_history_context=True on their
configuration. Default behavior preserves existing frozen bindings and offline replay.
This flag is not exposed as an unchecked serve.py override: existing checkpoint
manifests/plans were trained with the older contract and must not be silently relabeled.
A diagnostic harness may select it explicitly and record the candidate/checkpoint and
contract mismatch. A production binding requires matching versioned plans and evidence.

Tests cover prediction protection, conditioning on newly observed grounding frames,
action recency, snapshot restoration, malformed/future context rejection and dynamic
server callback wiring. Full-model recorded replay and simulator task success are
separate requirements; these tests establish neither.
