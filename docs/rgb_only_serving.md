RGB-only inference
==================
Serving with an explicit empty tactile_keys list marks video and action model inputs rgb_only. This follows the RGB-only training sequence: no tactile embeddings, tokens, RoPE positions, or timestep tails are created. Local tactile heads and contact gates must be disabled; conflicting tactile payloads are rejected. Legacy inputs without the flag retain the strict tactile contract. Training is unchanged.

Validation: tests/test_rgb_only_inference.py covers finite video/action output without invoking tactile encoding, rejection of conflicting payloads, and legacy missing-tactile rejection. Combined with tests/test_mot_sparse_padding.py: 20 passed. The first combined run hit an existing uninitialized cache-slot NaN comparison; unchanged rerun passed all 20. Hardware checkpoint and closed-loop integration are separate deployment gates.
