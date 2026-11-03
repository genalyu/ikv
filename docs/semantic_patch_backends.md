# Semantic patch indexing

DINOv3 ViT-L/16 is the raw patch backbone for visual and native tactile
indexing. Configure a complete local Transformers checkpoint using
`runtime.dino_model`. DINOv2 and its text adapter are no longer supported.

For the screenshot heuristic, use raw DINOv3 features, not prompt relevance.
The fixed screenshot formula, threshold 0.9, world-time decay 8 and 10% random
mixture are unchanged. Features remain detached retention metadata and never
enter WAN content embeddings or Q/K/V.

The NeoSim deployment explicitly uses full-field 224x224 bicubic antialiased
RGB and the checkpoint's ImageNet normalization. ViT-L/16 yields a 14x14 patch
grid (1024 channels); CLS/register tokens are excluded. Visual features are
mapped per camera onto the WAN grid; tactile features are pooled per sensor
to 4x4. Existing features must be rebuilt after changing backbone or resize.

Optional `runtime.semantic_encoder` supports `dinov3_txt` (official local,
pinned DINOv3/dino.txt code and matching backbone/head/BPE assets) and fixed
resolution `siglip2`. These prompt-relevance experiments are separate from
the raw DINOv3 screenshot configuration. NaFlex is unsupported. DINOv3 text
uses the local half of the text embedding and adapted spatial patches from
one backbone pass; text is encoded once per prompt.

All loaders default to local assets only. The raw DINOv3 loader checks
`model_type=dinov3_vit`, rejecting incompatible checkpoints. The deployment reuses the verified official ViT-L/16 checkpoint and pinned
source already present in the shared semantic-assets directory. A relative
`dinov3_encoder.json` records source revision and backbone SHA256; loading
checks both and uses strict state-dict loading without network access.

References:
- https://github.com/facebookresearch/dinov3
- https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
