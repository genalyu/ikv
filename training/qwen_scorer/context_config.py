from transformers import AutoConfig
def load_context_config(model_path, max_length, factor=1.0):
 cfg=AutoConfig.from_pretrained(model_path,local_files_only=True)
 text=cfg.text_config
 native=text.max_position_embeddings
 if max_length>native:
  if factor<=1 or max_length>native*factor:
   raise ValueError("Explicit sufficient RoPE scaling factor required")
  text.rope_parameters=dict(text.rope_parameters,rope_type="yarn",factor=float(factor),original_max_position_embeddings=native)
  text.max_position_embeddings=max_length
 return cfg
