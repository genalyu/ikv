"""Translate saved per-rank batch position when changing data-parallel world size."""
import json
from pathlib import Path
def restored_batch_index(state,resume_path,new_world_size):
 old_world=state.get("world_size")
 if old_world is None:
  metadata=Path(resume_path).parent/"run_config.json"
  if not metadata.exists():raise RuntimeError("Checkpoint world size unknown; refusing ambiguous data resume")
  old_world=json.loads(metadata.read_text())["world_size"]
 consumed=state["next_batch"]*old_world
 if consumed%new_world_size:raise RuntimeError("Checkpoint sample position is not divisible by new world size")
 return consumed//new_world_size
