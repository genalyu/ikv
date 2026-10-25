"""Score a cached packet and return native KV addresses using an explicit map."""
import argparse,json
from pathlib import Path
from .data import load_packet
from .runtime import RuntimeScorer
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True)
    p.add_argument("--packet",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--visual-budget",type=int,required=True)
    p.add_argument("--tactile-budget",type=int,default=0);p.add_argument("--kv-map")
    a=p.parse_args();packet=load_packet(a.packet)
    mapping=json.loads(Path(a.kv_map).read_text()) if a.kv_map else {}
    result=RuntimeScorer(a.checkpoint,a.device).score(packet,[a.visual_budget,a.tactile_budget],**mapping)
    result={k:v.tolist() if hasattr(v,"tolist") else v for k,v in result.items()}
    Path(a.output).write_text(json.dumps(result))
if __name__=="__main__":main()
