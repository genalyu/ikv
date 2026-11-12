import math, os, unittest
from pathlib import Path
import numpy as np
import torch
from ikv_runtime import Candidate, Region
from ikv_runtime.newformula import FormulaBridge, make_scorer

SOURCE=os.environ.get("IKV_CORE_SOURCE",str(Path(__file__).resolve().parents[2]))
class Encoder:
    revision="test"
    def __init__(self):self.calls=0
    def image(self,rgb):
        self.calls+=1
        return torch.as_tensor(rgb[0,0,:2].copy()).float().reshape(1,1,2)
def rgb(a=255,b=0):return np.full((2,2,3),[a,b,0],dtype=np.uint8)
def candidate(frame,t):return Candidate(frame,t,(Region(frame,0),))
@unittest.skipUnless((Path(SOURCE)/"n0_twam/models/heuristic_score.py").is_file(),"canonical source required")
class FormulaTests(unittest.TestCase):
    def bridge(self):return FormulaBridge(SOURCE,world_time_hz=25,encoder=Encoder())
    def test_world_steps_raw_unit_sum_causal(self):
        b=self.bridge()
        b.register_frame("a",0,{0:rgb()})
        b.register_frame("b",8/25,{0:rgb()})
        self.assertEqual(b.score([candidate("a",0)],"",0)["a"],3.)
        value=b.score([candidate("a",0),candidate("b",8/25)],"",8/25)
        self.assertAlmostEqual(value["a"],1+2*math.exp(-1),places=6)
        self.assertEqual(value["b"],3.)
        self.assertEqual(value,b.score([candidate("a",0),candidate("b",8/25)],"different",8/25,subtask="ignored"))
    def test_contact_term_and_zero_feature(self):
        b=self.bridge();b.register_frame("a",0,{0:rgb()},contact_duration={0:8})
        self.assertAlmostEqual(b.score([candidate("a",0)],"",0)["a"],2+math.exp(-1),places=6)
        b.register_frame("zero",.1,{0:rgb(0,0)})
        self.assertEqual(b.score([candidate("zero",.1)],"",.1)["zero"],2.)
    def test_retry_isolation_and_reset(self):
        b=self.bridge();b.register_frame("a",0,{0:rgb()});b.register_frame("a",0,{0:rgb()})
        self.assertEqual(b.encoded_images,1)
        with self.assertRaises(ValueError):b.register_frame("a",0,{0:rgb()},contact_duration={0:1})
        fork=b.fork();self.assertIs(fork.encoder,b.encoder);self.assertEqual(fork.registered_frames,())
        b.reset();self.assertEqual(b.registered_frames,())
    def test_no_future_or_missing_provenance(self):
        b=self.bridge();b.register_frame("a",1,{0:rgb()})
        with self.assertRaises(ValueError):b.score([candidate("a",1)],"",0)
        with self.assertRaises(ValueError):b.score([candidate("missing",1)],"",1)
    def test_prototypes_use_threshold_and_latest_real_time(self):
        b=self.bridge();b.register_frame("a",0,{0:rgb(255,0)})
        # cosine .8 matches .75 gate, unlike the removed .9 gate.
        b.register_frame("b",8/25,{0:rgb(204,153)})
        v=b.score([candidate("a",0)],"",8/25)["a"]
        self.assertAlmostEqual(v,1+2*math.exp(-1),places=6)
    def test_area_mapping_arbitrary_dino_grid(self):
        grid=torch.tensor([[1.,2.],[3.,4.]])
        self.assertEqual(FormulaBridge._region_score(grid,(0,0,.5,.5)),1.)
        self.assertEqual(FormulaBridge._region_score(grid,(0,0,1,1)),2.5)
    def test_cached_prototypes_rebuild_for_late_history_and_earlier_query(self):
        b=self.bridge();b.register_frame("a",0,{0:rgb()})
        b.register_frame("c",.64,{0:rgb(0,255)})
        b.score([candidate("a",0)],"",.64)
        b.register_frame("b",.32,{0:rgb(204,153)})
        rebuilt=b.score([candidate("a",0),candidate("b",.32)],"",.64)
        fresh=self.bridge()
        for name,t,pixels in [("a",0,rgb()),("b",.32,rgb(204,153)),("c",.64,rgb(0,255))]:
            fresh.register_frame(name,t,{0:pixels})
        self.assertEqual(rebuilt,fresh.score([candidate("a",0),candidate("b",.32)],"",.64))
        self.assertEqual(b.score([candidate("a",0)],"",0)["a"],3.)
        self.assertEqual(rebuilt,b.score([candidate("a",0),candidate("b",.32)],"",.64))
    def test_reject_fake_planner_interaction(self):
        with self.assertRaisesRegex(ValueError,"Planner"):
            make_scorer({"backend":"formula","planner_mode":"interactive"},world_time_hz=25)
if __name__=="__main__":unittest.main()
