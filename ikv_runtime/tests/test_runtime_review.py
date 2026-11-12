"""Independent behavioral checks for native KV retention and scorer provenance.

Run on the actual server with unittest; these fixtures never load model weights.
"""
import copy
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import torch
from torch import nn
from ikv_runtime import ArchiveBank, Candidate, Region, ScorerBridge


def candidate(uid, time=1., frame=None, sensor=0, box=(0.,0.,1.,1.)):
    return Candidate(uid=uid, time=time,
                     regions=(Region(frame_id=frame or uid, sensor=sensor, box=box),),
                     provenance="review-fixture")


def pair(values, layer=0, dtype=torch.float32, seq_dim=1):
    rows=torch.tensor([[float(x)+100*layer, float(x)+.5+100*layer] for x in values], dtype=dtype)
    rows=rows.reshape(len(values),1,1,2)
    key=rows.movedim(0,seq_dim)
    return key, key+1000


def commit(bank, fresh, values, budget=1, scores=None, selected_ids=None):
    tx=bank.begin(fresh,budget,scores=scores,selected_ids=selected_ids)
    results=[]
    for layer in range(bank.expected_layers):
        k,v=pair(values,layer)
        results.append(tx.layer(layer,k,v))
    tx.commit()
    return results


def retrieve(bank, selected):
    tx=bank.begin([],len(selected),selected_ids=selected)
    result=[]
    for layer in range(bank.expected_layers):
        result.append(tx.layer(layer,*pair([],layer)))
    tx.abort()
    return result


class ArchiveReview(unittest.TestCase):
    def test_dropped_resident_can_be_retrieved_from_complete_archive(self):
        bank=ArchiveBank(2)
        a,b,c=candidate("a",1),candidate("b",2),candidate("c",3)
        first=commit(bank,[a,b],[10,20],scores={"a":0,"b":1})
        self.assertEqual(first[0][0].flatten().tolist(),[20.,20.5])
        seen=[]
        tx=bank.begin([c],1,score_fn=lambda union:(seen.append(tuple(x.uid for x in union)) or {"a":9,"b":0,"c":1}))
        for layer in range(2):
            actual=tx.layer(layer,*pair([30],layer))
            self.assertTrue(torch.equal(actual[0],pair([10],layer)[0]))
        tx.commit()
        self.assertEqual(seen,[("a","b","c")])
        self.assertEqual(bank.candidate_ids,("a","b","c"))

    def test_refresh_replaces_payload_but_preserves_unique_provenance(self):
        bank=ArchiveBank(2)
        a=candidate("a")
        commit(bank,[a],[10])
        commit(bank,[a],[40])
        self.assertEqual(bank.candidate_ids,("a",))
        self.assertTrue(torch.equal(retrieve(bank,["a"])[1][0],pair([40],1)[0]))
        with self.assertRaises(ValueError):
            bank.begin([candidate("a",2)],1)

    def test_budget_padding_has_valid_mask_and_is_never_archived(self):
        bank=ArchiveBank(2)
        outputs=commit(bank,[candidate("a")],[10],budget=4)
        self.assertEqual(bank.candidate_ids,("a",))
        for key,value,valid in outputs:
            self.assertEqual(valid.tolist(),[True,False,False,False])
            self.assertEqual(tuple(key.shape),(1,4,1,2))
            self.assertTrue(torch.equal(key[:,1:],torch.zeros_like(key[:,1:])))
            self.assertTrue(torch.equal(value[:,1:],torch.zeros_like(value[:,1:])))
        self.assertEqual(bank.bytes,32)

    def test_zero_budget_still_archives_candidates_for_later_selection(self):
        bank=ArchiveBank(1)
        outputs=commit(bank,[candidate("a")],[12],budget=0)
        self.assertEqual(outputs[0][0].shape[1],0)
        self.assertEqual(bank.candidate_ids,("a",))
        self.assertTrue(torch.equal(retrieve(bank,["a"])[0][0],pair([12])[0]))

    def test_storage_is_detached_cloned_and_independent_from_input_mutation(self):
        bank=ArchiveBank(1)
        k,v=pair([11]);k.requires_grad_();v.requires_grad_()
        tx=bank.begin([candidate("a")],1)
        tx.layer(0,k,v);tx.commit()
        with torch.no_grad():k.fill_(99);v.fill_(99)
        out=retrieve(bank,["a"])[0]
        self.assertTrue(torch.equal(out[0],pair([11])[0]))
        self.assertFalse(out[0].requires_grad)

    def test_missing_layer_failed_commit_is_atomic(self):
        bank=ArchiveBank(2)
        commit(bank,[candidate("a")],[11])
        before=(bank.generation,bank.bytes,bank.candidate_ids)
        tx=bank.begin([candidate("b",2)],1,scores={"a":0,"b":1})
        tx.layer(0,*pair([22]))
        with self.assertRaises(RuntimeError):tx.commit()
        tx.abort()
        self.assertEqual(before,(bank.generation,bank.bytes,bank.candidate_ids))
        self.assertTrue(torch.equal(retrieve(bank,["a"])[1][0],pair([11],1)[0]))

    def test_nonfinite_layer_rejected_without_publication(self):
        bank=ArchiveBank(2)
        commit(bank,[candidate("a")],[11])
        tx=bank.begin([candidate("b",2)],1,scores={"a":0,"b":1})
        tx.layer(0,*pair([22]))
        k,v=pair([22],1);v.fill_(float("nan"))
        with self.assertRaises(ValueError):tx.layer(1,k,v)
        tx.abort()
        self.assertEqual(bank.candidate_ids,("a",))
        self.assertTrue(torch.equal(retrieve(bank,["a"])[0][0],pair([11])[0]))

    def test_archive_quota_failure_keeps_previous_generation(self):
        bank=ArchiveBank(2,max_bytes=64)
        commit(bank,[candidate("a")],[11])
        before=(bank.generation,bank.bytes)
        tx=bank.begin([candidate("b",2),candidate("c",3)],1,scores={"a":0,"b":1,"c":2})
        for layer in range(2):tx.layer(layer,*pair([22,33],layer))
        with self.assertRaises(MemoryError):tx.commit()
        tx.abort()
        self.assertEqual(before,(bank.generation,bank.bytes))
        self.assertEqual(bank.candidate_ids,("a",))

    def test_concurrent_stale_transaction_cannot_overwrite_commit(self):
        bank=ArchiveBank(1)
        t1=bank.begin([candidate("a")],1)
        t2=bank.begin([candidate("b")],1)
        t1.layer(0,*pair([11]));t2.layer(0,*pair([22]))
        t1.commit()
        with self.assertRaises(RuntimeError):t2.commit()
        self.assertEqual(bank.candidate_ids,("a",))

    def test_reset_invalidates_pending_transaction_and_clears_archive(self):
        bank=ArchiveBank(1)
        commit(bank,[candidate("a")],[11])
        tx=bank.begin([],1,selected_ids=["a"])
        tx.layer(0,*pair([]));bank.reset()
        with self.assertRaises(RuntimeError):tx.commit()
        self.assertEqual(bank.candidate_ids,())
        self.assertEqual(bank.bytes,0)

    def test_selection_requires_all_archived_scores_and_no_fallback(self):
        bank=ArchiveBank(1)
        commit(bank,[candidate("a")],[11])
        with self.assertRaises(ValueError):bank.begin([candidate("b",2)],1)
        with self.assertRaises(ValueError):bank.begin([candidate("b",2)],1,scores={"b":1})
        with self.assertRaises(ValueError):bank.begin([candidate("b",2)],1,scores={"a":float("nan"),"b":1})
        self.assertEqual(bank.candidate_ids,("a",))

    def test_explicit_selection_and_payload_geometry_validation(self):
        bank=ArchiveBank(1)
        a,b=candidate("a"),candidate("b",2)
        with self.assertRaises(ValueError):bank.begin([a,b],2,selected_ids=["a"])
        with self.assertRaises(ValueError):bank.begin([a],1,selected_ids=["missing"])
        with self.assertRaises(ValueError):bank.begin([a,a],1)
        tx=bank.begin([a],1)
        with self.assertRaises(ValueError):tx.layer(0,*pair([1,2]))
        tx.abort()
        commit(bank,[a],[11])
        tx=bank.begin([b],1,scores={"a":0,"b":1})
        with self.assertRaises(ValueError):tx.layer(0,*pair([22],dtype=torch.float64))
        tx.abort()

    def test_nondefault_sequence_dimension_is_preserved(self):
        bank=ArchiveBank(1)
        tx=bank.begin([candidate("a"),candidate("b",2)],1,scores={"a":1,"b":0})
        key,value,mask=tx.layer(0,*pair([11,22],seq_dim=2),seq_dim=2)
        self.assertEqual(tuple(key.shape),(1,1,1,2))
        self.assertTrue(torch.equal(key,pair([11],seq_dim=2)[0]))
        tx.commit()


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)
        self.revision="review-r1"
        self.image_batches=[]
        self.texts=[]
        self.fail=False
    def images(self,images):
        self.image_batches.append(len(images))
        rows=[]
        for image in images:
            color=float(np.asarray(image)[0,0,0])
            patch=torch.arange(64,dtype=torch.float32)+color
            rows.append(torch.stack((patch,torch.zeros_like(patch)),-1))
        out=torch.cat(rows)
        if self.fail:out.fill_(float("nan"))
        return out
    def task(self,text):
        self.texts.append(text)
        return torch.tensor([[float(len(text)),0.]])


class FakeRuntime:
    def __init__(self):
        self.info={"encoder_revision":"review-r1"}
        self.model=nn.Identity().eval()
        self.packets=[]
    def score(self,packet,budgets):
        self.packets.append((packet,budgets))
        return {"scores":packet["features"][:,0]+packet["text"][0,0]}


def rgb(value):
    return np.full((10,12,3),value,dtype=np.uint8)


class ScorerReview(unittest.TestCase):
    def setUp(self):
        self.encoder=FakeEncoder()
        self.runtime=FakeRuntime()
        self.bridge=ScorerBridge(encoder=self.encoder,runtime=self.runtime)

    def test_identical_frames_encode_once_and_mutated_identity_is_rejected(self):
        self.bridge.register_frame("f",1,{0:rgb(10),1:rgb(20)})
        self.bridge.register_frame("f",1,{0:rgb(10),1:rgb(20)})
        self.assertEqual(self.encoder.image_batches,[2])
        self.assertEqual(self.bridge.encoded_images,2)
        with self.assertRaises(ValueError):self.bridge.register_frame("f",1,{0:rgb(11)})
        with self.assertRaises(ValueError):self.bridge.register_frame("f",2,{0:rgb(10)})

    def test_fractional_spatial_overlap_uses_area_weights(self):
        self.bridge.register_frame("f",1,{0:rgb(10)})
        full=candidate("full",frame="f")
        fractional=candidate("fractional",frame="f",box=(0.,0.,1.,3/16))
        result=self.bridge.score([full,fractional],"go",1)
        self.assertAlmostEqual(result["full"],43.5,places=6)
        self.assertAlmostEqual(result["fractional"],40+1/3,places=6)

    def test_sensor_and_temporal_regions_are_not_flattened_together(self):
        self.bridge.register_frame("early",1,{0:rgb(10),1:rgb(100)})
        self.bridge.register_frame("late",2,{0:rgb(20)})
        c=Candidate("mixed",2,(Region("early",1,weight=1.),Region("late",0,weight=3.)))
        result=self.bridge.score([c],"go",2)
        self.assertAlmostEqual(result["mixed"],(133.5+3*53.5)/4,places=6)

    def test_future_frames_excluded_from_context_and_candidates(self):
        self.bridge.register_frame("past",1,{0:rgb(10)})
        self.bridge.register_frame("future",2,{0:rgb(99)})
        result=self.bridge.score([candidate("a",1,frame="past")],"go",1)
        self.assertAlmostEqual(result["a"],43.5)
        packet,_=self.runtime.packets[-1]
        self.assertEqual(len(packet["features"]),64)
        self.assertTrue(bool((packet["times"]<=1).all()))
        with self.assertRaises(ValueError):self.bridge.score([candidate("bad",2,frame="future")],"go",1)
        with self.assertRaises(ValueError):self.bridge.score([candidate("bad",1,frame="future")],"go",2)
        with self.assertRaises(ValueError):self.bridge.score([candidate("bad",1,frame="missing")],"go",2)

    def test_feedback_rescores_cached_visual_features(self):
        self.bridge.register_frame("f",1,{0:rgb(10)})
        c=candidate("a",frame="f")
        first=self.bridge.score([c],"go",1)["a"]
        second=self.bridge.score([c],"go",1,subtask="pick",memory="seen red")["a"]
        self.assertGreater(second,first)
        self.assertEqual(self.encoder.image_batches,[1])
        self.assertEqual(self.bridge.score_calls,2)
        self.assertIn("Current subtask: pick",self.encoder.texts[-1])
        self.assertIn("Semantic memory: seen red",self.encoder.texts[-1])

    def test_fork_shares_models_but_not_episode_memory(self):
        self.bridge.register_frame("f",1,{0:rgb(10)})
        other=self.bridge.fork()
        self.assertIs(other.encoder,self.bridge.encoder)
        self.assertIs(other.runtime,self.bridge.runtime)
        self.assertEqual(other.registered_frames,())
        other.register_frame("f",1,{0:rgb(20)})
        self.assertAlmostEqual(self.bridge.score([candidate("a",frame="f")],"go",1)["a"],43.5)
        self.assertAlmostEqual(other.score([candidate("a",frame="f")],"go",1)["a"],53.5)
        other.reset()
        self.assertEqual(self.bridge.registered_frames,("f",))

    def test_encoder_failure_does_not_publish_partial_frame(self):
        self.encoder.fail=True
        with self.assertRaises(ValueError):self.bridge.register_frame("bad",1,{0:rgb(10),1:rgb(20)})
        self.assertEqual(self.bridge.registered_frames,())
        self.assertEqual(self.bridge.encoded_images,0)
        self.encoder.fail=False
        self.bridge.register_frame("bad",1,{0:rgb(10)})
        self.assertEqual(self.bridge.registered_frames,("bad",))

    def test_model_revision_and_input_validation_fail_closed(self):
        wrong=FakeEncoder();wrong.revision="other"
        with self.assertRaises(ValueError):ScorerBridge(encoder=wrong,runtime=self.runtime)
        with self.assertRaises(ValueError):ScorerBridge(encoder=self.encoder)
        with self.assertRaises(ValueError):self.bridge.register_frame("f",1,{0:np.zeros((2,2,3),dtype=np.float32)})
        with self.assertRaises(ValueError):Region("f",0,(0.,0.,0.,1.))
        with self.assertRaises(ValueError):Candidate("a",1,())
        with self.assertRaises(ValueError):Region("f",0,weight=float("nan"))


class ImportReview(unittest.TestCase):
    def test_preloaded_foreign_leaf_modules_are_rejected(self):
        from ikv_runtime.scorer import _import_source
        with tempfile.TemporaryDirectory(prefix="ikv-import-review-") as directory:
            root=Path(directory)
            package=root/"training"/"parallel_scorer"
            package.mkdir(parents=True)
            (package/"runtime.py").write_text("class RuntimeScorer: pass")
            (package/"encoders.py").write_text("class QwenEncoders: pass")
            parent=types.ModuleType("training");parent.__path__=[str(root/"training")]
            child=types.ModuleType("training.parallel_scorer");child.__path__=[str(package)]
            foreign_runtime=types.ModuleType("training.parallel_scorer.runtime")
            foreign_runtime.__file__="/different/deployment/training/parallel_scorer/runtime.py"
            foreign_runtime.RuntimeScorer=object
            foreign_encoder=types.ModuleType("training.parallel_scorer.encoders")
            foreign_encoder.__file__="/different/deployment/training/parallel_scorer/encoders.py"
            foreign_encoder.QwenEncoders=object
            modules={"training":parent,"training.parallel_scorer":child,
                     "training.parallel_scorer.runtime":foreign_runtime,
                     "training.parallel_scorer.encoders":foreign_encoder}
            original_path=list(sys.path)
            try:
                with patch.dict(sys.modules,modules):
                    with self.assertRaises(RuntimeError):_import_source(root)
            finally:
                sys.path[:]=original_path


if __name__=="__main__":
    unittest.main()
