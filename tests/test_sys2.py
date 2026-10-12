import json
import pytest
import torch
from n0_twam.preprocessing.sys2 import binary_contact, validate_scores, Sys2Client


def candidate(uid="1", modality="tactile"):
    return dict(id=uid, time=1., modality=modality, frame=123, patch=0, grid=[2,2], contact=True)


def test_binary_contact_rejects_missing_uncertain_and_scores():
    for value in (None, -1, .5, float("nan"), "unknown"):
        with pytest.raises(ValueError): binary_contact(value, 3)
    assert binary_contact(True, 3).tolist() == [True]*3
    assert binary_contact([1,0,1],3).tolist() == [True,False,True]


def test_scores_require_complete_independent_modalities():
    cs = [candidate(), candidate("2", "visual")]
    good = dict(visual_scores=[dict(id="2",score=91)], tactile_scores=[dict(id="1",score=52)])
    assert validate_scores(json.dumps(good), cs) == {"1":52,"2":91}
    for output in (
        dict(good,tactile_scores=[]),
        dict(good,tactile_scores=[dict(id="1",score=True)]),
        dict(good,tactile_scores=[dict(id="1",score=52.0)]),
        dict(good,tactile_scores=[dict(id="1",score=101)]),
        dict(good,tactile_scores=[dict(id="2",score=52)]),
        dict(good,tactile_scores=good["tactile_scores"]*2),
        dict(good,reason="extra"),
    ):
        with pytest.raises(ValueError): validate_scores(output,cs)
    with pytest.raises(ValueError):
        validate_scores('{"visual_scores": [], "visual_scores": [], "tactile_scores": []}', [])


def test_visual_and_tactile_layout_and_uncontacted_features_preserved():
    client = Sys2Client("http://localhost:1", "task")
    client.encode = lambda image: int(image[0,0,0])
    videos = torch.stack([torch.full((3,2,4,4), n) for n in (11,22)])
    visual = client.index(videos, [0,1], (2,2))
    assert visual["sys2_frame"].tolist() == [11,11,22,22,11,11,22,22]*2
    tactile = client.index(videos,[0,1],(2,2),tactile=True,contact=False)
    assert tactile["sys2_frame"].tolist() == [11]*8+[22]*8
    assert not tactile["contact_present"].any()
    assert (tactile["sys2_frame"] >= 0).all()


def test_archive_callback_includes_incoming_and_history():
    client = Sys2Client("http://localhost:1", "task")
    seen=[]
    def score(candidates, time):
        seen.extend(candidates)
        return {c["id"]: 77 for c in candidates}
    client.score=score
    def rows(kinds):
        n=len(kinds)
        return dict(kind=torch.tensor(kinds), observation_flag=torch.ones(n,dtype=torch.bool),
                    token_uid=torch.arange(n),world_time_id=torch.ones(n),
                    sys2_frame=torch.ones(n,dtype=torch.long),sys2_patch=torch.zeros(n,dtype=torch.long),
                    sys2_height=torch.ones(n,dtype=torch.long),sys2_width=torch.ones(n,dtype=torch.long),
                    contact_present=torch.zeros(n,dtype=torch.bool))
    result=client(rows([0,1,2]),rows([0,2]),3)
    assert [c["id"] for c in seen] == ["0","2","3","4"]
    assert result.tolist() == [77,0,77]


@torch.no_grad()
def test_sys2_scores_recall_history_and_failure_rolls_back():
    from test_cpu_kv_archive import model, policy
    from test_global_kv_retention import model_input, cache_snapshot, assert_cache_equal
    m=model(score_backend="qwen_sys2",sys2_endpoint="http://localhost:1",sys2_task="remember target")
    p=policy(m)
    requests=[]
    def score(candidates,time):
        requests.append(candidates)
        return {c["id"]: (100 if c["id"]=="0" else 1) for c in candidates}
    p.sys2_scorer.score=score
    def ground(t):
        packet=model_input(t)
        index=dict(sys2_frame=torch.full((4,),t+1,dtype=torch.long),sys2_patch=torch.arange(4),
                   sys2_height=torch.full((4,),2,dtype=torch.long),sys2_width=torch.full((4,),2,dtype=torch.long))
        packet["kv_index"]=index
        packet["tactile_kv_index"]=dict(index,contact_present=torch.ones(4,dtype=torch.bool))
        return m(packet,update_cache=2,cache_name="test")
    for t in range(5):ground(t)
    active=m.get_global_retention("test")
    assert 0 in active["token_uid"].tolist()
    assert len(requests[-1])==40
    assert p.archive.stats["archived_tokens"]==40
    before=cache_snapshot(m);uid=p.next_uid
    def invalid(candidates,time):raise ValueError("bad JSON")
    p.sys2_scorer.score=invalid
    with pytest.raises(ValueError,match="bad JSON"):ground(6)
    assert_cache_equal(before,cache_snapshot(m))
    assert p.next_uid==uid and p.archive.stats["archived_tokens"]==40


def test_only_complete_json_wrapper_is_removed():
    from n0_twam.preprocessing.sys2 import unwrap_json_fence
    raw='{"visual_scores":[],"tactile_scores":[]}'
    assert unwrap_json_fence("```json\n"+raw+"\n```")==raw
    for bad in ("Explanation: "+raw, "```json\n"+raw, raw+" extra"):
        with pytest.raises(ValueError): validate_scores(unwrap_json_fence(bad),[])
