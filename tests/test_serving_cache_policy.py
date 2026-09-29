import json,os,subprocess,sys
import pytest

def config(policy=None):
    env=dict(os.environ)
    env.pop("IKV_CACHE_POLICY",None)
    if policy is not None:env["IKV_CACHE_POLICY"]=policy
    code="""import json
from n0_twam.configs.twam_posttrain_cfg import twam_posttrain_cfg as t
from n0_twam.configs.twam_posttrain_server_cfg import twam_posttrain_server_cfg as s
from n0_twam.configs.twam_multitask_server_cfg import twam_multitask_server_cfg as m
print(json.dumps([t.use_ikv_training,t.kv_cache_policy,s.kv_cache_policy,m.kv_cache_policy]))
"""
    return subprocess.run([sys.executable,"-c",code],env=env,text=True,capture_output=True)

def test_official_training_recipe_does_not_disable_ikv_serving():
    result=config()
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])==[False,"fifo","global","global"]

def test_fifo_serving_is_explicit_and_does_not_modify_training():
    result=config("fifo")
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])==[False,"fifo","fifo","fifo"]

def test_invalid_serving_policy_fails_loudly():
    result=config("typo")
    assert result.returncode!=0
    assert "IKV_CACHE_POLICY must be global or fifo" in result.stderr
