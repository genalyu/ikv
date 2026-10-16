"""Real-weight feature, cache, retention and training validation on local task RGB."""
import argparse, gc, json, statistics, sys, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from n0_twam.preprocessing.semantic_patch import load_patch_encoder
from n0_twam.preprocessing.kv_index import encode_dense_semantic, observed_index
from n0_twam.task_pipeline.features import build_payloads
from n0_twam.dataset.ikv_index import load_dense_index
from n0_twam.models.global_kv_retention import RetentionConfig, token_rows
from n0_twam.models.multimodal_kv_retention import make_retention_policy

PROMPT = "Remember which USB slot is highlighted. After the highlight disappears and the waiting period ends, insert the USB into that same slot."

def configurations(root):
    shared = dict(image_size=[224,224],dtype="bfloat16",task_temperature=.07,task_bias=0.)
    return {
        "dinov2_txt": dict(shared,backend="dinov2_txt",repo=str(root/"dinov2"),
            backbone_weights=str(root/"dinov2_vitl14_reg4_pretrain.pth"),
            head_weights=str(root/"dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth"),
            text_weights=str(root/"dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth"),
            bpe=str(root/"bpe_simple_vocab_16e6.txt.gz")),
        "dinov3_txt": dict(shared,backend="dinov3_txt",repo=str(root/"dinov3"),
            backbone_weights=str(root/"dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
            head_weights=str(root/"dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"),
            bpe=str(root/"bpe_simple_vocab_16e6.txt.gz")),
        "siglip2": dict(shared,backend="siglip2",model=str(root/"siglip2-base-patch16-224")),
    }

def frames(path):
    cap=cv2.VideoCapture(str(path)); result=[]
    for i in range(9):
        ok,frame=cap.read()
        if not ok:raise ValueError(f"video ended at {i}")
        result.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(256,256)))
    cap.release()
    return np.stack(result)

def retention_check(dense):
    cfg=RetentionConfig(version=2,video_capacity=192,action_capacity=32,tactile_capacity=32,
        task_weight=1.,class_recency_weight=1.,visual_weight=0.,persistence_weight=0.,
        query_weight=0.,action_query_weight=0.,tactile_query_weight=0.)
    p=make_retention_policy(256,"cpu",cfg)
    occupied=torch.zeros(256,dtype=torch.bool)
    differences=[]
    for frame in range(3):
        count=dense["dino_features"].shape[1]
        index=observed_index(dict(dino=dense["dino_features"][frame],
             task_relevance=dense["task_relevance"][frame]),count,"cpu")
        grid=torch.zeros(1,4,count);grid[:,0]=frame
        rows=token_rows(dict(grid_id=grid,index=index),batch_size=1,length=count,
                        main_count=count,action_mode=False,update_cache=2,device="cpu")
        p.config=replace(cfg,task_weight=0.)
        _,old_victims=p.plan(occupied,count,rows)
        p.config=cfg
        slots,victims=p.plan(occupied,count,rows)
        differences.append(len(set(old_victims.tolist())^set(victims.tolist())))
        old_mask=occupied.clone()
        p.commit(slots,rows,old_mask)
        occupied[victims]=False;occupied[slots]=True
        p.observe_dense(index["dino"],rows["world_time_id"],rows["duration"])
    return dict(retained=int(occupied.sum()),task_victim_symmetric_difference=differences,
                content_classes=int(len(p.history.features)),
                score_min=float(p.scores(occupied.nonzero().flatten()).min()))

def train_check(dense):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))
    from test_ikv_training import case
    from test_global_kv_retention import tiny_model
    from n0_twam.models.ikv_training import run_ikv_training,build_ikv_support_plan
    h,text,ts,temb,rope,mem=case()
    feat=dense["dino_features"][:2,:2]
    q=dense["task_relevance"][:2,:2]
    n=len(mem["rows"]["kind"])
    mem["rows"]["dino"]=torch.zeros(n,feat.shape[-1])
    mem["rows"]["dino"][:4]=feat.flatten(0,1)
    mem["rows"]["dino"][4:8]=feat.flatten(0,1)
    mem["rows"]["task_relevance"]=torch.zeros(n)
    mem["rows"]["task_relevance"][:4]=q.flatten()
    mem["rows"]["task_relevance"][4:8]=q.flatten()
    mem["dense_dino_features"]=feat[None]
    mem["config"].update(capacity=6,retention=dict(version=2,
        video_capacity=2,action_capacity=2,tactile_capacity=2,task_weight=1.,
        class_recency_weight=1.,visual_weight=0.,persistence_weight=0.,
        query_weight=0.,action_query_weight=0.,tactile_query_weight=0.))
    model=tiny_model(False).mot.train()
    out=run_ikv_training(model,h,text,ts,temb,rope,mem)
    loss=out.square().mean();loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    build_ikv_support_plan(mem,"cpu")
    return dict(loss=float(loss.detach()),gradient_norm=float(h.grad.norm()))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    rgb={key:frames(args.input/key/"episode_000000.mp4") for key in
         ("observation.images.top","observation.images.wrist_l")}
    report=dict(input=str(args.input),prompt=PROMPT,
                gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),backends={})
    for backend,cfg in configurations(args.model_root).items():
        print("loading",backend,flush=True)
        start=time.perf_counter()
        encoder=load_patch_encoder(cfg,device=args.device,prompt=PROMPT)
        load_s=time.perf_counter()-start
        batch=torch.from_numpy(np.stack([x[0] for x in rgb.values()])).permute(0,3,1,2).to(args.device)
        encoder(batch)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        latency=[]
        for _ in range(5):
            start=time.perf_counter();output=encoder(batch);torch.cuda.synchronize()
            latency.append((time.perf_counter()-start)*1000)
        original=output.task_relevance.clone()
        encoder.set_prompt("A red ball is hidden under a cup.")
        changed=encoder(batch).task_relevance
        contrast=float((changed-original).abs().mean())
        assert contrast>1e-6,"prompt does not affect relevance"
        encoder.set_prompt(PROMPT)
        motion,dense=build_payloads({k:iter(v) for k,v in rgb.items()},list(range(9)),[0,4,8],encoder)
        videos=torch.stack([torch.from_numpy(x).permute(3,0,1,2).float()/255 for x in rgb.values()])
        online=encode_dense_semantic(videos,[0,4,8],(8,8),encoder)
        torch.testing.assert_close(online["task_relevance"].cpu(),dense["task_relevance"].flatten(),rtol=2e-3,atol=2e-3)
        path=args.out/backend;path.mkdir(exist_ok=True)
        torch.save(dense,path/"dense.pth");torch.save(motion,path/"motion.pth")
        loaded=load_dense_index(path/"dense.pth",camera_keys=list(rgb),patch_size=[1,2,2],
            grid_shape=[8,16],latent_frame_ids=list(range(9)),full_frames=3,
            semantic_expected=encoder.provenance(),require_task=True)
        torch.testing.assert_close(loaded["task_relevance"],dense["task_relevance"])
        result=dict(load_s=load_s,native_grid=list(output.grid_size),feature_dim=output.tokens.shape[-1],
            parameters=sum(p.numel() for p in encoder.parameters()),dtype=cfg["dtype"],
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            two_camera_forward_median_ms=statistics.median(latency),
            task_min=float(dense["task_relevance"].min()),task_max=float(dense["task_relevance"].max()),
            prompt_contrast_mean_abs=contrast,retention=retention_check(dense),
            training=train_check(dense),provenance=encoder.provenance())
        report["backends"][backend]=result
        (path/"encoder.json").write_text(json.dumps(cfg,indent=2))
        (args.out/"report.json").write_text(json.dumps(report,indent=2))
        print(backend,json.dumps({k:v for k,v in result.items() if k!="provenance"}),flush=True)
        del encoder,output,changed,batch;gc.collect();torch.cuda.empty_cache()
    report["status"]="passed"
    (args.out/"report.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
