# RGB motion sparse-token path

本文说明 RGB motion v1 相对原版 N0-TWAM 的改动、sidecar 数据契约，以及训练和在线服务的数据流。该版本先实现二维 RGB/WAN patch 稀疏化；它不是最终的 3D 点云世界模型。

## 1. 核心设计

RGB motion v1 保留原版 N0-TWAM 的内容表示和三个 expert，只在 video expert 的输入前增加稀疏 patch 选择：

```text
RGB ──WAN VAE──> dense WAN latent
                         │
motion_indices ──────────┤ SparsePatchGather（在原投影之前）
                         ▼
              selected raw WAN patches
                         │
              原 patch_embedding_mlp
                         ▼
                 Video Expert ─┐
Tactile ───原 tactile embedding├─ shared attention ─> 原输出头
Action ──────原 action embedding┘

{world_time_id, DINO, NeoForce, observation_flag}
                         └────> 独立的 cache slot metadata
```

因此有两类数据，不能混为一谈：

- **content**：原来的 WAN latent、触觉 latent 和 action，经原来的 projection 得到 hidden state，并由它们产生 Q/K/V。
- **index**：`{world_time_id, DINO, NeoForce, observation_flag}`，用于描述和管理一个稀疏 video token 对应的世界状态，独立存储在 KV 槽位旁。

`DINO` 和 `NeoForce` 不与 `t` 拼接，不彼此相加，也不投影后加进 hidden state。当前实现没有 `IndexProjection`，更没有“零初始化的 IndexProjection”。这意味着旧 checkpoint 不需要为 index 加载任何新权重；但启用稀疏选择本身仍会改变送入 video expert 的 token 集合。

## 2. 原版 content embedding 包含什么

RGB motion 没有替换以下原版表示：

| 模态/条件 | 原版表示 |
|---|---|
| Video | WAN VAE latent 按 `patch_size` patchify；每个原始 latent patch 进入原 `patch_embedding_mlp` 线性层 |
| Global tactile | 触觉 WAN latent 进入共享的 `tactile_patch_embed`，再加 `sensor_id_embed` |
| Local tactile（可选） | `local_tactile_patch_embed`，再加独立的 sensor、frame、height、width embedding，经 norm 后作为 action 的局部触觉 cross-attention 条件 |
| Action | action vector 进入原 `action_embedder` 线性层 |
| Text | umT5 embedding 经原 `PixArtAlphaTextProjection`，供 video/action text cross-attention 使用 |

此外，原模型仍使用：

- video/action/global-tactile 的时空 grid id 和 RoPE；
- diffusion timestep embedding 与 AdaLN modulation；
- expert 类型和既有的 attention mask。

这些是位置、扩散步和模态条件，不是新语义 index。尤其要区分：

- `world_time_id`：token 在 episode/stream 的 WAN latent 时间轴上的 step；
- `timesteps`：flow matching 的扩散噪声步。

二者用途不同，不能互换。

## 3. 独立 index 和 presence mask

每个有效的稀疏 video token 的正式 index 是：

```text
i = {world_time_id, DINO, NeoForce, observation_flag}
```

- `world_time_id`：该 token 在 episode/stream 的 **WAN latent 时间轴**上的序号；每增加 1 表示前进一个 WAN latent step。它不是原始 RGB/depth 的 source-frame id，也不是 flow-matching 的 diffusion timestep。
- `DINO`：该位置的视觉语义特征。
- `NeoForce`：该位置的触觉语义特征；纯 RGB 数据可以是宽度为 0 的空特征。
- `observation_flag`：`1` 表示实际观测，`0` 表示模型预测。

另有三个 mask：

- `motion_valid_mask`：这一行是否是真 token；`False` 表示为了 batch 对齐而填充的空位。
- `visual_valid`：这一行是否真的具有可用 DINO 特征。
- `tactile_valid`：这一行是否真的具有可用 NeoForce 特征。

`visual_valid` 和 `tactile_valid` 只是 **presence metadata**，不是 index 的第五、第六个分量。必须保留它们，因为全零向量可能表示“缺少该模态/填充”，也可能是特征计算的合法数值；只看特征值无法可靠区分这两种情况。每个 `motion_valid_mask=True` 的 token 至少要满足 `visual_valid` 或 `tactile_valid` 之一。

模型使用 `motion_indices` 选择 content patch，并使用 `motion_valid_mask` 屏蔽 padding。语义 index 会随对应 video token 写入 shared-attention KV cache 的独立 `semantic` sidecar，但不会加到、拼到或覆盖 K/V 数值；未建立 RGB index 的 tactile/action token 仍按原路径工作。

MoT 模型提供只读检查接口：

```python
snapshot = model.get_semantic_cache(
    cache_name, layer=0, valid_only=True
)
```

存在对应 sidecar 时，它返回 `slot_indices`、`valid`、`world_time_id`、`dino`、`neoforce`、`observation_flag`、`visual_valid` 和 `tactile_valid`；cache/sidecar 尚未初始化时返回 `None`。默认 `valid_only=True` 只返回有效语义行，因而跳过无 RGB index 的 tactile/action KV 行；设为 `False` 则返回 sidecar 的完整容量，并由 `valid` 指明哪些 slot 具有有效的 RGB 语义 index。这里的 `valid` 不是物理 KV occupancy：活跃但无 RGB index 的 tactile/action KV slot 仍会是 `False`。所有返回 tensor 都是 detached clone，修改 snapshot 不会改变 live cache。该 API 不返回 K/V，也不能通过它修改 K/V；它只是对当前已接线 metadata sidecar 的观测接口，不代表 standalone `SemanticKVCache` 的匹配或压缩策略已经接入 streaming attention。

streaming cache 的 committed 写入是事务性的：单层 attention/backend 报错会恢复该层覆盖或淘汰的槽位；后续 layer 或输出 head 报错会恢复本次 model forward 已写入的所有层。server 的一次 grounding 又从 `clear_pred_cache` 之前开始一个更外层事务，并把 video pass 与 action pass 都放在其中，因此 action pass 失败不会留下“旧 prediction 已清掉、但只有 video 新 KV 已提交”的半状态。普通/冷启动 imagination 同样把最终 video 与 action 的 `update_cache=1` 写入放在一个请求事务里，事务一直覆盖到 action postprocess 成功；因此不支持的输出格式等晚期错误也不会留下半个预测 chunk。回滚同时覆盖 K、V、slot id、occupancy、predicted 标志和独立 semantic sidecar；实现只暂存本次触及的槽位，而不会复制整个 cache pool。

grounding 和 cold imagination 还对 video/global-tactile/local-tactile 三个 streaming VAE 的 `feat_cache` 容器做轻量快照，并覆盖 `init_latent`、`last_tactile_latents`、最近 observed/generated tactile/video 状态、delta smoothing 状态、tactile 首帧/前帧以及 RGB index episode state。WAN causal encoder 会替换 cache entry，而不会原地修改旧 tensor，所以快照只复制很小的 Python 容器并保留旧 tensor 引用，不会把三套大 feature cache 再复制一份到 GPU。任一 preprocess、encode、shape check、video/action model pass 或 action postprocess 失败后，旧 VAE 时间上下文和 episode 字段都会恢复，可以用同一个 observation 安全重试。

## 4. Motion Detector 与网格映射

`EgoMotionCompensatedMotionDetector` 的输入是相邻两帧的：

```text
DINO_previous, DINO_current
Depth_previous, Depth_current
CameraPose_previous, CameraPose_current
CameraIntrinsics_previous[, CameraIntrinsics_current]
```

其流程为：

1. 在当前 DINO patch 网格取当前深度，用 `CameraIntrinsics` 将像素和深度反投影为当前相机坐标中的 3D 点。
2. 用两帧 `CameraPose` 将点从当前相机坐标变换到上一帧相机坐标，再用上一帧内参投影回上一帧图像。
3. 在对应位置采样上一帧 depth 和 DINO，计算重投影后的 depth residual 与 DINO cosine distance。
4. 对二者分别阈值化，取并集，按配置膨胀，并依据 motion score 做 `top-k/max-token` 限制。
5. 将 motion mask/score 和 DINO 特征映射到 WAN transformer patch 网格。

`CameraIntrinsics` 是相机矩阵（`fx, fy, cx, cy`），用于像素、3D 相机坐标之间的转换；它对应 depth/RGB 图像分辨率，而不是 DINO 网格分辨率。`CameraPose` 用来抵消相机自身运动，否则静止物体会因为相机移动而被误判成运动。默认 pose convention 是 `world_from_camera`，即 `p_world = pose @ p_camera`；也可显式选择 `camera_from_world`。如果前后帧内参不同，应另传 `camera_intrinsics_current`。

默认建议的单相机网格对齐为：

```text
DINOv2 ViT-B/14, 224x224 input  -> 16x16 DINO patches
WAN VAE, 256x256 RGB            -> 16x16 latent
WAN transformer patch (1,2,2)  ->  8x8 video tokens
```

检测先在 16×16 DINO 网格进行；映射到 8×8 时，motion mask/score 使用“任一运动/max score”的池化语义，DINO index feature 使用 2×2 average pooling。因此四个 DINO patch 稳定对应一个 WAN transformer token。

多相机数据仍按现有数据集逻辑沿 latent width 拼接。sidecar 生产者应分别使用每台相机自己的内参和 pose 做检测，再按 `obs_cam_keys` 的固定顺序把各自 8×8 结果拼成 `8 × (8 * num_cameras)` 网格。`motion_indices` 必须索引这个拼接后的网格，而不是某一台相机自己的 8×8 网格。

## 5. 训练 sidecar 契约

每个 latent segment 对应一个 PyTorch sidecar 文件：

```text
<dataset>/<rgb_motion_root_name>/chunk-XXX/
    episode_XXXXXX_START_END.pth
```

推荐的 canonical payload（文件内是一个 `dict`）如下。这里 `F` 是 WAN transformer 的时间 patch 数，`K` 是每帧固定的最大稀疏 token 数，`D_dino`/`D_neo` 是两个特征宽度：

| 字段 | shape | dtype/语义 |
|---|---:|---|
| `motion_indices` | `[F,K]` | 整数；每帧拼接后 WAN 空间网格内的局部 flat index，`-1` 表示 padding |
| `motion_valid_mask` | `[F,K]` | bool 或 `0/1`；可省略，默认由 `motion_indices >= 0` 推出 |
| `motion_scores` | `[F,K]` | float；用于每帧排序和 top-k，可省略 |
| `world_time_id` | `[F,K]`、`[F]` 或 scalar | 整数；可省略，默认使用完整 episode/stream latent clip 的 WAN latent ordinal `0..F-1`。随机 crop 会先生成完整 ordinal 再切片，不会把 crop 重新从 0 编号；如果保存的 sidecar 是从非零 step 开始的中途 clip，应写入显式 `world_time_id`，且显式值始终优先 |
| `dino_features` | `[F,K,D_dino]` | float；必需，且 `D_dino > 0` |
| `neoforce_features` | `[F,K,D_neo]` | float；可省略；省略时规范化为 `[F,K,0]` |
| `observation_flag` | `[F,K]`、`[F]` 或 scalar | 只能是 `0/1`；`1=observed`，`0=predicted`；训练 sidecar 默认有效项为 `1` |
| `visual_valid` | `[F,K]`、`[F]` 或 scalar | bool 或 `0/1`；可省略，默认等于 motion validity |
| `tactile_valid` | `[F,K]`、`[F]` 或 scalar | bool 或 `0/1`；`D_neo > 0` 时必须显式提供，因为数值零不能表示模态是否存在；无 NeoForce/宽度为 0 时强制为 `False` |

也可以用全局地址形式 `rgb_motion_indices: [N]` 代替 `motion_indices`，其中地址位于展平的 `F × H_p × W_p` 网格；与其对应的标量字段使用 `[N]`，feature 使用 `[N,D]`。两个 index key 必须且只能提供一个。数据集加载器最终总会规范化成 `[F,K]` canonical schema。

多相机 sidecar 必须同时写入以下地址校验字段，避免相机顺序、patch
配置或单台相机分辨率变化后静默取错 patch（单相机可省略）：

```python
{
    "camera_keys": ["observation.images.top", "observation.images.wrist_l"],
    "patch_size": (1, 2, 2),
    "spatial_grid_shape": (8, 16),  # 例如两台相机沿 width 拼接
    "provenance": {
        # 按 camera_keys 的同一顺序；不能只核对拼接后的总宽度。
        "camera_wan_grid_shapes": {
            "observation.images.top": [8, 8],
            "observation.images.wrist_l": [8, 8],
        }
    },
}
```

加载器会按 `motion_scores` 对每帧降序选择最多 `rgb_motion_max_tokens` 个 token，然后把不足部分补到固定 `K`，以便默认 DataLoader collation。padding 的 index/time 分别为 `-1`，其余数值清零，三个 valid mask 均为 `False`。

为兼容手工制作的单相机 sidecar，dataset 仍允许其省略 `provenance.bundle_frame_ids`；这种旁路的时间对齐由 sidecar 作者负责。下面的官方 builder 总会写入该字段，并进一步用 latent 内的 WAN temporal provenance 校验 anchor schedule，因此正式训练建议只使用 builder 输出。

### 5.1 从对齐的 RGB-D 数据构建 sidecar

仓库提供 `script/build_rgb_motion_sidecars.py`。它不猜测 LeRobot 中哪个字段是 depth、相机 pose 或相机内参；这些字段在当前数据集格式中没有统一约定。构建器使用一个显式 JSON manifest，把用户准备好的、时间和像素已经对齐的 RGB-D tensor bundle 与现有 WAN latent 文件交叉校验，然后生成上述 canonical sidecar。

一个最小 manifest 如下：

```json
{
  "schema_version": 1,
  "camera_keys": ["observation.images.top", "observation.images.wrist_l"],
  "output_root": "../dataset/rgb_motion",
  "dino_model": "/models/dinov2-base",
  "dino_image_size": [224, 224],
  "max_tokens": 32,
  "patch_size": [1, 2, 2],
  "first_frame_policy": "require_previous",
  "detector": {
    "depth_threshold": 0.02,
    "dino_threshold": 0.2,
    "dilation_radius": 1,
    "pose_convention": "world_from_camera"
  },
  "segments": [
    {
      "episode_index": 12,
      "chunk_index": 0,
      "start_frame": 0,
      "end_frame": 17,
      "bundle": "bundles/episode_000012_0_17.pth",
      "latent_files": {
        "observation.images.top": "../dataset/latents/chunk-000/observation.images.top/episode_000012_0_17.pth",
        "observation.images.wrist_l": "../dataset/latents/chunk-000/observation.images.wrist_l/episode_000012_0_17.pth"
      },
      "anchor_indices": [0, 4, 8, 12, 16],
      "world_time_ids": [0, 1, 2, 3, 4]
    }
  ]
}
```

manifest 中的相对路径都相对 manifest 文件所在目录解析。每个 segment 的输出路径不是任意字符串，而是由构建器固定生成：

```text
<output_root>/chunk-XXX/episode_XXXXXX_START_END.pth
```

`latent_files` 必须按 `camera_keys` 的同一顺序列出每台相机的现有 WAN latent。每条路径都必须位于与该 segment 的 `chunk_index` 完全一致、采用标准零填充名称的 `chunk-NNN` 祖先目录下；相机目录可以位于 chunk 目录和文件之间，例如 `latents/chunk-000/<camera>/episode_....pth`。路径落在另一 chunk、写成非标准的 `chunk-0`，或完全不含 chunk 目录都会直接报错，构建器不会根据 episode 编号猜测或修正 chunk。构建器随后会核对文件名、`latent_num_frames`、`latent_height/width`、`video_num_frames`、`frame_ids` 和编码器写入的 `temporal_provenance`；每台相机的 latent 空间大小还会用 `patch_size` 转成 WAN transformer grid，并写入 `provenance.camera_wan_grid_shapes`。dataset 会把它与本次实际加载的每台相机 grid 逐一比较，而不只是比较 width-concat 后的总 grid。各 segment 的 grid 必须稳定，多相机 grid 高度必须一致，才能沿 width 拼接。

`bundle` 是一个受信任的本地 PyTorch `.pt/.pth` 文件，结构为：

```python
{
    # 与每个 latent payload 的 frame_ids 完全一致。
    "frame_ids": [0, 1, 2, ..., 16],
    "cameras": {
        # 顺序必须与 manifest.camera_keys 完全一致。
        "observation.images.top": {
            "rgb": uint8_tensor,              # [T,H,W,3] 或 [T,3,H,W]
            "depth": float_tensor,             # [T,H,W] 或 [T,1,H,W]
            "camera_pose": float_tensor,       # [T,4,4]、[1,4,4] 或 [4,4]
            "camera_intrinsics": float_tensor, # [T,3,3]、[1,3,3] 或 [3,3]
        },
        "observation.images.wrist_l": { ... }
    },
    # first_frame_policy=require_previous 时必需；每台相机各一帧。
    "previous_frames": {
        "observation.images.top": {
            "rgb": uint8_tensor,
            "depth": float_tensor,
            "camera_pose": float_tensor,
            "camera_intrinsics": float_tensor
        },
        "observation.images.wrist_l": { ... }
    }
}
```

这里的 depth 必须已经是与 RGB 像素对齐的浮点 z-depth，内参也必须对应该分辨率。构建器不会猜 `uint16` 的比例因子，遇到整数 depth 会直接报错。pose 必须是相机 pose，而不是机器人末端执行器状态。

`anchor_indices` 是 WAN 每个 latent time step 对应的 bundle 原始帧位置，长度必须等于所有相机 latent 的 `latent_num_frames`；`world_time_ids` 是同样长度的 episode/stream WAN latent-step 序号。二者都必须显式给出、严格递增，构建器不会用 raw `frame_ids` 代替 world time。当前编码脚本会在每个 latent 文件的 `temporal_provenance` 中记录 `anchor_semantics=causal_chunk_end`、时间 stride、`latent_anchor_indices` 以及对应的 episode-local `latent_anchor_frame_ids`。构建器要求 manifest anchors 与这些编码期事实完全一致；旧 latent 没有该 provenance 时会要求使用当前 `encode_lerobot_n0_latents.py --overwrite` 重新编码，而不会退回到 `0,4,8,...` 的隐式猜测。

首帧有三种明确策略：

- `require_previous`（默认）：bundle 必须带每台相机的 `previous_frames`，首个 anchor 也执行真实相邻帧检测。
- `empty`：首个 anchor 不选 video token，适合 episode 没有前一帧的情况。
- `all`：首帧选完整拼接网格；此时 `max_tokens` 必须至少等于所有相机的空间 token 总数。

执行命令：

```bash
python script/build_rgb_motion_sidecars.py \
  --manifest /path/to/rgb_motion_manifest.json \
  --device cuda
```

DINOv2 默认 `local_files_only=True`，本地路径/缓存缺失会明确失败，不会静默访问网络。只有显式添加 `--allow-dino-download` 才允许下载。构建器会在加载 DINO 前解析全部输出地址并检查上述 latent chunk 路径契约；已存在的 sidecar 默认跳过，只处理尚缺失的 segment。若全部输出都存在，则会直接成功退出，既不读取已跳过 segment 的 bundle/latent payload，也不要求 DINO checkpoint。确认需要重新生成时使用 `--overwrite`。由于 `.pth` 使用 PyTorch 反序列化，manifest、bundle 和 latent 文件都应来自受信任来源。

## 6. 训练数据流

启用后，训练路径为：

1. 现有离线流程照常生成 dense WAN video latents、触觉 latents、action 和 text embedding；另行生成上述 RGB-motion sidecar。
2. dataset 按 latent segment 加载 sidecar，检查帧数、相机顺序、patch size 和空间网格，并规范化/排序/补齐到 `[F,K]`。
3. trainer 照常对 dense video latent 做 flow-matching 加噪，并把 motion/index 字段原样放入 `latent_dict`。
4. transformer 先 patchify noisy/clean WAN latent，再按 `motion_indices` gather 原始 patch，之后才调用原 `patch_embedding_mlp`。因此静止区域不会经过 video input projection 和 video expert。
5. 稀疏 video token 继续使用它们原本的时空 grid/RoPE 和每帧 diffusion timestep；padding token 被 attention mask 排除。tactile/action expert 的输入与损失不变。
6. video 输出仍是原 WAN patch velocity 格式。loss 端从 dense flow target 中 gather 同一批 patch，只在有效 motion token 上按原每帧 scheduler 权重计算 MSE。

若选中所有 WAN patch，稀疏 loss 与原 dense loss 的尺度一致。

## 7. Serving 数据流

server 不会在 diffusion denoising 循环里运行 DINOv2。最稳定的接口仍是客户端预先计算 canonical sidecar，并作为 `obs["rgb_motion"]` 发送：

```python
obs["rgb_motion"] = {
    "motion_indices": ...,       # [F,K]
    "motion_valid_mask": ...,    # [F,K]
    "motion_scores": ...,        # [F,K]
    "dino_features": ...,        # [F,K,D_dino]
    "neoforce_features": ...,    # [F,K,D_neo]，RGB-only 时可省略
    "visual_valid": ...,         # [F,K]
    "tactile_valid": ...,        # [F,K]
    # 多相机时三项必需，且必须与 server 完全一致。
    "camera_keys": ["observation.images.top", "observation.images.wrist_l"],
    "patch_size": (1, 2, 2),
    "spatial_grid_shape": (8, 16),
}
```

示例闭环客户端的 `infer_chunk` / `commit_kv_cache` 可通过 `rgb_motion=` 原样发送该 sidecar；高层 `run_chunk` / `run_episode` 对应使用 `make_infer_rgb_motion` 和 `make_commit_rgb_motion` builder。它们也可与 raw `rgb_motion_inputs` 同时提供，最终仍由 server 按“预计算 sidecar 优先”处理。

多相机的 `motion_indices` 依赖 width-concat 顺序，所以 server 会强制校验上述 `camera_keys` / `patch_size` / `spatial_grid_shape` provenance；缺失或不一致都会拒绝。单相机为兼容旧 client 仍允许省略三项；但只要提供了其中一项，就必须完整提供且通过校验。

如果开启 `rgb_motion_online_preprocess=True`，真实 observation 在没有 `obs["rgb_motion"]` 时也可以发送 raw RGB-D：

```python
obs["rgb_motion_inputs"] = {
    # 必需，必须与 job_config.obs_cam_keys 完全同序。
    "camera_keys": ["observation.images.top", "observation.images.wrist_l"],
    "cameras": {
        "observation.images.top": {
            "rgb": ...,                # [T,H,W,3] 或 [T,3,H,W]；单帧可省 T
            "depth": ...,              # float z-depth, [T,H,W] 或 [T,1,H,W]
            "world_from_camera": ...,  # [T,4,4]、[1,4,4] 或 [4,4]
            "intrinsics": ...,         # [T,3,3]、[1,3,3] 或 [3,3]
        },
        "observation.images.wrist_l": { ... },
    },
    # 可省略；server 总是按下一次 streaming VAE 调用推导并校验。
    # 此例为已有 cold seed 的 warm T=8、temporal stride=4，因此 F=2。
    "anchor_indices": [3, 7],
    # 可省略；默认使用 server frame_st_id 起的 WAN latent step。
    # 如显式提供，必须与该 server 坐标完全相等，不接受静默改写。
    "world_time_ids": [12, 13],
    # require_previous 冷启动时必需；结构与上方相同，每相机可给一帧。
    "previous": {
        "camera_keys": ["observation.images.top", "observation.images.wrist_l"],
        "cameras": {
            "observation.images.top": { ... },
            "observation.images.wrist_l": { ... },
        },
    },
}
```

`world_from_camera` 只能是相机外参；server 绝不会用 `obs["state"]`、机器人末端位姿或 action 代替。各相机必须有同样的 raw `T`，depth/RGB 像素对齐，内参对应该分辨率。此外，`rgb_motion_inputs.cameras[cam].rgb[t]` 必须与 streaming WAN VAE 真正读取的 `obs["obs"][t][cam]` 具有相同 shape 和完全相同的像素值（仅 dtype 不同可以）。server 会校验整条 raw 序列，包括没有被 anchor 选中但仍会进入 VAE 的帧；不允许一套 RGB 给 WAN VAE、另一套给 DINO/geometry。

在线 raw 的 anchor 由**下一次实际 streaming WAN VAE 编码的冷/热状态**唯一决定，而不是由 `frame_st_id` 决定，也不能沿用离线 whole-video 的 `[0,stride,2*stride,...]`：

- causal cache 为空的 cold seed 必须恰好发送 `T=1`，对应 `anchor_indices=[0]`；
- causal cache 已有 seed 的 warm grounding 使用 VAE `config.scale_factor_temporal`（WAN 通常为 4），要求 `T` 能被 stride 整除；每个 latent 对应完整 warm chunk 的因果末帧，因此 `T=8,stride=4` 对应 `[3,7]`；
- `anchor_indices` 可以省略并由 server 填入；如果客户端显式发送，则必须与上述派生值完全相等，否则在 DINO、VAE 和 KV cache 发生任何修改前拒绝；
- cold imagination 缓存 seed 后，第一次 grounding 的 `frame_st_id` 仍可能是 0，但 VAE 已经是 warm，仍必须使用 `[3,7,...]` 规则。

预计算 canonical `obs["rgb_motion"]` 已直接按 latent 行给出 sidecar，不含 raw anchor，因而不经过这项在线 raw 校验。raw DINO/geometry 只运行一次，并连同 motion address、四项独立 index、presence mask 与 provenance 一起完整规范化；只有全部成功后才允许清理 prediction cache 或推进 streaming VAE。候选 `_last_rgb_motion` 和 previous-raw support 也先保持私有，确认 WAN latent 行数一致后才提交。因此 DINO checkpoint 缺失、bad anchor/world time、非法 sidecar 等错误不会把当前 episode 推进一半。

真实 grounding 最终生成的 sidecar 必须恰好是 `F` 行；不会拿一帧真实观测复制成多帧。

`first_frame_policy=require_previous` 时，startup/reset 后第一次 raw 调用必须带 `previous`，否则立即报冷启动错误。每次成功处理后，server 保存每台相机 `anchor_indices[-1]` 指向的 raw RGB/depth/pose/intrinsics，而不是无条件保存 raw 序列尾帧；因此下次比较严格接在最新的实际 WAN anchor 上。下次可以不再传 `previous`；reset 会清空这些状态。`empty` 和 `all` 不要求 previous，但 `all` 的 token budget 必须容纳完整多相机网格。

如果同一个 observation 同时含有 `rgb_motion` 和 `rgb_motion_inputs`，预计算 `rgb_motion` 始终优先，raw producer 不会运行。在线 DINOv2 仅在第一个确实需要 raw 处理的 observation 才懒加载，并强制 `local_files_only=True`；本地目录/Hugging Face cache 缺失时明确失败，不会在 serving 期间自动下载。

canonical `rgb_motion` 中的 `world_time_id` 和 `observation_flag` 即使由客户端提供，server 也会按当前调用的真实语义重新生成：grounding 为 `observation_flag=1`，future imagination 为 `0`，world time 使用当前 `frame_st_id` 起的 WAN latent-step 坐标。raw `rgb_motion_inputs.world_time_ids` 如果显式给出，则会先校验它与这个坐标完全相等，不会先接受再静默忽略。这里的 `frame_st_id` 同样不是原始相机帧号或 diffusion timestep。

运行时：

1. server 仍用 streaming WAN VAE 将真实 RGB 编码成 dense latent。
2. grounding 时，用真实 observation sidecar gather 稀疏 WAN patch，并把 index 标成 observed 后写入 cache。
3. future video denoising 暂时把最近一次真实观测的 motion support 和 DINO/NeoForce 延展到预测帧，改写 world time，并标成 predicted。cold chunk 只会沿 frame 维将实际 seed 覆盖到前导帧，不会把 token 数 `K` 误当成 seed 帧数而污染后续 prediction。
4. dense 预测 canvas 的静止位置以最近一次真实观测 latent 为背景；只有 motion patch 从噪声开始并参加 video denoising。
5. 稀疏 video velocity 被 scatter 回 dense canvas，action expert 继续通过 shared attention 使用 video/tactile 上下文。
6. 新真实观测到达时重新 grounding。

如果 `use_rgb_motion_tokens=True` 且真实 grounding observation 既没有预计算 sidecar，也没有已开启的完整 raw RGB-D 输入，server 会直接报错；不会静默退回 dense token。
真实 grounding 的 sidecar 必须恰好提供与本次 WAN latent 相同的 `F` 行，不能用一行当前观测冒充多帧历史；只有 future prediction 可以把最近一次 support carry-forward。RGB-motion serving 同训练数据契约一样，当前明确要求 `patch_size[0] == 1`。

RGB-motion serving 当前只支持 `transformer/config.json` 中 `is_mot=true` 的 MoT checkpoint（训练时应使用 `cfg.use_mot=True`）。legacy `WanAttention` streaming cache 没有逐 token 的 padding/index sidecar；若对 `is_mot=false` checkpoint 开启 RGB-motion，server 会在启动时直接报错。

## 8. 如何开启

共享默认配置保持关闭，以兼容现有数据和 released checkpoint：

```python
use_rgb_motion_tokens = False
rgb_motion_root_name = "rgb_motion"
rgb_motion_max_tokens = 32
rgb_motion_require_index = True
rgb_motion_online_preprocess = False
```

在实际训练配置中覆盖：

```python
cfg.use_rgb_motion_tokens = True
cfg.rgb_motion_root_name = "rgb_motion"
cfg.rgb_motion_max_tokens = 32
cfg.rgb_motion_require_index = True
```

要让 server 自动从 raw RGB-D 生成 sidecar，再设置：

```python
cfg.rgb_motion_online_preprocess = True
cfg.rgb_motion_dino_model_name_or_path = "/local/models/dinov2-base"
cfg.rgb_motion_dino_device = "cpu"  # 也可显式指定 cuda device
cfg.rgb_motion_dino_image_size = (224, 224)
cfg.rgb_motion_dino_float_input_range = "0_1"  # 仅影响 float RGB
cfg.rgb_motion_first_frame_policy = "require_previous"
cfg.rgb_motion_depth_threshold = 0.02
cfg.rgb_motion_dino_threshold = 0.2
cfg.rgb_motion_dilation_radius = 1
```

在线 producer 要求 `rgb_motion_max_tokens > 0`。这些 producer 参数只决定如何由 RGB-D 得到 canonical sidecar，不会把 DINO/geometry 特征混入原 WAN content embedding。

`posttrain_server` 继承 `posttrain` 配置；若使用其他 server config，也要设置相同的 `use_rgb_motion_tokens`、`rgb_motion_max_tokens`、`rgb_motion_require_index` 和 `patch_size`。checkpoint 的 `train_meta.json` 会记录并在 server 启动时核对这些值；RGB-motion 开启时还会逐项核对 `obs_cam_keys` 的顺序，因为它决定多相机 token 在拼接网格中的地址。训练和 serving 不一致会被拒绝；旧 checkpoint 的 metadata 没有 `patch_size` 时会跳过这一项。

`rgb_motion_max_tokens=0` 表示 loader 不主动统一 K；除非所有 sidecar 已经使用相同 padding 宽度，或 batch size 为 1，否则不要这样设置。

`EgoMotionCompensatedMotionDetector` 当前作为可复用组件暴露，常用构造参数是：

```python
from n0_twam.models.rgb_motion import EgoMotionCompensatedMotionDetector

detector = EgoMotionCompensatedMotionDetector(
    depth_threshold=0.02,
    dino_threshold=0.2,
    dilation_radius=1,
    max_tokens=32,
    pose_convention="world_from_camera",
)
```

阈值的单位依赖 depth 标定和 DINO cosine distance 分布，应针对数据集标定。共享配置中这些字段只配置离线/在线 sidecar producer；它们不是 transformer 的可学习参数，也不改变原 content embedding。

## 9. 当前边界

RGB motion v1 刻意只完成“二维运动区域稀疏化 + 独立语义 index”这一层：

- 仓库现在同时支持预计算 `obs["rgb_motion"]` 和 opt-in `obs["rgb_motion_inputs"]` raw RGB-D 路径；后者使用懒加载、仅本地的 frozen DINOv2，且只在 observation grounding 之前运行一次。
- 自动离线/在线 producer 当前是 **RGB-only**：它输出宽度为 0 的 NeoForce 和 `tactile_valid=False`。canonical sidecar 接口可以接收外部提供的 NeoForce，但仓库尚未实现 NeoForce encoder，也没有实现“触觉接触关联到哪个运动 patch”的生产逻辑。
- DINO、NeoForce 和 `world_time_id` 当前作为彼此独立的 cache sidecar metadata 保存，可读取但不会进入 Q/K/V、attention score 或训练 loss。`observation_flag` 会参与 observed/predicted cache 状态管理。也就是说，语义 index 的存储通路已经接好，利用各分量做检索、替换或压缩的在线策略尚未接好。
- future imagination 当前沿用最近一次真实观测的 motion support 和 DINO/NeoForce，仅更新 `world_time_id` 并标记 `observation_flag=0`；它没有预测新的 motion mask、DINO、NeoForce 或 identity。这里是稀疏 denoising 支撑集的 carry-forward，不是完整世界模型的 index prediction head。
- server 当前在真实 observation 重新 grounding 时沿用原策略清除 prediction cache。遮挡区域的预测点长期保留、实际观察后按语义替换，以及“压缩而非删除”的双向拖影策略，**不是 RGB v1 的默认在线策略**。
- `n0_twam.models.semantic_cache.SemanticKVCache` 已提供独立 K/V-index 存储、同 world-time 预测匹配和分项 importance/top-k 等研究接口，但它目前是研究组件；shared-attention 默认 streaming policy 尚未改成该压缩策略。
- live MoT 当前由每一层 shared-attention cache 各自保存一份 semantic sidecar。这样可以保证每层槽位淘汰和 index 严格对齐，但高维 DINO/NeoForce 在长 cache 上会线性增加显存。以 8064 个槽位、DINO-B/14 的 768 维 float32 metadata 为例，约为 23.6 MiB/层、30 层约 709 MiB；若 NeoForce 也是 768 维则大致翻倍。把不可变语义字段提升为 model-level 共享存储仍是后续优化，不属于本版的功能正确性承诺。
- 离线 builder 当前按“单个 segment 的全部 anchor”为一个 DINO batch，没有独立的 microbatch 参数；超长 segment 应先在 manifest 中切小，否则可能触发显存不足。
- 本版本的 token 地址仍是二维 WAN patch 地址，不是 `{x,y,z}`；3D 几何仅在 motion detector 内临时用于相机自运动补偿。
- 自动测试覆盖 CPU 上的数据契约、几何、稀疏 gather/loss、cache 事务和 server/client 流程；尚未在本仓库环境中完成真实 DINO/WAN checkpoint、CUDA 以及真实机器人相机标定的端到端验证。

后续点云版本可以继续复用独立 index/cache 语义，但需要另行实现点云构建、预测点与真实点替换、跨时刻 association，以及正式的 cache compression policy。
