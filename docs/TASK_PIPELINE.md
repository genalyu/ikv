# 可复用任务分析与后训练

统一入口是 `python script/task_pipeline.py`。任务模板位于
`examples/tasks/ur5e_phone_weight.json` 和 `examples/tasks/neosim_hidden_usb.json`。
命令均从仓库根目录执行。当前交付使用 CPU 验证；没有申请 GPU，
没有执行 VAE/DINO 编码、真实模型单/双卡训练或 2000 步正式训练。

## 官方配方与边界

参考 [N0-TWAM 固定版本](https://github.com/neoteai/N0-TWAM/tree/cdd87b6a141667123ad2c25f452478afdb71e287)
的 `run_posttrain.sh`、`twam_posttrain_cfg.py`、base/shared config 和官方预处理脚本。
当前代码以官方 posttrain 配置为底，再覆盖数据路径、prompt、机器人映射、
motion/IKV 和少卡累积设置。损失、噪声流程和可训练 Transformer 参数范围不变；
这不是只训练新模块的 adapter，也不使用 RL 或新增辅助损失。

| 项目 | 设置 |
|---|---|
| 更新次数 | 2000 次 optimizer.step，非 micro-step |
| 学习率 | 1e-4，cosine，最小比例 0.1，warmup 20 |
| 优化器 | AdamW，betas=(0.9,0.95)，weight_decay=0.1，原梯度裁剪 |
| 精度/并行 | BF16，FP32 reduction，FSDP2，activation checkpointing |
| 保存 | 每 500 次优化器更新 |
| 动作 | absolute EE，q01/q99，12 actions/frame |
| 触觉 | global 开启，local-current 开启，不伪造缺失触觉 |
| 数据 | 全部有效 episode，train/val 指向同一池，val_interval=9999 |
| 序列 | 保留 episode；max_latent_frames/max_tactile_frames=0 |

官方发布多任务模型的 10000 步与此处单任务模板的 2000 步不是同一设置。
普通 baseline 采用原始序列注意力训练，不模拟在线 FIFO 写入/淘汰；
IKV 模式使用现有有限容量、按时间顺序的 teacher-forced 历史注意力。
文本编码的 max_sequence_length=512 是文本 token 上限，不是视频帧截断。

| 卡数 | 每卡 batch | 梯度累积 | 有效 batch |
|---|---:|---:|---:|
| 1 | 1 | 32 | 32 |
| 2 | 1 | 16 | 32 |
| 8 | 1 | 4 | 32 |

少卡保持学习率、更新次数和有效 batch，但速度、每卡显存和浮点计算顺序不同。
累积无法消除模型状态和长序列激活的显存开销。不得把 CPU 测试解释为单卡能装下模型。

## 环境与配置

在已经验证的 N0-TWAM Python 环境补充数据工具依赖，不重装 PyTorch：

```bash
python -m pip install -r requirements-task-pipeline.txt
python -m pip install --no-deps lerobot==0.3.3
```

若环境由 uv 管理，可用 `uv pip install --python /path/to/python ...`。
LeRobot 使用 --no-deps 是为了避免替换训练环境的 torch 等固定依赖。

模板支持环境变量。以下均替换成你的服务器路径：

```bash
export TASK_SOURCE=/path/to/yanqiang.tar
export TASK_WORK_ROOT=/path/to/task_outputs
export BASE_CHECKPOINT=/path/to/n0-twam-base
export DINO_MODEL=/path/to/local/dinov2-base
TASK=examples/tasks/ur5e_phone_weight.json
```

base 目录需要 transformer/、vae/、tokenizer/、text_encoder/ 和 empty_emb.pt；
也可在 runtime.model_path 指定独立的 VAE/文本组件目录。
DINO 权重必须预先存在本地，工具不会隐式联网下载模型。

任务 JSON 的 robot/runtime 可以是内嵌对象，也可以是相对于任务 JSON 的另一份 JSON 路径。
换 task 修改 name/source/prompt；换机器人明确修改坐标系、单位、动作语义、夹爪映射、
相机/触觉顺序和单双臂，不靠机器人名字猜测数据语义。
runtime 中的相对路径以任务 JSON 所在目录为基准。
机器路径和生成数据存放在仓库之外。

## 1. 分析原始数据（CPU）

```bash
python script/task_pipeline.py analyze --task "$TASK"
```

输出到 `$TASK_WORK_ROOT/<name>/reports/`：

- report.html：自带 Plotly，可离线查看；点击图例选择轨迹。与 PNG 一起保留。
- overview.png、trajectories.png：时长、空间分布、轨迹、路程和目标—观测偏差。
- episodes.csv、summary.json、audit.json：逐轨迹统计、全量汇总和时间对齐审计。
- 仿真另有 memory_episodes.csv、memory_summary.json、memory_distribution.png。

数值统计使用所有采样点；交互折线最多约 600 点/轨迹，并包含终点。
空间密度是采样点占用分布，不是“独立轨迹出现概率”。不同坐标系必须分开分析。
目标—观测偏差只有在动作是记录的控制命令时才可解释为跟踪偏差。
idle 的默认判据是末端速度小于 0.005 m/s。

真机数据原始任务标签为 Cup_Load_Memory，与本次手机盒 prompt 不一致。
转换使用任务配置的 prompt，原标签保存在来源记录，不覆盖原始压缩包。
采集器“完全正常”是数据完整性标签，不代表任务成功率。
重复时间戳明确计数；因果对齐在同一时间取最后一条，速度统计排除零时间间隔。
各传感器共同可观测起点及偏移量会被记录，不用未来图像填补。
超出门槛的对齐默认报错；本次真机模板明确设置 on_invalid_alignment=exclude，
会生成 meta/excluded_episodes.json。实测第 118 号轨迹最大图像年龄约 0.539 秒，
超出 0.1 秒门槛；其余 127 条通过该门槛。初始共同起点偏移不超过 0.008 秒。
原始报告仍统计全部 128 条，训练只使用通过质量门槛的轨迹。

UR5e 映射经过采集代码核实：米制绝对 TCP 位姿、矩阵前两列 rot6d、
动作夹爪为开口比例，观测夹爪为米制宽度，最大行程 0.085 m。
输出夹爪统一为开口比例，原始记录仍保留。目标位姿来自 actions.eef_pose，
不以观测位姿代替命令。

## 2. 转换与编码

```bash
# CPU：对齐到官方 30 Hz 动作时钟，生成 LeRobot v2.1 和来源映射
python script/task_pipeline.py prepare --task "$TASK" --stage convert

# 申请 GPU 后执行；当前尚未运行
python script/task_pipeline.py prepare --task "$TASK" --stage rgb --device cuda:0
python script/task_pipeline.py prepare --task "$TASK" --stage tactile --device cuda:0
python script/task_pipeline.py prepare --task "$TASK" --stage features --device cuda:0
python script/task_pipeline.py prepare --task "$TASK" --stage pool
```

也可一次运行 `prepare --stage all`。默认 stage 是 convert，避免无意启动 GPU 编码；
`--dry-run` 只显示准备流程。RGB 256×256、触觉 128×128，编码采样率 10 Hz。
先产生 latent 的实际 frame_ids 和 causal anchors，再构建索引；
不以原始相机帧号冒充 WAN 时间编号，不让 patch 跨相机边界。

features 阶段使用与线上同一 RGBFrameDifferencePreprocessor，
保存 rgb_motion/ 与 ikv_index/。第一帧完整保留；后续只比较已观测到的帧。
dense DINO 的空间池化复用线上实现。无 NeoForce 标定输入时该索引字段为空，
不会把触觉 RGB 或深度直接当成力；触觉图像仍参与官方触觉训练分支。
四组消融共用转换数据、latents 和归一化，不各自随机重建数据池。

转换可重入：完成时检查配置指纹；配置发生变化须使用新 work_root。
转换数据不覆盖原始数据。中断后可以继续完成同一配置的 episode。
修改 prompt 后必须重新编码文本，不能沿用旧 text_emb。

## 3. 检查、启动与恢复

```bash
python script/task_pipeline.py check --task "$TASK" --mode motion_ikv

# 无 GPU 时预览启动配置
python script/task_pipeline.py train --task "$TASK" --mode motion_ikv --gpus 1 --dry-run

# 申请 GPU 后任选一个启动；四组独立从 base 权重初始化
CUDA_VISIBLE_DEVICES=0 python script/task_pipeline.py train --task "$TASK" --mode baseline --gpus 1
CUDA_VISIBLE_DEVICES=0 python script/task_pipeline.py train --task "$TASK" --mode motion --gpus 1
CUDA_VISIBLE_DEVICES=0,1 python script/task_pipeline.py train --task "$TASK" --mode ikv --gpus 2
CUDA_VISIBLE_DEVICES=0,1 python script/task_pipeline.py train --task "$TASK" --mode motion_ikv --gpus 2
```

同一组的输出为 `<work_root>/<name>/runs/<mode>/`。
已有运行目录不会被新实验覆盖；重新实验使用新的 work_root。
检查会拒绝缺少 latent、触觉、索引或 norm 的情况，而不是静默关闭功能。

每次运行记录 resolved_config.json、recipe_diff.json、task.json、
provenance.json 和 serve_overrides.json。包含版本、prompt、归一化文件校验值、
有效 batch 和实际功能开关。日志中的验证损失不代表独立验证集。

完整恢复使用 `checkpoint_step_N/training_state/`，同时恢复原精度模型状态、
AdamW 动量、学习率调度器、优化器步数、主进程随机状态和数据采样位置：

```bash
CUDA_VISIBLE_DEVICES=0 python script/task_pipeline.py train \
  --task "$TASK" --mode motion_ikv --gpus 1 \
  --resume /path/to/runs/motion_ikv/checkpoints/checkpoint_step_500/training_state
```

完整恢复要求相同任务配置、模式和 GPU 数。不同卡数间的无缝恢复尚不支持。
旧版只含 transformer/ 的检查点可以作为权重初始化，但不能声称恢复了训练状态。
只有 complete.json 存在的完整状态可恢复。

## 4. 配套推理

```bash
python script/make_serve_bundle.py --checkpoint /path/to/checkpoint_step_2000 \
  --base "$BASE_CHECKPOINT" --bundle /path/to/serve_bundle

python -m n0_twam.n0_twam_server --bundle /path/to/serve_bundle \
  --task-overrides /path/to/runs/motion_ikv/serve_overrides.json \
  --save_root /path/to/serve_output --port 29536
```

导出的配置明确设置 baseline/motion 使用 FIFO、ikv/motion_ikv 使用 global，
并绑定训练时的容量、相机顺序、动作归一化和触觉设置。服务仍执行 train_meta 一致性检查。
客户端必须提供同样的 RGB 顺序、坐标系和夹爪比例约定；硬件动作下发需走对应机器人适配器。
这里没有新增真机运动脚本。

## NeoSim hidden USB slot

[数据集](https://www.modelscope.cn/datasets/genalyu/neosim-mem/tree/master/neosim_mem_hidden_usb_slot)
目前公开 210 条成功轨迹，包含 HDF5、预览视频和独立 metadata。
下载器先写远端文件清单，并验证所下载文件的大小与 SHA256：

```bash
# 所有轻量元数据 + 一条 HDF5；适合 CPU 检查格式
python script/download_neosim_mem.py --output /path/to/hidden_usb --episodes 0
# 正式训练前下载全部；图像数据较大
python script/download_neosim_mem.py --output /path/to/hidden_usb --episodes all
export TASK_SOURCE=/path/to/hidden_usb
TASK=examples/tasks/neosim_hidden_usb.json
python script/task_pipeline.py analyze --task "$TASK"
```

HDF5 中 step 是仿真步，120 Hz；已检查样本每 2 步记录一次，即 60 Hz。
duration 按仿真时间计算，不把 metadata.cost_time（采集墙钟耗时）当成轨迹时长。
位姿为世界坐标系 Panda hand 的 xyz+wxyz；转换成矩阵前两列 rot6d。
夹爪使用两个 finger joint 的和除以 2×0.039 m，此参数来自 NeoSim 配置，其他机器人须显式修改。

HDF5 没有保存控制器动作命令。NeoSim 原数据工具的约定是
state=data[:-1]、action=data[1:]，然后对齐下采样：
最后一帧没有下一帧标签，必须剔除，不能跨 episode 拼接。
模板将 action_labels 显式设为 unavailable，避免默认把观测伪装成命令。
若选择 NeoSim 的这套监督约定，明确改为 next_observation 后再 prepare；
其余训练算法和四组消融一致。该选择不影响原始可视化。

target_slot_idx、selected_slot_idx、actor 位置和 atom phase 标签只进入报告/来源记录，
不进入 Parquet action/state、RGB、text embedding 或 prompt。
模板使用不透露目标插槽的通用指令。memory_summary 统计提示—查询延迟与目标分布。
示例检查使用 1 条 HDF5 + 全部 metadata，不能声称做完 210 条轨迹的空间统计；
训练预检默认要求 expected_episodes=210，防止把样本目录当成完整数据集。

## 验证与待 GPU 项目

当前完整 CPU 回归为 348 项通过；新增流程与训练/推理关键回归在格式整理后另有 40 项通过。
CPU 回归涵盖单位/旋转、时间戳、颜色通道、真实视频转换及官方 LeRobot 元数据读取、
motion 与线上输出一致、未来帧不改变过去索引、四模式配置、
梯度累积等价更新，以及完整状态恢复后的下一次更新一致。

另已对一条完整 UR5e 真实轨迹执行 CPU 转换：2835 帧、四路视频，官方 LeRobot 元数据读取通过。
这不代表已经完成全量数据编码或真实模型训练。

申请资源后仍需：完整真实样本（含最长轨迹）的 VAE/DINO 编码和数据加载检查、
四模式单卡的峰值显存/更新耗时、双卡 FSDP 与恢复检查、训练/推理接线验证。
OOM 时明确报告，不自动裁短历史。单元测试和训练损失均不等同于真机/仿真任务成功率。
