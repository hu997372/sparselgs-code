# SparseLGS 语义高斯重建流程说明

这个仓库目前做的是：用少量输入视角重建 3D Gaussian 场，同时给每个 Gaussian 学一个低维语义特征，使模型不仅能渲染 RGB，也能渲染可和文本查询对齐的 open-vocabulary 语义场。主入口是 `new_ins_train.sh`，它把 DUSt3R 初始化、RGB Gaussian 训练、SAM/OpenCLIP 特征提取、可选的跨视角语义特征融合、语义特征训练和渲染串起来。

## 总体流程

```text
data/<scene>/images
    |
    | 1. coarse_init_eval.py
    |    从原始图像中抽稀疏视角，用 DUSt3R 得到初始相机、点云和 COLMAP 风格 sparse/0
    v
data/<scene>/dust3r_<N>_views/images
data/<scene>/dust3r_<N>_views/sparse/0
    |
    | 2. train_joint.py --include_feature_get 0
    |    训练 RGB 3D Gaussian，可选优化训练视角 pose
    v
output/<scene>/<N>_views_<base_level>
    |
    | 3. preprocess.py
    |    SAM 分割不同粒度 mask，OpenCLIP 给每个 mask 提 512 维语言特征
    v
data/<scene>/dust3r_<N>_views/language_features
    |
    | 4. autoencoder/train.py + autoencoder/test.py
    |    把 512 维语言特征压到 3 维，并保留分割索引
    v
data/<scene>/dust3r_<N>_views/language_features_dim3
    |
    | 5. lang_fusion.py，可选
    |    用跨视角匹配、深度和 pose 把不同视角中对应同一位置的语义 feature 对齐
    v
data/<scene>/dust3r_<N>_views/language_features
data/<scene>/dust3r_<N>_views/language_features_dim3
    |
    | 6. train_joint.py --include_feature_get 1
    |    从 RGB Gaussian checkpoint 初始化，训练 Gaussian 上的 3 维语义特征
    v
output/<scene>/<N>_views_<feature_level>
    |
    | 7. render.py --include_feature
    |    渲染每个视角的 3 维语义 feature map 和 gt feature map
    v
output/<scene>/<N>_views_<feature_level>/train/ours_1000/renders_npy
```

一句话版：DUSt3R 负责稀疏视角几何初始化，3DGS 负责可渲染的几何和外观，SAM+OpenCLIP 给 2D 图像提供语义监督，autoencoder 把 CLIP 特征压到 3 维，`lang_fusion.py` 可以在训练前统一跨视角对应区域的语义 feature，最后训练一个能从任意视角渲染语义特征的 Gaussian 场。

## 数据目录约定

每个场景建议按下面放：

```text
data/<scene>/
    images/                         # 原始全量图像
    dust3r_<N>_views/
        images/                     # 抽出的 N 张训练图像
        sparse/0/
            cameras.txt
            images.txt
            points3D.ply
            intrinsics.pt
            poses.pt
            depths.pt
        language_features/          # SAM/OpenCLIP 512D 特征，preprocess.py 生成
        language_features_dim3/     # 3D 压缩特征，autoencoder/test.py 生成
        language_features_origin/   # lang_fusion.py 第一次运行时自动备份
        language_features_origin_dim3/
```

当前脚本里默认的视角数规则：

- `teatime`, `waldo_kitchen`, `ramen`, `figurines` 使用 `N_VIEWS=4`
- 其他场景默认使用 `N_VIEWS=3`
- 可以用环境变量覆盖，比如 `N_VIEWS=6 bash new_ins_train.sh`

## Checkpoint 约定

默认脚本会找：

```text
ckpt/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
ckpt/sam_vit_h_4b8939.pth
```

如果 checkpoint 放在别处：

```bash
DUST3R_CKPT=/path/to/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
SAM_CKPT=/path/to/sam_vit_h_4b8939.pth \
bash new_ins_train.sh
```

脚本默认用 `python3` 运行 Python 入口。如果你的环境里入口命令是 `python`，可以这样覆盖：

```bash
PYTHON_BIN=python bash new_ins_train.sh
```

## 主脚本用法

最短命令：

```bash
bash new_ins_train.sh
```

默认等价于：

```bash
DATASET=waldo_kitchen \
ITER=1000 \
BASE_FEATURE_LEVEL=3 \
FEATURE_LEVELS="2" \
RUN_DUST3R_INIT=0 \
RUN_RGB_TRAIN=1 \
RUN_SAM_CLIP=1 \
RUN_AUTOENCODER=0 \
RUN_LANG_FUSION=0 \
RUN_FEATURE_TRAIN=1 \
RUN_RENDER=1 \
bash new_ins_train.sh
```

如果从原始图像尽量完整地跑一遍：

```bash
DATASET=teatime \
RUN_DUST3R_INIT=1 \
RUN_AUTOENCODER=1 \
FEATURE_LEVELS="2 3" \
bash new_ins_train.sh
```

如果要在语义训练前做跨视角语言特征融合：

```bash
DATASET=teatime \
RUN_AUTOENCODER=1 \
RUN_LANG_FUSION=1 \
LANG_FUSION_LEVEL=2 \
bash new_ins_train.sh
```

`LANG_FUSION_LEVEL` 默认取 `FEATURE_LEVELS` 的第一个值。`lang_fusion.py` 会把当前 `language_features/` 和 `language_features_dim3/` 备份到 `language_features_origin/`、`language_features_origin_dim3/`，之后从备份读原始特征并把融合后的 `_f.npy` 写回原目录。如果你刚重新生成了特征、并且想覆盖旧备份作为新的融合起点，可以加 `LANG_FUSION_REFRESH_ORIGIN=1`。

如果已经有 `language_features/` 和 `language_features_dim3/`，只想重新训练语义 Gaussian：

```bash
DATASET=teatime \
RUN_RGB_TRAIN=0 \
RUN_SAM_CLIP=0 \
RUN_AUTOENCODER=0 \
RUN_FEATURE_TRAIN=1 \
RUN_RENDER=1 \
bash new_ins_train.sh
```

如果 RGB checkpoint 不在默认的 `output/<scene>/<N>_views_<base_level>/chkpnt1000.pth`，可以直接指定：

```bash
DATASET=teatime \
RUN_RGB_TRAIN=0 \
START_CHECKPOINT=./data/teatime/dust3r_4_views/dust3r_4_views/chkpnt1000.pth \
bash new_ins_train.sh
```

也可以从 JSON 读配置。配置文件不会压过命令行环境变量，所以临时覆盖仍然方便：

```bash
CONFIG=configs/pipeline.example.json bash new_ins_train.sh
CONFIG=configs/pipeline.example.json DATASET=teatime FEATURE_LEVELS="2 3" bash new_ins_train.sh
```

示例配置在 `configs/pipeline.example.json`，里面把 dataset、阶段开关、checkpoint 路径、autoencoder 学习率和 render 参数放在一起。

## new_ins_train.sh 里每一步在做什么

`RUN_DUST3R_INIT=1`：运行 `coarse_init_eval.py`。它从 `data/<scene>/images` 均匀抽 `N_VIEWS` 张图，调用 DUSt3R 做全局对齐，然后写出 `dust3r_<N>_views/images` 和 `sparse/0`。这个阶段是几何起点。

`RUN_RGB_TRAIN=1`：运行 `train_joint.py --include_feature_get 0`。这一步不训练语言特征，只训练普通 3D Gaussian 的位置、颜色、尺度、透明度等。`OPTIM_POSE=1` 时还会把训练图像 pose 放进 optimizer 一起微调。

`COPY_GEOMETRY_TO_SOURCE=1`：把 `output/<scene>/<N>_views_<base_level>` 复制到 `data/<scene>/dust3r_<N>_views/dust3r_<N>_views`。这是旧脚本保留下来的布局。严格来说，后续只需要 checkpoint；如果不想复制一份输出，可以设 `COPY_GEOMETRY_TO_SOURCE=0`。

`RUN_SAM_CLIP=1`：运行 `preprocess.py`。它用 SAM 对每张训练图生成多粒度 mask，再用 OpenCLIP 对每个 mask 裁剪区域编码，写出 `language_features/*_s.npy` 和 `language_features/*_f.npy`。这里的 `_s.npy` 是分割索引，`_f.npy` 是 512 维 CLIP 特征。

`RUN_AUTOENCODER=1`：进入 `autoencoder/` 训练一个 512D 到 3D 再还原 512D 的 autoencoder，然后用 encoder 生成 `language_features_dim3/`。语义 Gaussian 训练默认读的是这个目录；如果不跑 autoencoder，必须提前准备好这个目录。

`RUN_LANG_FUSION=1`：运行 `lang_fusion.py`，位置在 `RUN_AUTOENCODER` 之后、`RUN_FEATURE_TRAIN` 之前。它会同时读取 512D 的 `language_features/` 和 3D 的 `language_features_dim3/`，根据 RoMa 匹配、DUSt3R 深度、pose 和当前 `LANG_FUSION_LEVEL` 对跨视角对应 mask 的 feature 做统一，然后把融合后的 `_f.npy` 写回这两个目录。默认不生成调试可视化；需要看中间结果时设 `LANG_FUSION_VISUALIZE=1`。

`RUN_FEATURE_TRAIN=1`：再次运行 `train_joint.py`，这次 `--include_feature_get 1`。代码会从 RGB checkpoint 恢复 Gaussian，然后给每个 Gaussian 新增 `_language_feature` 参数，监督信号来自 `language_features_dim3/` 对应 feature level 的 3 维图像特征。

`RUN_RENDER=1`：运行 `render.py --include_feature`。输出语义 feature map 到：

```text
output/<scene>/<N>_views_<level>/train/ours_1000/renders_npy
output/<scene>/<N>_views_<level>/train/ours_1000/renders
```

默认不优化 test pose。需要显式打开时用：

```bash
OPTIMIZE_TEST_POSE=1 OPTIM_TEST_POSE_ITER=500 bash new_ins_train.sh
```

`render.py` 里对应的是 `--optimize_test_pose` 和 `--optim_test_pose_iter`。`--optim_test_pose_iter` 默认是 `0`，单独传它也不会启用 pose 优化。

语义渲染默认加载 `chkpnt1000.pth`，可以用 `FEATURE_CKPT_ITER` 改：

```bash
FEATURE_CKPT_ITER=500 bash new_ins_train.sh
```

## feature_level 含义

`scene/cameras.py` 里按下面读取特征：

- `0`: SAM default mask
- `1`: small 粒度 mask
- `2`: medium 粒度 mask
- `3`: large 粒度 mask

当前默认 `FEATURE_LEVELS="2"`，也就是只训练 medium 粒度语义场。可以一次跑多个：

```bash
FEATURE_LEVELS="1 2 3" bash new_ins_train.sh
```

每个 level 会写到独立目录，比如：

```text
output/waldo_kitchen/4_views_1
output/waldo_kitchen/4_views_2
output/waldo_kitchen/4_views_3
```

## 评估和可视化

`eval/evaluate_iou_loc.py` 用渲染出的 3 维语义 feature map，结合 `autoencoder/ckpt/<scene>/best_ckpt.pth` 解码回 512 维 CLIP 空间，然后和文本 query 做相关性，生成 heatmap、mask 和 IoU。

典型命令：

```bash
cd eval
python evaluate_iou_loc.py \
    --dataset_name teatime \
    --feat_dir ../output/teatime \
    --output_dir ../eval_result \
    --mask_thresh 0.6 \
    --json_folder ../download/lerf_ovs/label \
    --n_views 4 \
    --total_iters 1000
```

`my_ren.sh`/`my_render.py` 更偏可视化和视频渲染，会读取优化后的 pose，插值相机路径，并生成 `my_result/` 下的视频和 heatmap。当前 `my_render.py` 里 `feature_level = 2` 是硬编码。

## 当前状态和需要注意的点

主链路已经比较清楚，但还不是完全“干净工程化”的状态：

- `new_ins_train.sh` 现在显式检查了输入目录和 checkpoint，不再默默依赖旧注释里的手动步骤。
- `configs/pipeline.example.json` 已经给出一份集中配置样例，`new_ins_train.sh` 可以通过 `CONFIG=...` 读取。
- OpenCLIP wrapper 已经抽到 `utils/openclip_encoder.py`，`preprocess.py`、`eval/openclip_encoder.py`、`my_render.py` 都复用这一份。
- 语义训练依赖 `language_features_dim3/`，而 `preprocess.py` 只生成 512D 的 `language_features/`。新脚本可以用 `RUN_AUTOENCODER=1` 补齐这一步。
- `lang_fusion.py` 已接到 `new_ins_train.sh`，但默认关闭，因为它会原地重写 `language_features/` 和 `language_features_dim3/`。第一次运行会自动备份 origin 目录，旧备份存在时会复用旧备份；需要用当前新生成的特征重建备份时设 `LANG_FUSION_REFRESH_ORIGIN=1`。
- `eval/eval.sh` 已修成用 `CASE_NAME` 判断视角数，并会把 `--n_views` 传给评估脚本。
- `init_test_pose.py` 默认 checkpoint 路径是旧机器上的绝对路径；新脚本调用时会传 `DUST3R_CKPT` 避开这个问题。
- `pca_feature.py` 只写 3D `_f.npy`，没有复制 `_s.npy`，所以它不能直接替代 autoencoder/test.py 给训练用，除非额外复制 segmentation map。
- `lang_fusion.py` 里仍有很多针对场景的阈值和开关；主脚本只负责把它放到正确阶段，不保证这些阈值对每个新场景都是最优。
- `render.py` 的 test pose 优化现在需要显式 `--optimize_test_pose`，默认关闭。
- `render.py --include_feature` 已加 `--feature_checkpoint_iteration`，主脚本用 `FEATURE_CKPT_ITER` 控制。
- 一些研究脚本还保留了硬编码路径，例如 `get_data.py`、`ablation.py`、`text_corr_fig.py`，更适合当作实验记录，不适合直接当主入口。

## 建议的下一轮整理

最值得优先做的工程化改动：

1. 把剩下的实验脚本，例如 `ablation.py`、`text_corr.py`，也逐步接入共享 OpenCLIP 和统一配置。
2. 如果要保留 PCA 压缩路线，让 `pca_feature.py` 同时复制 `_s.npy`，这样它生成的 `language_features_dim3/` 才能直接训练。
3. 把 `train_joint.py` 的 checkpoint 保存轮次也参数化到主脚本里，避免训练迭代数和渲染 checkpoint 轮次分离。
