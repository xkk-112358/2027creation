# RAF / K-Radar 数据可信度审计

对 ECCV 2026 论文 **《RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR for Robust 3D Object Detection in Adverse Weather》**（arXiv `2607.04587`）的数据与数字做逐项核验：数字层自洽、代码层对照、真实数据实测，并标注**已验证 / 未验证 / 存疑**。

配套产出：一份可离线打开的审计报告，以及一个可以直接跑模型的在线 Demo（Web 上传 → 后端推理 → BEV 结果可视化 → AI 解读）。

---

## 结论速览

| 分类 | 数量 | 说明 |
|---|---:|---|
| 已验证 | 12 | 数据划分、代码配置、公开数字交叉比对 |
| 未验证 | 6 | 受数据/权重/公开性阻塞 |
| 存疑 | 6 | 需要作者澄清或更多数据 |

**核心判断**

- 论文的 baseline 数字与 L4DR 官方日志**逐位吻合**（77.5 / 53.5，7 个天气列全部 0 偏差），未发现造假迹象。
- 论文头条精度（Tab.1 的 82.0 AP<sub>BEV</sub> / 57.4 AP<sub>3D</sub>）**仍不可判定**：本地只有极坐标 1% 稀疏雷达（ROI 内约 1,400 点/帧），论文训练用笛卡尔稀疏雷达（10,240 点/帧），密度差约 7 倍；且本仓库用单阈值 Recall/Precision，与论文 AP 口径不同。
- 论文报告的是**标签总数**（17,458 / 17,536），实际参与训练与评测的帧数因 `remove_0_obj` 过滤而更少（训练侧 17,458 → 11,021；seq 1 测试侧 597 → 193）。
- 论文的动机（恶劣天气下雷达可靠性下降）**得到实测支持**：平均回波功率 Normal 0.5388 → HeavySnow 0.0346，降幅 93.6%。

---

## 目录结构

```
.
├── report/                 审计报告（双击 raf-audit-standalone.html 即可离线查看）
│   ├── raf-audit.html
│   ├── raf-audit-standalone.html    # 单文件版，ECharts 已内联
│   ├── assets/                      # 图表与交互脚本
│   └── _shared/                     # ECharts 与字体
├── demo/web/index.html    在线 Demo 前端（选帧 / 上传 / BEV / AI 分析）
├── tools/
│   ├── demo_server.py     报告 + Demo 同源服务（/、/demo、/api/*）
│   ├── eval_seq1.py       seq 1 全量评测（多阈值 × 多 IoU）
│   ├── radar_weather_stats.py  58 序列 × 7 天气雷达统计
│   ├── pc1p_to_raf.py     pc1p.zip → RAF 稀疏雷达格式
│   ├── extract_kradar_local.py  从 218GB 序列包中按需抽帧
│   ├── extract_kradar_zip.py    HTTP Range 远程切片
│   └── demo_l4dr.py       最早的推理 + BEV 出图脚本
├── configs/                实测时使用的三份配置
├── results/                seq 1 评测与天气统计的原始输出
├── weights/                L4DR K-Radar v1.1 / v2.1 权重（Git LFS）
├── dataset_meta/           K-Radar 标签与标定归档（Git LFS）
└── external/               上游仓库（git submodule）
```

---

## 快速开始

### 1. 克隆

```bash
git lfs install
git clone --recursive https://github.com/xkk-112358/2027creation.git
cd 2027creation
```

不加 `--recursive` 不会拉取 `external/` 里的上游代码；没装 `git lfs` 则 `weights/` 只有指针文件。

### 2. 环境

实测环境：Ubuntu 20.04 / Python 3.8 / CUDA 11.3 / RTX 3090。

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install spconv-cu113==2.3.6 mmcv==1.7.1 timm==0.6.13 easydict open3d
# 以下两个需要源码编译 / 打补丁
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
pip install nms    # 之后需打两处补丁，见「已知坑」
```

关键前提：`export PYTHONPATH=$PWD/ops:$PWD`，否则 CUDA 算子（`iou3d_nms` 等）导入失败。

### 3. 数据

| 内容 | 来源 | 放置位置 |
|---|---|---|
| 58 序列标签/标定 | 本仓库 `dataset_meta/kradar-labels-meta.tar.gz` | `/hy-tmp/kradar_dataset/` |
| 稀疏雷达（34,994 帧） | `pc1p.zip` + `tools/pc1p_to_raf.py` | `/hy-tmp/kradar_dataset/rtnh_wider_1p_1/` |
| seq 1 的 LiDAR 与相机 | K-Radar 官方（`1.zip`，218GB） | `/hy-tmp/kradar_dataset/seqs/1/` |

```bash
tar xzf dataset_meta/kradar-labels-meta.tar.gz -C /hy-tmp/kradar_dataset/
python tools/pc1p_to_raf.py     # 先改脚本里 ZIP / OUT 两个路径
```

### 4. 启动服务

```bash
cd external/RAF            # 或你本地的 RAF 仓库，需与 configs/ 中的路径一致
PYTHONPATH=$PWD/ops:$PWD setsid nohup python3 ../../tools/demo_server.py --port 8080 \
    > /tmp/site_server.log 2>&1 < /dev/null &
```

打开 `http://localhost:8080` 是审计报告，`http://localhost:8080/demo` 是交互 Demo。

首次启动约 25 秒（加载数据集索引 + 权重 + CUDA 预热），日志出现 `[demo] ready` 即可用。

> `demo_server.py` 顶部的 `WEB_DIR`、`AUDIT_DIR`、`CKPT` 以及配置里的数据路径都是**绝对路径**（对应本次实测的目录结构），迁移到别的机器时请先按实际情况修改。

---

## 复现实验

### 雷达 × 天气统计

```bash
cd external/RAF && PYTHONPATH=$PWD/ops:$PWD python3 ../../tools/radar_weather_stats.py
```

58 序列、每序列前 200 帧，共 11,600 帧：

| 天气 | 帧数 | ROI 内点/帧 | 平均功率 |
|---|---:|---:|---:|
| Normal | 3,800 | 2,003 | 0.5388 |
| Overcast | 400 | 2,447 | 0.5032 |
| Rain | 1,600 | 1,981 | 0.4521 |
| Sleet | 2,400 | 2,226 | 0.2715 |
| LightSnow | 800 | 2,431 | 0.1634 |
| Fog | 1,200 | 1,881 | 0.0717 |
| HeavySnow | 1,400 | 1,900 | 0.0346 |

注意：每帧点数恒为 10,240（`sample_points` 上限封顶），点云稀疏度无法用点数衡量，只有功率有区分度。

### seq 1 端到端评测

```bash
cd external/RAF && PYTHONPATH=$PWD/ops:$PWD python3 ../../tools/eval_seq1.py
```

193 个有效帧 / 254 个 GT，L4DR v1.1 权重：

| score 阈值 | IoU | Recall | Precision |
|---|---|---:|---:|
| ≥ 0.1 | 0.3 | 54.3% | 64.8% |
| ≥ 0.2 | 0.3 | 39.4% | 70.4% |
| ≥ 0.3 | 0.3 | 28.0% | 71.0% |
| ≥ 0.1 | 0.5 | 15.0% | 17.8% |
| ≥ 0.2 | 0.5 | 7.9% | 14.1% |
| ≥ 0.3 | 0.5 | 3.5% | 9.0% |

推理耗时 103 ms/帧（9.7 FPS）。**这些数字不能与论文的 AP 直接比较**，原因见「已知限制」。

---

## 在线 Demo

| 路由 | 内容 |
|---|---|
| `/` | 审计报告 |
| `/demo` | 交互页：选帧 / 上传 / BEV 结果 / AI 分析 |
| `/api/meta` | 帧列表与天气信息 |
| `/api/camera?idx=N` | 指定帧的相机图 |
| `/api/infer` | 单帧推理，返回 BEV 图与检测框 |
| `/api/analyze` | 结果分析（本地规则 / 多模态大模型） |

**上传格式**

| 输入 | 格式 |
|---|---|
| 雷达 | `.npy`，`[N,4]` float32 = `x, y, z, power` |
| LiDAR | `.npy` `[N,≥3]` 或 ASCII `.pcd`（自动解析 x/y/z/intensity） |
| 相机 | `.jpg` / `.png`，建议 16:9 前视图 |

**注意三种传感器的帧号并不相同**（seq 1 实测）：

| | 关系 | 第 0 帧 |
|---|---|---|
| 雷达 | 标签前缀 | `00033` |
| LiDAR | 雷达 − 32 | `00001` |
| 相机 | ≈ 3 × 雷达 | `00002` |

页面上会直接列出当前帧的三者编号，按表取文件即可。

**AI 分析（可选）**：在 `tools/ai.env` 中填入 OpenAI 兼容网关配置，重启后「分析检测结果」按钮会改用多模态模型；不配置则走本地规则分析，不发送任何数据出机器。

```ini
AI_API_KEY=sk-...
AI_BASE_URL=https://your-gateway/v1
AI_MODEL=your-vision-model
AI_TIMEOUT=90
```

---

## 已知限制

1. **雷达密度不足**：公开的 `pc1p.zip` 是极坐标 1% 稀疏雷达，ROI 内约 1,400 点/帧；论文训练用笛卡尔稀疏雷达固定 10,240 点/帧。这是本地指标显著低于论文的主因，而非模型或数据错误。
2. **指标口径不同**：论文报 AP<sub>BEV</sub> / AP<sub>3D</sub>（41 点积分），本仓库报单阈值 Recall/Precision。
3. **只覆盖 seq 1**：只有 seq 1 具备 LiDAR 与相机；其余 57 个序列只有标签与雷达。
4. **RAF 变体未复现**：Frozen / RAF 依赖作者自训的 LiDAR-雷达骨干权重（代码中路径写死且未公开）。
5. **帧数口径**：`remove_0_obj: True` 会过滤无目标帧，论文报告的帧数是标签总数。

---

## 已知坑

- **`nms` 包**：`nms.py` 里对 tuple 的处理会抛 `TypeError`，且会 `print` 刷屏，需要打补丁。
- **CUDA 算子**：仓库自带的 `.so` 是用旧版 torch 编译的，在 torch 1.12 下会报 `undefined symbol`；需要重新编译或替换。
- **导入顺序**：必须先 `import torch` 再导入 nms / 算子，否则 `libc10.so` 找不到。
- **退出崩溃**：进程结束时偶发 `free(): invalid pointer`（spconv/mmcv 析构问题），不影响产物。

---

## 数据与权重来源

- 论文：arXiv `2607.04587`（ECCV 2026）
- RAF 官方代码：[parkie0517/RAF](https://github.com/parkie0517/RAF)
- L4DR：[ylwhxht/L4DR](https://github.com/ylwhxht/L4DR)（baseline 数字来源）
- K-Radar：[kaist-avelab/K-Radar](https://github.com/kaist-avelab/K-Radar)
- 3D-LRF 原始报告值：[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Chae_Towards_Robust_3D_Object_Detection_with_LiDAR_and_4D_Radar_CVPR_2024_paper.pdf)
- L4DR K-Radar 权重：[hx24 @ HuggingFace](https://huggingface.co/hx24)

> 审计只回答「论文报告的数据是否准确、是否可复现」，不对方法创新性做评价。
