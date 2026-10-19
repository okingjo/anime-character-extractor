# 动漫人物训练集提取器 / Anime Character Dataset Extractor

全自动: 动漫视频(mp4/mkv…)或图片(jpg/png, 支持黑白漫画页) → 人物检测 → 裁剪(全身+头部) → 去重 → 去背景 → CCIP 聚类 → **按角色分文件夹输出训练集**。

Fully automatic: anime videos (mp4/mkv…) or images (jpg/png, monochrome manga pages supported) → character detection → cropping (full-body + head) → deduplication → background removal → CCIP clustering → **training dataset organized into per-character folders**.

- 🇨🇳 [中文说明](#中文说明)
- 🇬🇧 [English](#english)

---

## 中文说明

### 启动 (免安装, 不污染本地环境)

```bat
双击 启动GUI.bat
```
或命令行:
```bat
cd anime-character-extractor
uv run extractor.py
```
- 首次运行 [uv](https://docs.astral.sh/uv/) 会自动创建隔离环境并安装依赖 (dghs-imgutils / onnxruntime / av / gradio 等), 模型权重首次使用时自动从 HuggingFace 下载并缓存。
- **所有数据都存放在 repo 目录内, 不占用 C 盘**:
  - `.venv/` — 虚拟环境 (gradio/imgutils 等所有依赖)
  - `runtime/uv_cache/` — uv 包下载缓存
  - `runtime/python/` — uv 管理的 Python 解释器 (通过 启动GUI.bat 的环境变量指定)
  - `runtime/hf_cache/` — HuggingFace 模型缓存 (CCIP/YOLO/isnetis 权重, 约 1GB)
  - 手动命令行启动时请先设置环境变量 (启动GUI.bat 已自动设置): `UV_CACHE_DIR`、`UV_PYTHON_INSTALL_DIR`、`HF_HOME`
- 之后浏览器自动打开 `http://127.0.0.1:7861`。
- 国内网络如 HF 下载慢, 可在 启动GUI.bat 中取消 `HF_ENDPOINT=https://hf-mirror.com` 一行的注释。
- 需要 Python 3.11 或 3.12 (uv 会自动下载到 runtime/python, 无需手动安装)。

### 技术方案 (调研结论)

角色区分采用 **imgutils CCIP** (Contrastive Character Image Pretraining)。调研过的替代方案:

| 方案 | 结论 |
|---|---|
| **CCIP** (deepghs) | ✅ 采用。无监督、开箱即用、任意番剧通用。使用基准中 F1 最高的 `ccip-caformer_b36-24` (F1=0.941, 高于 imgutils 默认的 pruned-24 的 0.917) |
| CAST (SIGGRAPH Asia 2022) | ❌ 需针对每部动画自监督训练, 不"全自动" |
| DINOv2 角色库少样本检索 (2025) | ❌ 需要先有人工构建的角色库(exemplars) |
| Dan:mul 多模态 (IJCNN 2023) | ❌ 需标注数据训练分类器 |

处理管线:
1. **抽帧**: PyAV 解码(自带, 无需系统 ffmpeg), 按秒间隔采样
2. **检测**: `detect_person`(YOLOv8, AniDet3+v1.1) 找全身; `detect_heads`(v2.0) 找头部并与全身框关联
3. **裁剪策略**:
   - 默认输出 **全身(body)** + **同帧关联的头部(head)** 两种裁剪
   - **仅当画面检测不到全身、只有头/脸可见时**, 才降级输出 head_only / face 裁剪 (单独聚类到 `faceonly_XX/`)
   - 头部裁剪低于最小分辨率标准 → 直接丢弃, 不影响全身
4. **去重**: dHash 感知哈希, 剔除近似静止镜头的重复帧
5. **分割**: isnetis 动漫抠图去背景, 白底单角色图正是 CCIP 的训练分布, 显著提升聚类质量(尤其黑白漫画)
6. **聚类**: CCIP 特征 + OPTICS/DBSCAN (eps/min_samples 用官方按模型调优的推荐值)
7. **过滤输出**: 簇内全身图 < 最小张数 → 视为路人丢弃; 每角色可设上限均匀采样

### 输出结构

```
输出目录/
├─ character_01/
│  ├─ body/            全身 (RGBA 透明背景 PNG)
│  └─ head/            头部
├─ character_02/ ...
├─ faceonly_01/        仅头/脸可见时降级提取的组
│  ├─ head_only/
│  └─ face/
├─ _rejected/          (可选) 被路人过滤丢弃的小簇
└─ report.json         统计 + 全部参数
```

### 关键参数速查

GUI 内每个参数都有推荐值与调大/调小效果说明:

| 参数 | 推荐 | 说明 |
|---|---|---|
| 抽帧间隔 | 0.5~1.0s | 调小样本多但慢、重复多 |
| 全身最小边长 | 160px | 训练集质量 vs 数量的主要开关 |
| 头部最小边长 | 80px | 达不到即丢弃该头部裁剪 |
| dHash 去重距离 | 6 | 调大去重狠, -1 关闭 |
| isnetis 去背景 | 开 | 黑白漫画/复杂背景强烈建议开 |
| CCIP 模型 | b36-24 | F1 最高 |
| eps / min_samples | 0 (自动) | 样本 <100 张时 min_samples 手动降到 2~3 |
| 最小簇张数 | 8 | 路人过滤阈值 |
| 每角色最大张数 | 400 | 0=不限 |

### 性能参考

- **GPU 加速 (NVIDIA, 推荐)**: `启动GUI.bat` 检测到 nvidia-smi 会自动 `uv sync --extra gpu`; 手动启用: `uv sync --extra gpu`。无需安装 CUDA Toolkit (依赖里带 pip 版 cublas/cudnn/cufft 运行库)。RTX 5090 实测:

| 阶段 | CPU | GPU | 加速比 |
|---|---|---|---|
| CCIP b36 特征提取 | ~640ms/张 | ~25-37ms/张 | **~17-25x** |
| detect_person 检测 | ~350ms/张 | ~32ms/张 | **~11x** |
| isnetis 分割 | ~600ms/张 | ~61ms/张 | **~10x** |
| 全流程 E2E 示例 (66裁剪) | ~157s | ~19s | **~8x** |

- 首次运行每个模型有 5~10s 的 CUDA 初始化 (已用 cuDNN HEURISTIC 卷积搜索把 CCIP warmup 从 ~100s 压到 ~6s)。
- 想强制 CPU: `set ONNX_MODE=cpu` 后再启动。
- 检测阶段约占 60% 时间。1 小时番剧 @1s 间隔 ≈ 3600 帧: CPU 预计 1~3 小时, GPU 预计 10~25 分钟。

### 已知限制

- CCIP 对**同一角色换装/换发型**可能拆成两簇; 对**双胞胎/制服群众**可能并簇。可调 eps 或事后手动合并文件夹。
- `detect_person` 对艺术插画类(非动画截图)召回稍弱, 官方仍在迭代。
- isnetis 是全身角色分割模型, 对**纯头部特写**可能分割失败 → 自动回退保存 RGB 原图。
- 黑白漫画页: 建议开启 isnetis 去背景, 否则网点纸背景会干扰特征。

---

## English

### Quick Start (no installation, isolated environment)

Double-click `启动GUI.bat`, or from a terminal:

```bat
cd anime-character-extractor
uv run extractor.py
```

- On first run, [uv](https://docs.astral.sh/uv/) automatically creates an isolated virtual environment and installs dependencies (dghs-imgutils / onnxruntime / av / gradio, etc.). Model weights are downloaded from HuggingFace on first use and cached.
- **Everything stays inside the repo folder — nothing is written to C:**
  - `.venv/` — the virtual environment (all dependencies)
  - `runtime/uv_cache/` — uv package download cache
  - `runtime/python/` — uv-managed Python interpreter (set via env vars in `启动GUI.bat`)
  - `runtime/hf_cache/` — HuggingFace model cache (CCIP/YOLO/isnetis weights, ~1GB)
  - When launching manually from a terminal, set `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` and `HF_HOME` first (the .bat does this automatically).
- Your browser then opens `http://127.0.0.1:7861`.
- If HuggingFace downloads are slow, uncomment the `HF_ENDPOINT=https://hf-mirror.com` line in `启动GUI.bat` (or use any mirror).
- Requires Python 3.11 or 3.12 (uv downloads it into runtime/python automatically — no manual install needed).

### Approach

Character discrimination uses **imgutils CCIP** (Contrastive Character Image Pretraining). Alternatives surveyed:

| Approach | Verdict |
|---|---|
| **CCIP** (deepghs) | ✅ Adopted. Unsupervised, works out-of-the-box on any series. Uses `ccip-caformer_b36-24`, the highest-F1 model in the official benchmark (F1=0.941 vs 0.917 for imgutils' default pruned-24) |
| CAST (SIGGRAPH Asia 2022) | ❌ Requires per-show self-supervised training — not "fully automatic" |
| DINOv2 character-bank few-shot retrieval (2025) | ❌ Requires a manually built exemplar bank per movie |
| Dan:mul multimodal (IJCNN 2023) | ❌ Requires labeled data to train a classifier |

Pipeline:
1. **Frame sampling**: PyAV decoding (bundled, no system ffmpeg needed), sampled at a configurable interval
2. **Detection**: `detect_person` (YOLOv8, AniDet3+v1.1) for full bodies; `detect_heads` (v2.0) for heads, associated with body boxes
3. **Cropping policy**:
   - By default outputs **full-body** + **head** crops associated within the same frame
   - **Only when no full body is detected and just a head/face is visible** does it fall back to head_only / face crops (clustered separately into `faceonly_XX/`)
   - Head crops below the minimum resolution are discarded without affecting the body crop
4. **Deduplication**: dHash perceptual hashing removes near-identical frames from static shots
5. **Segmentation**: isnetis anime matting removes backgrounds; white-background single-character images match CCIP's training distribution and substantially improve clustering quality (especially for monochrome manga)
6. **Clustering**: CCIP features + OPTICS/DBSCAN (eps/min_samples default to the officially tuned per-model values)
7. **Filtered output**: clusters with fewer than the minimum number of body images are treated as background characters and dropped; a per-character cap with uniform sampling is available

### Output Layout

```
output_dir/
├─ character_01/
│  ├─ body/            full-body crops (RGBA transparent PNG)
│  └─ head/            head crops
├─ character_02/ ...
├─ faceonly_01/        fallback group (only head/face visible)
│  ├─ head_only/
│  └─ face/
├─ _rejected/          (optional) small clusters dropped by the background-character filter
└─ report.json         statistics + all parameters used
```

### Key Parameters

Every parameter in the GUI shows a recommended value plus the effect of increasing/decreasing it:

| Parameter | Recommended | Notes |
|---|---|---|
| Frame interval | 0.5–1.0s | Smaller = more samples but slower and more duplicates |
| Min body side | 160px | Main quality-vs-quantity knob |
| Min head side | 80px | Head crops below this are discarded |
| dHash distance | 6 | Larger = more aggressive dedup; -1 disables |
| isnetis background removal | On | Strongly recommended for manga / busy backgrounds |
| CCIP model | b36-24 | Highest F1 |
| eps / min_samples | 0 (auto) | For <100 samples, set min_samples manually to 2–3 |
| Min cluster size | 8 | Background-character filter threshold |
| Max images per character | 400 | 0 = unlimited |

### Performance

- **GPU acceleration (NVIDIA, recommended)**: `启动GUI.bat` auto-detects nvidia-smi and runs `uv sync --extra gpu`; manual: `uv sync --extra gpu`. No CUDA Toolkit install needed (pip-provided cublas/cudnn/cufft runtime DLLs are bundled as dependencies). Measured on RTX 5090:

| Stage | CPU | GPU | Speedup |
|---|---|---|---|
| CCIP b36 feature extraction | ~640ms/img | ~25-37ms/img | **~17-25x** |
| detect_person | ~350ms/img | ~32ms/img | **~11x** |
| isnetis segmentation | ~600ms/img | ~61ms/img | **~10x** |
| Full E2E sample (66 crops) | ~157s | ~19s | **~8x** |

- Each model has a 5–10s CUDA init on first use (cuDNN HEURISTIC conv search cuts CCIP warmup from ~100s to ~6s).
- Force CPU: `set ONNX_MODE=cpu` before launching.
- Detection accounts for ~60% of total time. A 1-hour episode @1s interval ≈ 3600 frames: expect 1–3 hours on CPU, 10–25 minutes on GPU.

### Known Limitations

- CCIP may split **the same character with different outfits/hairstyles** into two clusters, or merge **twins / uniformed crowds**. Tune eps or merge folders manually afterwards.
- `detect_person` has slightly lower recall on artistic illustrations (vs animation screenshots).
- isnetis is a full-character matting model; it may fail on **pure head close-ups** — the pipeline automatically falls back to saving the RGB crop.
- For monochrome manga pages, keep isnetis background removal on, otherwise screentone backgrounds interfere with features.

---

Built on [deepghs/imgutils](https://github.com/deepghs/imgutils). CCIP models trained by [7eu7d7](https://github.com/7eu7d7).
