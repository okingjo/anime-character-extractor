"""
anime-character-extractor — 全自动动漫人物训练集提取器

从动漫视频(mp4/mkv等)或图片(jpg/png, 支持黑白)中:
  1. 抽帧 / 读图
  2. 检测人物(detect_person) + 头部(detect_heads), 必要时降级到脸部(detect_faces)
  3. 裁剪: 全身 + 头部 (仅当只检测到脸时才输出脸部裁剪)
  4. 近似重复去重 (dHash)
  5. 可选 isnetis 分割去背景
  6. CCIP (ccip-caformer_b36-24, F1 0.941) 特征提取 + 聚类
  7. 按角色输出到文件夹, 过滤出现次数过少的路人簇

启动方式 (免安装, uv 在 repo 目录内建隔离环境, 不占用 C 盘):
    uv run extractor.py
或双击  启动GUI.bat
"""
import io
import json
import os
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 所有缓存/模型/环境都放在 repo 目录内 (必须在导入 huggingface 相关库之前设置)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = REPO_ROOT / 'runtime'
RUNTIME_DIR.mkdir(exist_ok=True)
os.environ.setdefault('HF_HOME', str(RUNTIME_DIR / 'hf_cache'))            # HF 模型缓存
os.environ.setdefault('HF_HUB_CACHE', str(RUNTIME_DIR / 'hf_cache' / 'hub'))
os.environ.setdefault('XDG_CACHE_HOME', str(RUNTIME_DIR / 'xdg_cache'))     # 其它杂项缓存
os.environ.setdefault('GRADIO_TEMP_DIR', str(RUNTIME_DIR / 'gradio_tmp'))   # gradio 临时文件
os.environ.setdefault('UV_CACHE_DIR', str(RUNTIME_DIR / 'uv_cache'))        # uv 包缓存 (对子进程生效)
os.environ.setdefault('UV_PYTHON_INSTALL_DIR', str(RUNTIME_DIR / 'python')) # uv 管理的 Python
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def _setup_gpu_dll_dirs():
    """Windows + onnxruntime-gpu: 把 pip 安装的 nvidia cublas/cudnn/cufft DLL 目录
    注册进 DLL 搜索路径, 否则 CUDAExecutionProvider 创建失败并静默回退 CPU."""
    if os.name != 'nt':
        return
    try:
        import glob
        import sysconfig
        sp = sysconfig.get_paths()['purelib']  # 正确的 site-packages 路径
        for d in glob.glob(os.path.join(sp, 'nvidia', '*', 'bin')):
            os.add_dll_directory(d)
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass


_setup_gpu_dll_dirs()


def _patch_onnx_cuda_options():
    """给 imgutils 的 ONNX session 注入 CUDA provider options:
    - cudnn_conv_algo_search=HEURISTIC: 默认 EXHAUSTIVE 会对每种输入尺寸穷举搜索卷积算法,
      首次 warmup 高达 ~100s; HEURISTIC 降到 ~6s 且稳态推理略快 (实测 37ms -> 25ms/张).
    - arena_extend_strategy=kSameAsRequested: 防止显存 arena 过度扩张.
    CPU-only 环境下为无操作."""
    try:
        import imgutils.utils.onnxruntime as iort
        from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

        def _open_patched(ckpt, provider, use_cpu=True):
            opts = SessionOptions()
            opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            if provider == 'CPUExecutionProvider':
                opts.intra_op_num_threads = os.cpu_count()
                providers = [provider]
            else:
                providers = [(provider, {'cudnn_conv_algo_search': 'HEURISTIC',
                                         'arena_extend_strategy': 'kSameAsRequested'}),
                             'CPUExecutionProvider']
            return InferenceSession(ckpt, opts, providers=providers)

        iort._open_onnx_model = _open_patched
    except Exception:
        pass  # CPU-only 或版本差异时不影响主流程

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.ts', '.mpg', '.mpeg'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

CCIP_MODELS = {
    'ccip-caformer_b36-24 (最强, F1=0.941, 推荐)': 'ccip-caformer_b36-24',
    'ccip-caformer-24-randaug-pruned (imgutils默认, F1=0.917, 稍快)': 'ccip-caformer-24-randaug-pruned',
    'ccip-caformer-6-randaug-pruned_fp32 (轻量, F1较低)': 'ccip-caformer-6-randaug-pruned_fp32',
}
CCIP_DEFAULT_MODEL = 'ccip-caformer_b36-24'

# 头部检测模型 (deepghs/anime_head_detection 仓库实际存在的模型)
HEAD_MODELS = {
    'head_detect_v2.0_s (默认, 均衡)': 'head_detect_v2.0_s',
    'head_detect_v2.0_n (最快, 精度略低)': 'head_detect_v2.0_n',
    'head_detect_v2.0_m (更准, 较慢)': 'head_detect_v2.0_m',
    'head_detect_v2.0_l (最准, 慢)': 'head_detect_v2.0_l',
    'head_detect_v1.0_s (旧版)': 'head_detect_v1.0_s',
}

# 各模型对应的默认聚类参数 (来自 deepghs/ccip_onnx 仓库 metrics)
# model -> {method: (eps, min_samples)}
CLUSTER_DEFAULTS = {
    'ccip-caformer_b36-24': {
        'dbscan': (0.213231, 2),
        'optics': (0.5, 5),
        'optics_best': (0.21196399114951675, 5),
    },
    'ccip-caformer-24-randaug-pruned': {
        'dbscan': (0.17847511429108218, 2),
        'dbscan_2': (0.12921094122454668, 2),
        'dbscan_free': (0.1291187648928262, 2),
        'optics': (0.5, 5),
        'optics_best': (0.1836453739562513, 5),
    },
    'ccip-caformer-6-randaug-pruned_fp32': {
        'dbscan': (0.1951224011983088, 2),
        'optics': (0.5, 5),
        'optics_best': (0.5, 5),
    },
}


# ---------------------------------------------------------------------------
# 参数定义
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # 输入
    input_path: str = ''
    input_mode: str = '自动检测'      # 自动检测 / 视频 / 图片
    # 抽帧
    frame_interval: float = 1.0        # 秒
    max_frames: int = 3000
    # 人物检测 (detect_person)
    person_version: str = 'v1.1'       # v0 / v1 / v1.1
    person_level: str = 's'            # n / s / m / x
    person_conf: float = 0.35
    person_iou: float = 0.5
    # 头部检测 (detect_heads)
    head_model: str = 'head_detect_v2.0_s (默认)'
    head_conf: float = 0.4
    head_iou: float = 0.7
    # 脸部检测 (detect_faces, 仅降级兜底用)
    face_version: str = 'v1.4'         # v0 / v1 / v1.3 / v1.4
    face_level: str = 's'              # n / s
    face_conf: float = 0.3
    face_iou: float = 0.7
    # 分辨率过滤
    min_body_res: int = 160            # 全身裁剪最短边 px
    min_head_res: int = 80             # 头部裁剪最短边 px
    min_face_res: int = 64             # 降级脸部裁剪最短边 px
    crop_pad_ratio: float = 0.02       # 裁剪框外扩比例
    # 去重
    dedup_hash_dist: int = 6           # dHash 汉明距离 <= 该值视为重复
    # 分割 (isnetis)
    use_segmentation: bool = True      # isnetis 去背景后再提特征
    save_rgba: bool = True             # 输出 RGBA 透明背景图
    seg_min_input_side: int = 480      # 送入 isnetis 前的最小放大边长
    # CCIP 特征
    ccip_model_label: str = 'ccip-caformer_b36-24 (最强, F1=0.941, 推荐)'
    ccip_size: int = 384               # 特征提取输入尺寸
    # 聚类
    cluster_method: str = 'optics_best'
    cluster_eps: float = 0.0           # 0 = 使用模型推荐值
    cluster_min_samples: int = 0       # 0 = 使用模型推荐值
    # 输出过滤
    min_cluster_count: int = 8         # 簇内图片少于此数 => 视为路人, 不输出
    max_per_character: int = 400       # 每个角色最多保留张数, 0=不限
    keep_rejected: bool = False        # 是否把被过滤的小簇放到 _rejected/
    output_dir: str = ''

    @property
    def ccip_model(self) -> str:
        return CCIP_MODELS.get(self.ccip_model_label, CCIP_DEFAULT_MODEL)

    @property
    def head_model_name(self) -> str:
        return HEAD_MODELS.get(self.head_model, 'head_detect_v2.0_s')

    def resolved_eps_min_samples(self):
        table = CLUSTER_DEFAULTS.get(self.ccip_model, {})
        defaults = table.get(self.cluster_method) or table.get('dbscan') or (0.5, 5)
        eps = self.cluster_eps if self.cluster_eps > 0 else defaults[0]
        ms = self.cluster_min_samples if self.cluster_min_samples > 0 else defaults[1]
        return eps, ms


# ---------------------------------------------------------------------------
# 裁剪记录
# ---------------------------------------------------------------------------
@dataclass
class Crop:
    kind: str                 # 'body' | 'head' | 'face'
    source: str               # 来源文件
    frame_idx: int            # 帧号(图片输入为 -1)
    box: tuple                # (x0,y0,x1,y1) 在原图中的位置
    image: object = None      # PIL.Image (RGB)
    seg_image: object = None  # PIL.Image (RGBA, 分割后) 可选
    feature: object = None    # CCIP 特征向量
    cluster_id: int = -2      # -2=未聚类, -1=噪声
    group_key: str = ''       # 关联键: 同帧同人物 body/head 共享
    saved_path: str = ''


# ---------------------------------------------------------------------------
# 日志器 (线程安全, 供 GUI 轮询)
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, maxlen=2000):
        self.lines = []
        self.lock = threading.Lock()
        self.maxlen = maxlen
        self.progress = 0.0
        self.stage = ''

    def log(self, msg):
        ts = time.strftime('%H:%M:%S')
        with self.lock:
            self.lines.append(f'[{ts}] {msg}')
            if len(self.lines) > self.maxlen:
                self.lines = self.lines[-self.maxlen:]
        print(msg, flush=True)

    def set_progress(self, frac, stage=''):
        self.progress = max(0.0, min(1.0, frac))
        if stage:
            self.stage = stage

    def snapshot(self):
        with self.lock:
            return '\n'.join(self.lines[-400:]), self.progress, self.stage


# ---------------------------------------------------------------------------
# 输入收集
# ---------------------------------------------------------------------------
def collect_videos(path: Path):
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in VIDEO_EXTS else []
    vids = []
    for f in sorted(p.rglob('*')):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            vids.append(f)
    return vids


def collect_images(path: Path):
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []
    imgs = []
    for f in sorted(p.rglob('*')):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            imgs.append(f)
    return imgs


def detect_input_mode(path: Path):
    vids, imgs = collect_videos(path), collect_images(path)
    if vids and not imgs:
        return '视频', vids
    if imgs and not vids:
        return '图片', imgs
    if vids and imgs:
        # 混合: 都处理
        return '混合', vids + imgs
    return '空', []


# ---------------------------------------------------------------------------
# 视频抽帧 (PyAV, 无需系统 ffmpeg)
# ---------------------------------------------------------------------------
def iter_video_frames(video_path: Path, interval_sec: float, max_frames: int, logger: Logger):
    import av
    import numpy as np
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        step = max(1, int(round(fps * interval_sec)))
        duration = float(container.duration / av.time_base) if container.duration else 0
        logger.log(f'  {video_path.name}: {duration:.0f}s, fps={fps:.2f}, 每{step}帧取1帧')
        idx, yielded = 0, 0
        for frame in container.decode(stream):
            if idx % step == 0:
                img = frame.to_image()  # PIL RGB
                yield idx, img
                yielded += 1
                if max_frames and yielded >= max_frames:
                    logger.log(f'  达到单视频最大帧数 {max_frames}, 停止')
                    break
                if duration > 0 and yielded % 50 == 0:
                    t = idx / fps
                    logger.set_progress(min(0.9, t / max(duration, 1) * 0.5), '抽帧/检测')
            idx += 1
    finally:
        container.close()


# ---------------------------------------------------------------------------
# 检测 + 裁剪核心
# ---------------------------------------------------------------------------
def _crop(img, box, pad_ratio=0.02):
    from PIL import Image
    w, h = img.size
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad_ratio), int((y1 - y0) * pad_ratio)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def _min_side(box):
    return min(box[2] - box[0], box[3] - box[1])


def _center_in(box, outer):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (sa + sb - inter + 1e-9)


def process_frame(img, source: str, frame_idx: int, params: Params, logger: Logger):
    """对一帧/一张图做检测并生成 Crop 列表 (body / head / face-fallback)."""
    from imgutils.detect import detect_person, detect_heads, detect_faces
    crops = []
    pad = params.crop_pad_ratio
    try:
        persons = detect_person(img, level=params.person_level, version=params.person_version,
                                conf_threshold=params.person_conf, iou_threshold=params.person_iou)
    except Exception as e:
        logger.log(f'  person检测失败({source}#{frame_idx}): {e}')
        persons = []
    try:
        heads = detect_heads(img, model_name=params.head_model_name,
                             conf_threshold=params.head_conf, iou_threshold=params.head_iou)
    except Exception as e:
        logger.log(f'  head检测失败({source}#{frame_idx}): {e}')
        heads = []

    used_head_boxes = []

    for pi, (pbox, _, pconf) in enumerate(persons):
        # 找到落在这个 person 框内的 heads
        heads_inside = [hb for (hb, _, _) in heads if _center_in(hb, pbox)]
        if len(heads_inside) >= 2:
            # 一个 person 框套了多个头 => 检测把人群框在一起了, 跳过 body, 头单独处理
            continue
        group_key = f'{source}#{frame_idx}#p{pi}'
        # ---- body ----
        if _min_side(pbox) >= params.min_body_res:
            c = _crop(img, pbox, pad)
            if c:
                crop_img, real_box = c
                crops.append(Crop('body', source, frame_idx, real_box, crop_img, group_key=group_key))
        # ---- head (关联到 body) ----
        if heads_inside:
            hbox = heads_inside[0]
            used_head_boxes.append(hbox)
            if _min_side(hbox) >= params.min_head_res:
                c = _crop(img, hbox, pad)
                if c:
                    crop_img, real_box = c
                    crops.append(Crop('head', source, frame_idx, real_box, crop_img, group_key=group_key))

    # ---- 未关联到 person 的 head: 画面里只有头/脸可见的降级情况 ----
    for (hbox, _, hconf) in heads:
        if any(_iou(hbox, ub) > 0.5 for ub in used_head_boxes):
            continue
        # 已被某个 body 覆盖的也跳过(body存在说明不是"只有脸"的情况)
        if any(_center_in(hbox, pbox) for (pbox, _, _) in persons):
            continue
        group_key = f'{source}#{frame_idx}#h{hbox[0]}_{hbox[1]}'
        if _min_side(hbox) >= params.min_head_res:
            c = _crop(img, hbox, pad)
            if c:
                crop_img, real_box = c
                crops.append(Crop('head_only', source, frame_idx, real_box, crop_img, group_key=group_key))
        elif _min_side(hbox) >= params.min_face_res:
            c = _crop(img, hbox, pad)
            if c:
                crop_img, real_box = c
                crops.append(Crop('face', source, frame_idx, real_box, crop_img, group_key=group_key))

    # ---- head 检测不到但 person 也没有 => 尝试 face 兜底 ----
    if not crops and not persons:
        try:
            faces = detect_faces(img, level=params.face_level, version=params.face_version,
                             conf_threshold=params.face_conf, iou_threshold=params.face_iou)
        except Exception:
            faces = []
        for fi, (fbox, _, fconf) in enumerate(faces):
            if _min_side(fbox) >= params.min_face_res:
                c = _crop(img, fbox, pad)
                if c:
                    crop_img, real_box = c
                    crops.append(Crop('face', source, frame_idx, real_box, crop_img,
                                      group_key=f'{source}#{frame_idx}#f{fi}'))
    return crops


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------
class Deduper:
    def __init__(self, max_dist: int):
        import imagehash
        self.imagehash = imagehash
        self.max_dist = max_dist
        self.hashes = []  # list of imagehash

    def is_dup(self, pil_img) -> bool:
        if self.max_dist < 0:
            return False
        try:
            h = self.imagehash.dhash(pil_img.resize((64, 64)))
        except Exception:
            return False
        for old in self.hashes:
            if (h - old) <= self.max_dist:
                return True
        self.hashes.append(h)
        # 防止极长视频内存膨胀: 只保留最近 20000 个
        if len(self.hashes) > 20000:
            self.hashes = self.hashes[-15000:]
        return False


# ---------------------------------------------------------------------------
# 分割
# ---------------------------------------------------------------------------
def segment_crop(crop_img, scale=1024, min_input_side=480):
    """isnetis 去背景, 返回 RGBA (模型输入固定 1024, scale 不能改).
    小图直接送入会因 padding 占比过大导致 mask 全空, 故先放大;
    分割结果无效(过空/过满)时返回 None, 调用方回退原图."""
    import numpy as np
    from PIL import Image
    from imgutils.segment import segment_rgba_with_isnetis
    try:
        img = crop_img
        w, h = img.size
        short = min(w, h)
        if short < min_input_side:
            ratio = min_input_side / short
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        _, rgba = segment_rgba_with_isnetis(img, scale=scale)
        # 还原到原裁剪尺寸
        if rgba.size != (w, h):
            rgba = rgba.resize((w, h), Image.LANCZOS)
        alpha = np.asarray(rgba)[..., 3]
        fg_ratio = float((alpha > 8).mean())
        if fg_ratio < 0.01 or fg_ratio > 0.995:
            return None  # mask 无效
        return rgba
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------
def cluster_crops(crops, params: Params, logger: Logger):
    """对一组 Crop(已带 feature)聚类, 就地写入 cluster_id."""
    import numpy as np
    from sklearn.cluster import DBSCAN, OPTICS
    from imgutils.metrics import ccip_batch_differences

    if not crops:
        return
    feats = [c.feature for c in crops]
    n = len(feats)
    eps, min_samples = params.resolved_eps_min_samples()
    logger.log(f'  聚类 {n} 个样本: method={params.cluster_method}, eps={eps:.4f}, min_samples={min_samples}')

    # 分块计算距离矩阵 (n^2 内存保护)
    # ccip_batch_differences(a + b) 返回 (len(a)+len(b))^2 矩阵, 取左上->右下块 [0:k, k:k+n]
    CHUNK = 512
    diff = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, CHUNK):
        j_end = min(n, i + CHUNK)
        k = j_end - i
        sub = ccip_batch_differences(feats[i:j_end] + feats, model=params.ccip_model)
        diff[i:j_end, :] = sub[:k, k:]
        logger.set_progress(0.75 + 0.15 * (j_end / n), '聚类')
    diff = (diff + diff.T) / 2
    np.fill_diagonal(diff, 0)

    samples = np.arange(n).reshape(-1, 1)
    metric = 'precomputed'
    if params.cluster_method.startswith('dbscan'):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(diff)
    else:
        model = OPTICS(max_eps=min(eps, 0.5) if eps <= 0.5 else 0.5,
                       min_samples=min_samples, metric=metric).fit(diff)
    for c, lab in zip(crops, model.labels_):
        c.cluster_id = int(lab)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_pipeline(params: Params, logger: Logger, stop_event: threading.Event):
    import numpy as np
    from PIL import Image
    _patch_onnx_cuda_options()  # GPU: 注入 HEURISTIC 卷积搜索, 大幅缩短 warmup
    from imgutils.metrics import ccip_extract_feature

    # 报告当前使用的推理后端
    try:
        from imgutils.utils.onnxruntime import get_onnx_provider
        import onnxruntime as ort
        active = get_onnx_provider()
        logger.log(f'ONNX 推理后端: {active} (onnxruntime {ort.__version__})')
        if active == 'CPUExecutionProvider' and 'CUDAExecutionProvider' in ort.get_available_providers():
            logger.log('提示: 检测到 CUDA 可用但未启用, 请确认已 uv sync --extra gpu')
        elif active == 'CPUExecutionProvider':
            logger.log('提示: 当前为 CPU 推理。NVIDIA 显卡可执行 `uv sync --extra gpu` 启用 GPU 加速 (约10~17倍)')
    except Exception:
        pass

    t_start = time.time()
    input_path = Path(params.input_path.strip().strip('"'))
    if not input_path.exists():
        raise FileNotFoundError(f'输入路径不存在: {input_path}')

    # -- 输出目录 --
    if params.output_dir.strip():
        out_root = Path(params.output_dir.strip().strip('"'))
    else:
        out_root = input_path if input_path.is_dir() else input_path.parent
        out_root = out_root / f'character_dataset_{time.strftime("%Y%m%d_%H%M%S")}'
    work_dir = out_root / '_work'
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f'输出目录: {out_root}')

    # -- 收集输入 --
    mode = params.input_mode
    if mode == '自动检测':
        mode, files = detect_input_mode(input_path)
    elif mode == '视频':
        files = collect_videos(input_path)
    else:
        files = collect_images(input_path)
    if not files:
        raise ValueError(f'在 {input_path} 没有找到可处理的视频/图片')
    logger.log(f'输入模式: {mode}, 共 {len(files)} 个文件')

    deduper = Deduper(params.dedup_hash_dist)
    all_crops = []          # body / head (关联到body的)
    fallback_crops = []     # head_only / face (无全身可关联)
    stats = {'frames': 0, 'crops_raw': 0, 'crops_dedup': 0}

    seg_fail_count = 0

    def handle_image(img, source, frame_idx):
        crops = process_frame(img, source, frame_idx, params, logger)
        stats['crops_raw'] += len(crops)
        kept = []
        for c in crops:
            if stop_event.is_set():
                return
            if deduper.is_dup(c.image):
                continue
            kept.append(c)
        # 同一帧内 head 与 body 关联: head 与 body 的 dedup 是独立的,
        # 但若 body 被去掉而 head 保留, head 会失去 group => 转为 fallback
        body_keys = {c.group_key for c in kept if c.kind == 'body'}
        for c in kept:
            if c.kind == 'head' and c.group_key not in body_keys:
                c.kind = 'head_only'
        for c in kept:
            if c.kind in ('body', 'head'):
                all_crops.append(c)
            else:
                fallback_crops.append(c)
        stats['crops_dedup'] += len(kept)

    # -- 阶段1: 抽帧/读图 + 检测 + 裁剪 --
    for fi, f in enumerate(files):
        if stop_event.is_set():
            logger.log('收到停止请求, 中断')
            break
        logger.log(f'[{fi+1}/{len(files)}] 处理 {f.name}')
        if f.suffix.lower() in VIDEO_EXTS:
            for frame_idx, img in iter_video_frames(f, params.frame_interval, params.max_frames, logger):
                if stop_event.is_set():
                    break
                stats['frames'] += 1
                handle_image(img, str(f), frame_idx)
        else:
            try:
                img = Image.open(f).convert('RGB')
            except Exception as e:
                logger.log(f'  无法读取 {f.name}: {e}')
                continue
            stats['frames'] += 1
            handle_image(img, str(f), -1)

    total_crops = all_crops + fallback_crops
    logger.log(f"检测阶段完成: 帧/图 {stats['frames']}, 原始裁剪 {stats['crops_raw']}, 去重后 {stats['crops_dedup']}")
    if not total_crops:
        raise RuntimeError('没有提取到任何满足分辨率要求的角色裁剪, 请调低最小分辨率或检测置信度')

    # -- 阶段2: 可选分割 --
    if params.use_segmentation:
        logger.log(f'分割去背景 (isnetis) x{len(total_crops)} ...')
        for i, c in enumerate(total_crops):
            if stop_event.is_set():
                break
            rgba = segment_crop(c.image, min_input_side=params.seg_min_input_side)
            if rgba is not None:
                c.seg_image = rgba
            else:
                seg_fail_count += 1
            if i % 50 == 0:
                logger.set_progress(0.5 + 0.15 * (i / len(total_crops)), '分割')
        if seg_fail_count:
            logger.log(f'  {seg_fail_count} 张分割失败, 将使用原图')

    # -- 阶段3: CCIP 特征 --
    logger.log(f'CCIP 特征提取 ({params.ccip_model}) x{len(total_crops)} ...')
    feat_src = 'seg' if params.use_segmentation else 'raw'
    t0 = time.time()
    for i, c in enumerate(total_crops):
        if stop_event.is_set():
            break
        img_for_feat = None
        if feat_src == 'seg' and c.seg_image is not None:
            # RGBA -> RGB, 透明区域填白 (CCIP 训练数据为白底单角色)
            bg = Image.new('RGB', c.seg_image.size, (255, 255, 255))
            bg.paste(c.seg_image, mask=c.seg_image.split()[-1])
            img_for_feat = bg
        else:
            img_for_feat = c.image
        try:
            c.feature = ccip_extract_feature(img_for_feat, size=params.ccip_size, model=params.ccip_model)
        except Exception as e:
            logger.log(f'  特征提取失败, 跳过该裁剪: {e}')
            c.feature = None
        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(i, 1) * (len(total_crops) - i)
            logger.set_progress(0.65 + 0.1 * (i / len(total_crops)),
                                f'特征 {i}/{len(total_crops)} (ETA {eta/60:.1f}min)')
    total_crops = [c for c in total_crops if c.feature is not None]

    # -- 阶段4: 聚类 (body+head 一组; head_only/face 一组) --
    main_crops = [c for c in total_crops if c.kind in ('body', 'head')]
    fb_crops = [c for c in total_crops if c.kind in ('head_only', 'face')]
    logger.log(f'主聚类(全身/关联头部): {len(main_crops)}, 兜底聚类(仅头/脸): {len(fb_crops)}')

    # 主聚类只用 body 的特征, head 通过 group_key 继承
    bodies = [c for c in main_crops if c.kind == 'body']
    if bodies:
        cluster_crops(bodies, params, logger)
    head_by_group = {}
    for c in bodies:
        head_by_group.setdefault(c.group_key, c.cluster_id)
    for c in main_crops:
        if c.kind == 'head':
            c.cluster_id = head_by_group.get(c.group_key, -1)

    if fb_crops:
        logger.log('兜底聚类...')
        cluster_crops(fb_crops, params, logger)

    # -- 阶段5: 过滤 + 输出 --
    logger.log('整理输出...')
    logger.set_progress(0.92, '输出')

    def group_stats(crop_list):
        groups = {}
        for c in crop_list:
            if c.cluster_id < 0:
                groups.setdefault(-1, []).append(c)
            else:
                groups.setdefault(c.cluster_id, []).append(c)
        return groups

    report = {'main': {}, 'fallback': {}, 'params': asdict(params)}
    # 主角色: 以 body 数量决定簇大小
    body_groups = group_stats(bodies)
    all_main_groups = group_stats(main_crops)

    char_idx = 0
    rejected_dir = out_root / '_rejected'
    for cid, body_list in sorted(body_groups.items(), key=lambda kv: -len(kv[1])):
        if cid == -1:
            continue
        n_body = len(body_list)
        if n_body < params.min_cluster_count:
            if params.keep_rejected:
                _save_group(all_main_groups.get(cid, []), rejected_dir / f'main_noise_{cid}',
                            params, logger)
            report['main'][f'noise_{cid}'] = {'count': n_body, 'status': 'rejected(too few)'}
            continue
        char_idx += 1
        name = f'character_{char_idx:02d}'
        members = all_main_groups.get(cid, [])
        members = _cap_members(members, params.max_per_character)
        dest = out_root / name
        _save_group(members, dest, params, logger)
        report['main'][name] = {'cluster_id': int(cid), 'body_count': n_body,
                                'total_saved': len(members)}
        logger.log(f'  {name}: body={n_body}, 共保存 {len(members)} 张')

    # 噪声 body (cluster_id=-1)
    noise = body_groups.get(-1, [])
    if noise and params.keep_rejected:
        _save_group([c for c in main_crops if c.cluster_id == -1],
                    rejected_dir / 'main_noise_unclustered', params, logger)
    report['main']['_unclustered'] = len(noise)

    # 兜底组
    fb_groups = group_stats(fb_crops)
    fb_idx = 0
    for cid, members in sorted(fb_groups.items(), key=lambda kv: -len(kv[1])):
        if cid == -1:
            report['fallback']['_unclustered'] = len(members)
            continue
        if len(members) < params.min_cluster_count:
            if params.keep_rejected:
                _save_group(members, rejected_dir / f'fb_noise_{cid}', params, logger)
            report['fallback'][f'noise_{cid}'] = {'count': len(members), 'status': 'rejected(too few)'}
            continue
        fb_idx += 1
        name = f'faceonly_{fb_idx:02d}'
        members = _cap_members(members, params.max_per_character)
        _save_group(members, out_root / name, params, logger)
        report['fallback'][name] = {'cluster_id': int(cid), 'total_saved': len(members)}
        logger.log(f'  {name} (仅头/脸): 保存 {len(members)} 张')

    # -- report --
    report['stats'] = stats
    report['elapsed_sec'] = round(time.time() - t_start, 1)
    report['characters'] = char_idx
    report['faceonly_groups'] = fb_idx
    with open(out_root / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 清理临时目录
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    logger.set_progress(1.0, '完成')
    logger.log(f"✅ 完成! 共 {char_idx} 个角色 (+{fb_idx} 个仅头/脸组), "
               f"耗时 {report['elapsed_sec']/60:.1f} 分钟")
    logger.log(f'输出: {out_root}')
    return report


def _cap_members(members, cap):
    if cap <= 0 or len(members) <= cap:
        return members
    # 均匀采样保留时间分布
    step = len(members) / cap
    return [members[int(i * step)] for i in range(cap)]


def _save_group(members, dest: Path, params: Params, logger: Logger):
    dest.mkdir(parents=True, exist_ok=True)
    kind_dir = {'body': 'body', 'head': 'head', 'head_only': 'head_only', 'face': 'face'}
    counter = {}
    for c in members:
        sub = dest / kind_dir.get(c.kind, c.kind)
        sub.mkdir(parents=True, exist_ok=True)
        counter[c.kind] = counter.get(c.kind, 0) + 1
        stem = f'{Path(c.source).stem}_f{c.frame_idx:06d}_{c.kind}_{counter[c.kind]:04d}'
        if params.save_rgba and c.seg_image is not None:
            path = sub / f'{stem}.png'
            try:
                c.seg_image.save(path)
                c.saved_path = str(path)
                continue
            except Exception as e:
                logger.log(f'  RGBA保存失败 {stem}: {e}, 回退原图')
        path = sub / f'{stem}.png'
        try:
            c.image.save(path)
            c.saved_path = str(path)
        except Exception as e:
            logger.log(f'  保存失败 {stem}: {e}')


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
PARAM_HELP = """
### 参数说明 (每项都含推荐值与调大/调小效果)

**输入**
- **输入路径**: 视频文件、视频文件夹、图片文件夹均可, 自动识别(文件夹会递归搜索)。
- **输出目录**: 留空则在输入目录下自动创建 `character_dataset_时间戳/`。

**抽帧 (仅视频)**
- **抽帧间隔(秒)** `推荐 0.5~1.0` — 每隔多少秒取一帧。调小→样本更多更密、速度慢、重复多; 调大→速度快但可能漏掉出场少的角色。
- **单视频最大帧数** `推荐 3000` — 保护上限, 超长视频防止处理时间失控。

**人物检测 (detect_person, YOLOv8)**
- **人物检测模型版本** `推荐 v1.1` — v1.1 在 AniDet3+v1.1 混合数据集上训练, 动画截图和插画都较好; v0/v1 为旧数据集。
- **人物检测模型规格** `推荐 s` — n/s/m/x, 越大越准越慢(s 与 m 差距不大, x 明显慢)。
- **人物检测置信度** `推荐 0.35` — 调低→召回更多人物(含误检); 调高→更干净但漏检远景/遮挡角色。
- **人物检测 IoU** `推荐 0.5` — NMS 重叠框合并阈值。调低→去重框更狠(贴近的角色可能被合并掉); 调高→保留更多重叠框(同一人可能出双框)。

**头部检测 (detect_heads, YOLO)**
- **头部检测模型** `推荐 v2.0_s` — n 最快, s 均衡, m/l 更准更慢; v1.0 为旧版。
- **头部检测置信度** `推荐 0.40` — 调低→更多头部候选(误检↑); 调高→漏掉侧脸/遮挡头。
- **头部检测 IoU** `推荐 0.7` — 同人物 IoU, 作用于头部框 NMS。

**脸部检测 (detect_faces, 仅"只有脸可见"降级兜底时使用)**
- **脸部检测版本/规格** `推荐 v1.4 / s` — v1.4 最新; n 更快精度略低。
- **脸部检测置信度** `推荐 0.30` — 只在降级路径生效。
- **脸部检测 IoU** `推荐 0.7`。

**分辨率过滤与裁剪**
- **全身裁剪最小边长(px)** `推荐 160` — 裁剪框最短边小于该值则丢弃。调大→训练集质量高但数量少; 调小→数量多但混入远景小人。
- **头部裁剪最小边长(px)** `推荐 80` — 达不到即丢弃该头部裁剪(不影响全身)。
- **脸部降级裁剪最小边长(px)** `推荐 64` — 仅当画面检测不到全身、只有头/脸可见时使用。
- **裁剪外扩比例** `推荐 0.02` — 检测框向外扩的比例, 保留少量边缘(头发/饰品)。调大→上下文多但可能带入旁人; 0=严格贴框。

**去重**
- **dHash 去重距离** `推荐 6` — 感知哈希汉明距离≤该值视为重复帧丢弃。调大→去重更狠(可能误删相似姿势的不同图); 调小→保留更多(近似静止镜头会留很多重复); -1=关闭去重。

**分割 (isnetis 动漫抠图)**
- **isnetis 去背景** `推荐 开` — 抠掉背景再提特征/保存, 对黑白漫画、复杂背景显著提升聚类质量; 关闭则更快但背景干扰 CCIP 特征。
- **输出 RGBA 透明图** `推荐 开` — 保存为透明背景 PNG(适合 LoRA 训练); 关闭则保存原始裁剪。
- **分割前最小放大边长** `推荐 480` — 小裁剪先放大到该边长再送 isnetis(小图直接送会得到空 mask)。调大→小图分割更稳但更慢。

**CCIP 特征**
- **CCIP 模型** `推荐 b36-24` — b36 官方基准 F1 最高(0.941); pruned-24 稍快 F1=0.917; caformer-6 最轻但精度明显低。
- **CCIP 输入尺寸** `推荐 384` — 特征提取时的缩放尺寸(模型标准值)。调大无收益且更慢, 调小可能损失细节特征。

**聚类 (CCIP + sklearn)**
- **聚类方法** `推荐 optics_best` — optics_best/optics 对大样本更稳(eps 取官方调优值); dbscan 对 eps 极敏感, 只建议角色外观规整时用; dbscan_2/dbscan_free 是 imgutils 提供的更严格 eps 预设(簇更碎但更纯)。
- **eps (邻域半径)** `0=自动(按模型官方推荐)` — 调大→更多图并入同簇(角色合并风险↑); 调小→簇更碎(同一角色拆多簇)。
- **min_samples** `0=自动` — 核心点所需邻居数。样本少(<100张)时建议手动降到 2~3, 否则大量图被判为噪声(-1 簇)。

**输出过滤**
- **最小簇张数(路人过滤)** `推荐 8` — 簇内 body 图少于此数视为路人/噪声不输出。调大→只留主要角色; 调小→保留次要角色但混入误检。
- **每角色最大保留张数** `推荐 400, 0=不限` — 均匀采样截断, 防止主角刷屏。
- **保留被拒绝的簇** — 勾选后小簇存到 `_rejected/` 供人工检查。

**输出结构**
```
输出目录/
├─ character_01/
│  ├─ body/   (全身)
│  └─ head/   (同帧关联的头部)
├─ faceonly_01/ (只有头/脸可见时降级提取的组)
├─ _rejected/   (可选)
└─ report.json  (完整统计+参数)
```
"""


def build_gui():
    import gradio as gr

    logger = Logger()
    stop_event = threading.Event()
    worker = {'thread': None, 'report': None, 'error': None}

    def do_run(*args):
        if worker['thread'] and worker['thread'].is_alive():
            yield '⚠️ 已有任务在运行中, 请先停止或等待完成。', 0, '', ''
            return
        keys = ['input_path', 'input_mode', 'output_dir',
                'frame_interval', 'max_frames',
                'person_version', 'person_level', 'person_conf', 'person_iou',
                'head_model', 'head_conf', 'head_iou',
                'face_version', 'face_level', 'face_conf', 'face_iou',
                'min_body_res', 'min_head_res', 'min_face_res', 'crop_pad_ratio',
                'dedup_hash_dist',
                'use_segmentation', 'save_rgba', 'seg_min_input_side',
                'ccip_model_label', 'ccip_size',
                'cluster_method', 'cluster_eps', 'cluster_min_samples',
                'min_cluster_count', 'max_per_character', 'keep_rejected']
        raw = dict(zip(keys, args))
        num = {'frame_interval': float, 'max_frames': int, 'person_conf': float,
               'person_iou': float, 'head_conf': float, 'head_iou': float,
               'face_conf': float, 'face_iou': float, 'min_body_res': int,
               'min_head_res': int, 'min_face_res': int, 'crop_pad_ratio': float,
               'dedup_hash_dist': int, 'seg_min_input_side': int, 'ccip_size': int,
               'cluster_eps': float, 'cluster_min_samples': int,
               'min_cluster_count': int, 'max_per_character': int}
        for k, t in num.items():
            raw[k] = t(raw[k])
        for k in ('use_segmentation', 'save_rgba', 'keep_rejected'):
            raw[k] = bool(raw[k])
        raw['input_path'] = raw['input_path'] or ''
        raw['output_dir'] = raw['output_dir'] or ''
        params = Params(**raw)
        if not params.input_path.strip():
            yield '❌ 请填写输入路径 (视频文件/视频文件夹/图片文件夹)', 0, '', ''
            return

        logger.lines.clear()
        logger.progress = 0
        stop_event.clear()
        worker['report'] = None
        worker['error'] = None

        def _work():
            try:
                worker['report'] = run_pipeline(params, logger, stop_event)
            except Exception:
                worker['error'] = traceback.format_exc()
                logger.log('❌ 出错:\n' + worker['error'])
            finally:
                logger.set_progress(1.0, '结束')

        worker['thread'] = threading.Thread(target=_work, daemon=True)
        worker['thread'].start()

        last_text, last_prog = '', -1
        while True:
            text, prog, stage = logger.snapshot()
            if text != last_text or prog != last_prog:
                last_text, last_prog = text, prog
                yield text, prog, f'阶段: {stage}', ''
            if worker['thread'] and not worker['thread'].is_alive():
                final_msg = text
                if worker['error']:
                    final_msg = text + '\n\n❌ 任务失败, 详见日志。'
                else:
                    final_msg = text + '\n\n✅ 任务结束。可点击"打开输出目录"查看结果。'
                yield final_msg, 1.0, '阶段: 结束', ''
                return
            time.sleep(0.5)

    def do_stop():
        stop_event.set()
        logger.log('⏹ 停止请求已发送, 当前步骤完成后中断...')
        return '停止请求已发送'

    def open_output(output_dir):
        p = output_dir.strip() if output_dir and output_dir.strip() else ''
        if not p:
            # 找最新
            return '请先填写输出目录'
        p = Path(p)
        if p.exists():
            os.startfile(str(p))
            return f'已打开 {p}'
        return f'目录不存在: {p}'

    with gr.Blocks(title='动漫人物训练集提取器') as demo:
        gr.Markdown('# 🎨 全自动动漫人物训练集提取器\n'
                    '视频/图片 → 人物检测 → 全身+头部裁剪 → 去重 → CCIP 聚类 → 按角色分类输出\n'
                    '*底层: [imgutils](https://github.com/deepghs/imgutils) detect_person/detect_heads/detect_faces + CCIP + isnetis 分割*')
        with gr.Row():
            with gr.Column(scale=2):
                input_path = gr.Textbox(label='输入路径 (视频文件 / 视频文件夹 / 图片文件夹, 自动识别)',
                                        placeholder=r'例如 D:\anime\ep01.mp4 或 D:\anime\manga_pages')
                input_mode = gr.Radio(['自动检测', '视频', '图片'], value='自动检测', label='输入模式')
                output_dir = gr.Textbox(label='输出目录 (留空=输入目录下自动创建)', placeholder=r'D:\datasets\my_char_set')
                with gr.Accordion('⚙️ 参数设置 (点开查看说明与推荐值)', open=False):
                    gr.Markdown(PARAM_HELP)
                    gr.Markdown('**── 抽帧 (仅视频) ──**')
                    with gr.Row():
                        frame_interval = gr.Slider(0.1, 5.0, value=1.0, step=0.1,
                                                   label='抽帧间隔(秒) [推荐0.5~1.0]')
                        max_frames = gr.Slider(100, 20000, value=3000, step=100,
                                               label='单视频最大帧数 [推荐3000]')
                    gr.Markdown('**── 人物检测 detect_person ──**')
                    with gr.Row():
                        person_version = gr.Dropdown(['v1.1', 'v1', 'v0'], value='v1.1',
                                                     label='人物模型版本 [推荐v1.1]')
                        person_level = gr.Dropdown(['n', 's', 'm', 'x'], value='s',
                                                   label='人物模型规格 [推荐s]')
                    with gr.Row():
                        person_conf = gr.Slider(0.05, 0.9, value=0.35, step=0.05,
                                                label='人物置信度 [推荐0.35]')
                        person_iou = gr.Slider(0.1, 0.95, value=0.5, step=0.05,
                                               label='人物IoU(NMS) [推荐0.5]')
                    gr.Markdown('**── 头部检测 detect_heads ──**')
                    with gr.Row():
                        head_model = gr.Dropdown(list(HEAD_MODELS.keys()),
                                                 value=list(HEAD_MODELS.keys())[0],
                                                 label='头部检测模型 [推荐v2.0_s]')
                        head_conf = gr.Slider(0.05, 0.9, value=0.40, step=0.05,
                                              label='头部置信度 [推荐0.40]')
                        head_iou = gr.Slider(0.1, 0.95, value=0.70, step=0.05,
                                             label='头部IoU(NMS) [推荐0.7]')
                    gr.Markdown('**── 脸部检测 detect_faces (仅"只有脸可见"降级时用) ──**')
                    with gr.Row():
                        face_version = gr.Dropdown(['v1.4', 'v1.3', 'v1', 'v0'], value='v1.4',
                                                   label='脸部模型版本 [推荐v1.4]')
                        face_level = gr.Dropdown(['s', 'n'], value='s',
                                                 label='脸部模型规格 [推荐s]')
                    with gr.Row():
                        face_conf = gr.Slider(0.05, 0.9, value=0.30, step=0.05,
                                              label='脸部置信度 [推荐0.30]')
                        face_iou = gr.Slider(0.1, 0.95, value=0.70, step=0.05,
                                             label='脸部IoU(NMS) [推荐0.7]')
                    gr.Markdown('**── 分辨率过滤与裁剪 ──**')
                    with gr.Row():
                        min_body_res = gr.Slider(64, 512, value=160, step=8,
                                                 label='全身最小边长px [推荐160]')
                        min_head_res = gr.Slider(32, 256, value=80, step=4,
                                                 label='头部最小边长px [推荐80]')
                    with gr.Row():
                        min_face_res = gr.Slider(32, 256, value=64, step=4,
                                                 label='脸部降级最小边长px [推荐64]')
                        crop_pad_ratio = gr.Slider(0.0, 0.2, value=0.02, step=0.01,
                                                   label='裁剪外扩比例 [推荐0.02]')
                    gr.Markdown('**── 去重 / 分割 isnetis ──**')
                    with gr.Row():
                        dedup_hash_dist = gr.Slider(-1, 16, value=6, step=1,
                                                    label='dHash去重距离(-1=关闭) [推荐6]')
                        seg_min_input_side = gr.Slider(256, 1024, value=480, step=32,
                                                       label='分割前最小放大边长 [推荐480]')
                    with gr.Row():
                        use_segmentation = gr.Checkbox(value=True, label='isnetis去背景 [推荐开]')
                        save_rgba = gr.Checkbox(value=True, label='输出RGBA透明PNG [推荐开]')
                    gr.Markdown('**── CCIP 特征与聚类 ──**')
                    with gr.Row():
                        ccip_model_label = gr.Dropdown(list(CCIP_MODELS.keys()),
                                                       value=list(CCIP_MODELS.keys())[0],
                                                       label='CCIP 模型 [推荐b36]')
                        ccip_size = gr.Dropdown([224, 384, 448, 512], value=384,
                                                label='CCIP 输入尺寸 [推荐384]')
                    with gr.Row():
                        cluster_method = gr.Dropdown(['optics_best', 'optics', 'dbscan',
                                                      'dbscan_2', 'dbscan_free'],
                                                     value='optics_best',
                                                     label='聚类方法 [推荐optics_best]')
                    with gr.Row():
                        cluster_eps = gr.Slider(0.0, 0.5, value=0.0, step=0.005,
                                                label='eps邻域半径 (0=模型推荐) [推荐0]')
                        cluster_min_samples = gr.Slider(0, 20, value=0, step=1,
                                                        label='min_samples (0=模型推荐, 样本<100时建议2~3)')
                    gr.Markdown('**── 输出过滤 ──**')
                    with gr.Row():
                        min_cluster_count = gr.Slider(1, 100, value=8, step=1,
                                                      label='最小簇张数-路人过滤 [推荐8]')
                        max_per_character = gr.Slider(0, 5000, value=400, step=50,
                                                      label='每角色最大张数(0=不限) [推荐400]')
                        keep_rejected = gr.Checkbox(value=False, label='保留被拒绝的小簇到 _rejected/')
                with gr.Row():
                    run_btn = gr.Button('🚀 开始提取', variant='primary', scale=3)
                    stop_btn = gr.Button('⏹ 停止', scale=1)
                    open_btn = gr.Button('📂 打开输出目录', scale=1)
            with gr.Column(scale=2):
                stage_text = gr.Textbox(label='状态', value='阶段: 空闲', interactive=False)
                prog = gr.Slider(0, 1, value=0, label='进度', interactive=False)
                log_box = gr.Textbox(label='运行日志', lines=24, max_lines=40, interactive=False)
                out_msg = gr.Textbox(label='', interactive=False, visible=False)

        # 输入顺序必须与 do_run 中 keys 一致
        run_btn.click(
            fn=do_run,
            inputs=[input_path, input_mode, output_dir,
                    frame_interval, max_frames,
                    person_version, person_level, person_conf, person_iou,
                    head_model, head_conf, head_iou,
                    face_version, face_level, face_conf, face_iou,
                    min_body_res, min_head_res, min_face_res, crop_pad_ratio,
                    dedup_hash_dist,
                    use_segmentation, save_rgba, seg_min_input_side,
                    ccip_model_label, ccip_size,
                    cluster_method, cluster_eps, cluster_min_samples,
                    min_cluster_count, max_per_character, keep_rejected],
            outputs=[log_box, prog, stage_text, out_msg],
        )
        stop_btn.click(fn=do_stop, outputs=[stage_text])
        open_btn.click(fn=open_output, inputs=[output_dir], outputs=[stage_text])

    return demo


def main():
    # 首次导入 imgutils 时它可能尝试 pip install onnxruntime —— 我们已在依赖里声明, 不会触发
    import inspect
    import gradio as gr
    demo = build_gui()
    port = int(os.environ.get('ACE_PORT', '7861'))
    # 兼容 Gradio 4/5 (show_api) 与 Gradio 6+ (footer_links)
    launch_kwargs = dict(server_name='127.0.0.1', server_port=port,
                         inbrowser=True, quiet=False)
    sig = inspect.signature(gr.Blocks.launch)
    if 'footer_links' in sig.parameters:          # Gradio >= 6
        launch_kwargs['footer_links'] = ['gradio', 'settings']
    elif 'show_api' in sig.parameters:            # Gradio 4/5
        launch_kwargs['show_api'] = False
    demo.launch(**launch_kwargs)


if __name__ == '__main__':
    main()
