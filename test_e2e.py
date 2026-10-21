"""E2E test v2: multi-episode merge + clustering correctness check.
- 模拟 2 集 (两个文件夹), 用多路径输入(分号分隔), 验证跨集合并
- 验证向量化 cosine 距离聚类与官方 ccip_clustering (metric ONNX) 结果一致
"""
import os, sys, shutil, urllib.request, threading, time, json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import extractor

BASE = Path('e2e_test')
EP1 = BASE / 'ep01'
EP2 = BASE / 'ep02'
OUT = BASE / 'output'
shutil.rmtree(BASE, ignore_errors=True)
EP1.mkdir(parents=True)
EP2.mkdir(parents=True)

url = 'https://raw.githubusercontent.com/deepghs/imgutils/main/docs/source/api_doc/detect/genshin_post.jpg'
p = EP1 / 'genshin_post.jpg'
if not p.exists():
    urllib.request.urlretrieve(url, p)
    print('downloaded', os.path.getsize(p))

img = Image.open(p)
w, h = img.size

for i in range(6):
    jx, jy = (i * 7) % 13, (i * 11) % 17
    img.crop((jx, jy, w - 13 + jx, h - 17 + jy)).resize((w, h)).save(
        EP1 / f'ep01_frame_{i:03d}.jpg', quality=92)
for i in range(5):
    jx, jy = (i * 5) % 11, (i * 13) % 15
    img.crop((jx, jy, w - 11 + jx, h - 15 + jy)).resize((w, h)).save(
        EP2 / f'ep02_frame_{i:03d}.jpg', quality=92)

params = extractor.Params(
    input_path=f'{EP1};{EP2}',
    input_mode='图片',
    person_level='s',
    person_conf=0.30,
    min_body_res=120,
    min_head_res=60,
    min_face_res=48,
    dedup_hash_dist=-1,
    use_segmentation=True,
    save_rgba=True,
    ccip_model_label=list(extractor.CCIP_MODELS.keys())[0],
    cluster_method='dbscan',
    cluster_eps=0.0,
    cluster_min_samples=2,
    min_cluster_count=3,
    max_per_character=100,
    keep_rejected=True,
    output_dir=str(OUT),
)

logger = extractor.Logger()
stop = threading.Event()
t0 = time.time()
report = extractor.run_pipeline(params, logger, stop)
print('elapsed', round(time.time() - t0, 1), 's')

print('=== per_source ===')
for k, v in report['per_source'].items():
    print(' ', Path(k).name, v)
print('=== main ===')
print(json.dumps(report['main'], ensure_ascii=False, indent=1))

merged_ok = 0
for d in sorted(OUT.glob('character_*')):
    files = list((d / 'body').glob('*.png')) if (d / 'body').exists() else []
    srcs = {f.stem.split('_f')[0] for f in files}
    has_both = any(s.startswith('ep01') for s in srcs) and any(s.startswith('ep02') for s in srcs)
    print(f'{d.name}: body={len(files)} sources={sorted(srcs)} cross_episode_merged={has_both}')
    if has_both:
        merged_ok += 1

print()
print('MERGE_CHECK:', 'PASS' if merged_ok >= 2 else 'FAIL', f'({merged_ok} characters merged across episodes)')
assert merged_ok >= 2, '跨集合并失败!'

print()
print('=== clustering correctness (12 official ccip demo images) ===')
demo_dir = BASE / 'ccip_demo'
demo_dir.mkdir()
for i in range(1, 13):
    u = f'https://raw.githubusercontent.com/deepghs/imgutils/main/docs/source/api_doc/metrics/ccip/{i}.jpg'
    fp = demo_dir / f'{i}.jpg'
    if not fp.exists():
        urllib.request.urlretrieve(u, fp)
imgs = [str(demo_dir / f'{i}.jpg') for i in range(1, 13)]

import numpy as np
from imgutils.metrics import ccip_clustering, ccip_extract_feature

p2 = extractor.Params(cluster_method='dbscan', cluster_min_samples=2)
feats = np.stack([ccip_extract_feature(pp, model=p2.ccip_model) for pp in imgs])

official = ccip_clustering(imgs, method='dbscan', min_samples=2, model=p2.ccip_model)
mine = extractor.cluster_features(feats, p2, logger)
print('official:', official)
print('mine    :', list(mine))

def pairs(labels):
    return {(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))
            if labels[i] == labels[j] and labels[i] != -1}

po, pm = pairs(official), pairs(mine)
inter = len(po & pm)
prec = inter / max(len(pm), 1)
rec = inter / max(len(po), 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(f'pairwise precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}')
print('CLUSTER_CHECK:', 'PASS' if f1 >= 0.95 else 'FAIL')
assert f1 >= 0.95
