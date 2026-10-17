"""E2E test: run full pipeline on a synthetic mini-dataset."""
import os, sys, shutil, urllib.request, threading, time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import extractor

BASE = Path('e2e_test')
IN = BASE / 'input'
OUT = BASE / 'output'
shutil.rmtree(BASE, ignore_errors=True)
IN.mkdir(parents=True)

# download a multi-character anime image
urls = {
    'genshin_post.jpg': 'https://raw.githubusercontent.com/deepghs/imgutils/main/docs/source/api_doc/detect/genshin_post.jpg',
}
for name, u in urls.items():
    p = IN / name
    if not p.exists():
        urllib.request.urlretrieve(u, p)
        print('downloaded', name, os.path.getsize(p))

# create N copies with slight variations (dedup off in params) to reach cluster counts
img = Image.open(IN / 'genshin_post.jpg')
w, h = img.size
print('image size:', w, h)
for i in range(10):
    # small random crops so images are not byte-identical
    jx, jy = (i * 7) % 13, (i * 11) % 17
    sub = img.crop((jx, jy, w - 13 + jx, h - 17 + jy)).resize((w, h))
    sub.save(IN / f'copy_{i:02d}.jpg', quality=92)

params = extractor.Params(
    input_path=str(IN),
    input_mode='图片',
    person_level='s',
    person_conf=0.30,
    min_body_res=120,
    min_head_res=60,
    min_face_res=48,
    dedup_hash_dist=-1,          # off: copies are near-identical
    use_segmentation=True,
    save_rgba=True,
    ccip_model_label=list(extractor.CCIP_MODELS.keys())[0],
    cluster_method='dbscan',
    cluster_eps=0.0,             # auto -> model threshold
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
print('=== REPORT ===')
import json
print(json.dumps({k: v for k, v in report.items() if k != 'params'}, ensure_ascii=False, indent=1, default=str))

print('=== OUTPUT TREE ===')
for p in sorted(OUT.rglob('*')):
    rel = p.relative_to(OUT)
    if p.is_dir():
        print('[D]', rel)
    else:
        print('   ', rel, os.path.getsize(p))
