import os, time
import extractor  # noqa: 导入即执行 GPU DLL 目录注册 + 缓存环境变量

import onnxruntime as ort
print('ort version:', ort.__version__)

from PIL import Image
import urllib.request
os.makedirs('e2e_test/input', exist_ok=True)
if not os.path.exists('e2e_test/input/genshin_post.jpg'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/deepghs/imgutils/main/docs/source/api_doc/detect/genshin_post.jpg',
        'e2e_test/input/genshin_post.jpg')
img = Image.open('e2e_test/input/genshin_post.jpg')

from imgutils.metrics import ccip_extract_feature
t0 = time.time()
f = ccip_extract_feature(img, model='ccip-caformer_b36-24')
print('warmup:', round(time.time()-t0, 2), 's')
t0 = time.time()
N = 30
for i in range(N):
    f = ccip_extract_feature(img, model='ccip-caformer_b36-24')
dt = (time.time()-t0)/N
print(f'RESULT CCIP b36 per-image: {dt*1000:.0f} ms  (CPU baseline ~640ms)')

from imgutils.detect import detect_person
t0 = time.time()
r = detect_person(img, level='s', conf_threshold=0.35)
print('detect warmup:', round(time.time()-t0, 2), 's, dets:', len(r))
t0 = time.time()
for i in range(20):
    r = detect_person(img, level='s', conf_threshold=0.35)
print(f'RESULT detect_person per-image: {(time.time()-t0)/20*1000:.0f} ms')

from imgutils.segment import segment_rgba_with_isnetis
big = img.crop((32, 140, 392, 717))
t0 = time.time()
_, rgba = segment_rgba_with_isnetis(big, scale=1024)
print('seg warmup:', round(time.time()-t0, 2), 's')
t0 = time.time()
for i in range(10):
    _, rgba = segment_rgba_with_isnetis(big, scale=1024)
print(f'RESULT isnetis per-image: {(time.time()-t0)/10*1000:.0f} ms')
