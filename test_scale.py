import os, time
import extractor
extractor._patch_onnx_cuda_options()
import numpy as np
import urllib.request
from PIL import Image
from imgutils.metrics import ccip_extract_feature, ccip_batch_differences

MODEL = 'ccip-caformer_b36-24'
os.makedirs('e2e_test/input', exist_ok=True)

# --- part A: metric distance vs feature-space distance correlation (real images) ---
urls = [f'https://raw.githubusercontent.com/deepghs/imgutils/main/docs/source/api_doc/metrics/ccip/{i}.jpg' for i in range(1, 13)]
paths = []
for i, u in enumerate(urls):
    p = f'e2e_test/input/ccip_{i}.jpg'
    if not os.path.exists(p):
        urllib.request.urlretrieve(u, p)
    paths.append(p)
feats = np.stack([ccip_extract_feature(p, model=MODEL) for p in paths])
metric = ccip_batch_differences(list(feats), model=MODEL)
# normalize features
fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
cos = 1.0 - fn @ fn.T
l2 = np.linalg.norm(feats[:, None, :] - feats[None, :, :], axis=-1)
iu = np.triu_indices(len(feats), 1)
m, c, l = metric[iu], cos[iu], l2[iu]
print('corr(metric, cosine):', round(float(np.corrcoef(m, c)[0, 1]), 4))
print('corr(metric, L2):', round(float(np.corrcoef(m, l)[0, 1]), 4))
print('metric range:', round(float(m.min()), 3), round(float(m.max()), 3))
print('cos range:', round(float(c.min()), 3), round(float(c.max()), 3))
# same-char pairs (1,2,3 / 4,5,6 groups per docs example: labels [0,0,0,3,3,3,1,1,1,1,2,2])
labels = [0,0,0,3,3,3,1,1,1,1,2,2]
same_m = [metric[i,j] for i in range(12) for j in range(i+1,12) if labels[i]==labels[j]]
diff_m = [metric[i,j] for i in range(12) for j in range(i+1,12) if labels[i]!=labels[j]]
same_c = [cos[i,j] for i in range(12) for j in range(i+1,12) if labels[i]==labels[j]]
diff_c = [cos[i,j] for i in range(12) for j in range(i+1,12) if labels[i]!=labels[j]]
print(f'metric: same avg {np.mean(same_m):.3f} (max {np.max(same_m):.3f}), diff avg {np.mean(diff_m):.3f} (min {np.min(diff_m):.3f})')
print(f'cosine: same avg {np.mean(same_c):.3f} (max {np.max(same_c):.3f}), diff avg {np.mean(diff_m):.3f}...')
print(f'cosine: same avg {np.mean(same_c):.3f} (max {np.max(same_c):.3f}), diff avg {np.mean(diff_c):.3f} (min {np.min(diff_c):.3f})')

# --- part B: metric model scale with RANDOM features ---
rng = np.random.default_rng(0)
for n in [1024, 2048, 4096, 8192]:
    rf = [rng.standard_normal(768).astype(np.float32) for _ in range(n)]
    t0 = time.time()
    try:
        d = ccip_batch_differences(rf, model=MODEL)
        dt = time.time() - t0
        print(f'RANDOM n={n}: OK {d.shape} {dt:.1f}s, out={n*n*4/1e6:.0f}MB')
    except Exception as e:
        print(f'RANDOM n={n}: FAIL {type(e).__name__} {str(e)[:100]}')
        break
