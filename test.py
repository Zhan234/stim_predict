import numpy as np, os, stim, pymatching
from utils.data_manager import DataManager
dm = DataManager(base_dir="./data")
exp="rep_code_21"
arr = np.load(os.path.join(dm.base_dir, exp, "samples.npz"))
det = arr['detectors'][:5000]
obs = arr['observables'][:5000].ravel().astype(int)

# load dem
dem = stim.DetectorErrorModel(open(os.path.join(dm.base_dir, exp, "ground_truth.dem")).read())
m = pymatching.Matching.from_detector_error_model(dem)
pred = m.decode_batch(det).ravel().astype(int)
print("orig equal_frac", (pred==obs).mean())

# flip 100 random obs bits
obs_flipped = obs.copy()
ix = np.random.choice(len(obs), 100, replace=False)
obs_flipped[ix] ^= 1
print("mean_obs_flipped", obs_flipped.mean())
# compare pred to flipped obs (we don't re-decode; check if pred equals flipped obs)
print("equal_frac_with_flipped", (pred==obs_flipped).mean())

# re-decode with a very different DEM (large random probs) and compare
import correlation
circuit = stim.Circuit(open(os.path.join(dm.base_dir, exp, "circuit.stim")).read())
dem0 = circuit.detector_error_model(decompose_errors=True)
tg = correlation.TannerGraph(dem0)
keys = list(tg.hyperedge_probs.keys())
from evaluators.decoder_ler import DecoderLEREvaluator
evalr = DecoderLEREvaluator()
rand_probs = {k: float(np.random.rand()) for k in keys}  # 更大范围[0,1)
rand_dem = evalr._build_dem_from_probs(circuit, rand_probs)
try:
    m2 = pymatching.Matching.from_detector_error_model(rand_dem)
    pred2 = m2.decode_batch(det).ravel().astype(int)
    print("rand_dem equal_frac_with_obs", (pred2==obs).mean())
except Exception as e:
    print("rand_dem decode failed:", e)