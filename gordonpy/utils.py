import re, numpy as np
from mediapipe.framework.formats import landmark_pb2
from pathlib import Path
import urllib.request

def norm_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def phrase_to_regex(phrase: str):
    p = norm_text(phrase)
    p_escaped = re.escape(p).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{p_escaped}\b")

def rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(x * x)))

def ensure_model(local_path: str, url: str):
    p = Path(local_path); p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists(): urllib.request.urlretrieve(url, p)

def _xy(img_w, img_h, lm):
    x = lm.x if hasattr(lm, "x") else lm[0]
    y = lm.y if hasattr(lm, "y") else lm[1]
    return int(x * img_w), int(y * img_h)

def _to_landmark_list(nl):
    ll = landmark_pb2.NormalizedLandmarkList()
    for l in (nl or []):
        if hasattr(l, "x"): x, y, z = float(l.x), float(l.y), float(l.z)
        else:
            x, y = float(l[0]), float(l[1]); z = float(l[2]) if len(l) > 2 else 0.0
        ll.landmark.add(x=x, y=y, z=z)
    return ll
