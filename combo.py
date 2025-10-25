import os, re, time, queue, threading, json
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import numpy as np
import torch
import sounddevice as sd
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import cv2
import urllib.request
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
)
from mediapipe.tasks.python.vision import RunningMode as RunningMode
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =====================
# CONFIG
# =====================
SR = 16000
TARGET_FPS = 24.0
TARGET_DT = 1.0 / TARGET_FPS

# Wake phrase / VAD tuning
WAKE_VAD = {"min_silence_duration_ms": 250, "speech_pad_ms": 80}
WAKE_ENERGY_THRESH = 0.008  # quick pre-gate so we don't ASR silence
WAKE_CHECK_INTERVAL = 0.75  # seconds between wake checks
WAKE_REFRAC_S = 2.0         # refractory after wake

# Intent phrases
INTENTS = {
    "mirror": ["mirror mode", "start mirroring", "mirror"],
    "record": ["record mode", "record", "begin recording"],
    "idle": ["idle mode", "go idle", "stop", "sleep", "standby", "cancel"],
}

# IO / persistence
REC_DIR = "recordings"
SESSION_TXT = os.path.join(REC_DIR, "session.txt")

# Action matching
ACTION_SIM_THRESHOLD = 0.70

# =====================
# SMALL UTILS
# =====================

def norm_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

INTENTS_NORM_MAP = {norm_text(p): k for k, vs in INTENTS.items() for p in vs}
FINISH_NORM_SET = {norm_text(p) for p in ["finish recording", "stop recording"]}


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def save_wav_file(path, audio, sr):
    try:
        import soundfile as sf
        sf.write(path, audio, sr)
    except Exception:
        import wave
        y = np.clip(audio, -1.0, 1.0)
        y = (y * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(y.tobytes())


# =====================
# RING BUFFERS (to avoid constant np.concatenate)
# =====================
class AudioRing:
    def __init__(self, max_seconds: float, sr: int):
        self.sr = sr
        self.max_samples = int(max_seconds * sr)
        self.chunks = deque()
        self.samples = 0

    def push(self, block: np.ndarray):
        b = block.astype(np.float32, copy=False)
        self.chunks.append(b.copy())
        self.samples += len(b)
        while self.samples > self.max_samples and self.chunks:
            old = self.chunks.popleft()
            self.samples -= len(old)

    def as_array(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self.chunks))

    def clear(self):
        self.chunks.clear()
        self.samples = 0

    def duration(self) -> float:
        return self.samples / float(self.sr)


class UtteranceCapture:
    """Simple energy + timeout segmenter.
    Keep debug prints in main loop to show state transitions.
    """
    def __init__(self, sr: int):
        self.sr = sr
        self.blocks = []
        self.samples = 0
        self.last_voice_t = None

    def reset(self):
        self.blocks.clear()
        self.samples = 0
        self.last_voice_t = None

    def feed(self, block: np.ndarray, voice_thresh=0.003):
        self.blocks.append(block.copy())
        self.samples += len(block)
        if rms(block) > voice_thresh:
            self.last_voice_t = time.monotonic()

    def ready(self, min_s: float, eos_ms: int, max_s: float) -> bool:
        dur = self.samples / float(self.sr)
        silent_for = (time.monotonic() - self.last_voice_t) if self.last_voice_t else 0.0
        eos_s = eos_ms / 1000.0
        return (dur >= min_s) and ((silent_for >= eos_s) or (dur >= max_s))

    def pop(self) -> np.ndarray:
        arr = np.concatenate(self.blocks) if self.blocks else np.zeros(0, dtype=np.float32)
        self.reset()
        # small trailing trim avoids padding from VAD
        if len(arr) > int(0.05 * self.sr):
            arr = arr[:-int(0.025 * self.sr)]
        return arr


# =====================
# MP drawing helpers
# =====================
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
HAND_CONN = mp.solutions.hands.HAND_CONNECTIONS


def _xy(img_w, img_h, lm):
    x = lm.x if hasattr(lm, "x") else lm[0]
    y = lm.y if hasattr(lm, "y") else lm[1]
    return int(x * img_w), int(y * img_h)


def _to_landmark_list(nl):
    ll = landmark_pb2.NormalizedLandmarkList()
    for l in (nl or []):
        if hasattr(l, "x"):
            x, y, z = float(l.x), float(l.y), float(l.z)
        else:
            x, y = float(l[0]), float(l[1])
            z = float(l[2]) if len(l) > 2 else 0.0
        ll.landmark.add(x=x, y=y, z=z)
    return ll


def draw_overlays(rgb_image, pose_landmarks_list, hand_landmarks_list, hand_handedness_list=None):
    img = rgb_image.copy()
    h, w = img.shape[:2]

    # Prefer left hand if both are present for consistency
    hands_to_draw = []
    if hand_landmarks_list:
        if hand_handedness_list and len(hand_handedness_list) == len(hand_landmarks_list):
            for i, hnd in enumerate(hand_handedness_list):
                s = str(hnd).lower()
                if "left" in s:
                    hands_to_draw = [hand_landmarks_list[i]]
                    break
            if not hands_to_draw:
                hands_to_draw = [hand_landmarks_list[0]]
        else:
            hands_to_draw = hand_landmarks_list

    for hand_pts in hands_to_draw:
        hand_proto = _to_landmark_list(hand_pts)
        mp_draw.draw_landmarks(
            img, hand_proto, HAND_CONN,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

    if pose_landmarks_list:
        pose = pose_landmarks_list[0]
        arm_idxs = [11, 13, 15]
        for idx in arm_idxs:
            if 0 <= idx < len(pose):
                cx, cy = _xy(w, h, pose[idx])
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        p0, p1, p2 = arm_idxs
        if 0 <= p0 < len(pose) and 0 <= p1 < len(pose):
            x0, y0 = _xy(w, h, pose[p0]); x1, y1 = _xy(w, h, pose[p1])
            cv2.line(img, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        if 0 <= p1 < len(pose) and 0 <= p2 < len(pose):
            x1, y1 = _xy(w, h, pose[p1]); x2, y2 = _xy(w, h, pose[p2])
            cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    return img


# =====================
# Embeddings & text helpers
# =====================

def embed_bank(embedder, phrases):
    return embedder.encode([norm_text(p) for p in phrases], normalize_embeddings=True)


def best_sim_against(embedder, refs_emb, text: str, ngram_max=1) -> float:
    t = norm_text(text)
    if not t:
        return 0.0
    toks = t.split()
    cands = [t]
    for n in range(2, min(ngram_max, len(toks)) + 1):
        cands.append(" ".join(toks[-n:]))
    embs = embedder.encode(cands, normalize_embeddings=True)
    sims = np.dot(embs, refs_emb.T)
    return float(np.max(sims))


# =====================
# Natural-language schedule parsing (keep debug prints)
# =====================
NUM_WORDS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
    "nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
    "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20
}


def _num_from_word_or_digit(s):
    w = s.strip().lower()
    if re.match(r"^\d+(\.\d+)?$", w):
        return float(w)
    return float(NUM_WORDS.get(w, np.nan))


def _unit_to_seconds(u):
    u = u.lower()
    if u.startswith("s"): return 1.0
    if u.startswith("m"): return 60.0
    if u.startswith("h"): return 3600.0
    return np.nan


def decode_action_schedule_regex(text):
    t = norm_text(text)
    period_s = duration_s = None
    count = None

    m = re.search(r"\bevery\s+([a-z0-9\.]+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b", t)
    if m:
        n = _num_from_word_or_digit(m.group(1))
        scale = _unit_to_seconds(m.group(2))
        if not np.isnan(n) and not np.isnan(scale):
            period_s = float(n) * float(scale)

    m = re.search(r"\bfor\s+([a-z0-9\.]+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b", t)
    if m:
        n = _num_from_word_or_digit(m.group(1))
        scale = _unit_to_seconds(m.group(2))
        if not np.isnan(n) and not np.isnan(scale):
            duration_s = float(n) * float(scale)

    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s*(times?|x)\b", t)
    if m:
        n = _num_from_word_or_digit(m.group(1))
        if not np.isnan(n):
            count = int(round(n))

    m = re.search(r"\brepeat\s+(\d+)\b", t)
    if m and count is None:
        count = int(m.group(1))

    if count and period_s:
        period_s = None
    return {"period_s": period_s, "duration_s": duration_s, "count": count}


# Optional LLM assistance; keep debug prints; safe fallback to regex

def build_qwen():
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
        dev = 0 if torch.cuda.is_available() else -1
        mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=dev)
        print("[QWEN] loaded Qwen2.5-1.5B-Instruct", flush=True)
        return pipe
    except Exception as e:
        print(f"[QWEN] unavailable, falling back to regex: {e}", flush=True)
        return None


def decode_action_schedule_qwen(pipe, text):
    try:
        prompt = (
            "Extract scheduling parameters from the utterance. "
            "Return only valid JSON with keys: period_s (number or null), duration_s (number or null), count (integer or null). "
            "If both period and count are present, set count and set period_s to null. "
            "Units must be seconds. Examples: "
            "\"stir every 5 minutes for 2 hours\" -> {\"period_s\":300,\"duration_s\":7200,\"count\":null}. "
            "\"stir every 5 minutes\" -> {\"period_s\":300,\"duration_s\":null,\"count\":null}. "
            "\"stir for 2 hours\" -> {\"period_s\":null,\"duration_s\":7200,\"count\":null}. "
            "\"stir 5 times\" -> {\"period_s\":null,\"duration_s\":null,\"count\":5}. "
            "Utterance: "
        ) + json.dumps(text)
        out = pipe(prompt, max_new_tokens=128, do_sample=False)
        cand = out[0].get("generated_text", "")
        js = re.search(r"\{.*\}", cand, re.DOTALL)
        if not js:
            return None
        data = json.loads(js.group(0))
        period_s = data.get("period_s", None)
        duration_s = data.get("duration_s", None)
        count = data.get("count", None)
        if isinstance(period_s, str):
            period_s = float(period_s) if period_s else None
        if isinstance(duration_s, str):
            duration_s = float(duration_s) if duration_s else None
        if isinstance(count, str):
            count = int(count) if count else None
        if count:
            period_s = None
        return {"period_s": period_s, "duration_s": duration_s, "count": count}
    except Exception as e:
        print(f"[QWEN] decode failed, fallback to regex: {e}", flush=True)
        return None


def decode_action_schedule(pipe, text):
    print(f"[SCHEDULE_PARSE] input='{text}'", flush=True)
    data = decode_action_schedule_qwen(pipe, text) if pipe is not None else None
    if data is None:
        data = decode_action_schedule_regex(text)

    period_s = data.get("period_s")
    duration_s = data.get("duration_s")
    count = data.get("count")

    # sanitize
    if count is not None and count <= 0:
        count = None
    if period_s is not None and period_s <= 0:
        period_s = None
    if duration_s is not None and duration_s <= 0:
        duration_s = None

    if count is not None:
        period_s = None
        duration_s = None
        kind = "count"
    elif period_s is not None and duration_s is not None:
        kind = "periodic"
    elif period_s is not None and duration_s is None:
        kind = "periodic"
    elif period_s is None and duration_s is not None:
        kind = "duration"
    else:
        kind = "count"
        count = 1

    print(f"[SCHEDULE_PARSE] kind={kind} period_s={period_s} duration_s={duration_s} count={count}", flush=True)
    return {"kind": kind, "period_s": period_s, "duration_s": duration_s, "count": count}


# =====================
# Model management
# =====================

def ensure_model(local_path: Path, url: str):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        urllib.request.urlretrieve(url, local_path)


# =====================
# MAIN
# =====================
q = queue.Queue()


def audio_cb(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO][status] {status}", flush=True)
    q.put(indata.copy().reshape(-1))


def transcribe(model, audio, vad, chunk_len):
    segs, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters=vad,
        condition_on_previous_text=False,
        chunk_length=chunk_len,
    )
    return " ".join(s.text.strip() for s in segs).strip()


def main():
    os.makedirs(REC_DIR, exist_ok=True)

    # ---------- Audio ----------
    blocksize = int(SR * 0.04)  # 40ms
    sd.default.channels = 1
    sd.default.samplerate = SR

    stream = sd.InputStream(dtype="float32", blocksize=blocksize, callback=audio_cb)

    # ---------- ASR (GPU/CPU safe) ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        wake_compute = "float16"
        main_model_name = "large-v3"
        main_compute = "float16"
    else:
        wake_compute = "int8"
        main_model_name = "small.en"  # CPU fallback (large-v3 on CPU is painful)
        main_compute = "int8"

    wake_asr = WhisperModel("tiny.en", device=device, compute_type=wake_compute)
    main_asr = WhisperModel(main_model_name, device=device, compute_type=main_compute)

    # ---------- Embeddings ----------
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embed_device)
    wake_ref = embed_bank(embedder, ["hey gordon"])  # wake phrase
    intent_refs = {k: embed_bank(embedder, v) for k, v in INTENTS.items()}
    finish_refs = embed_bank(embedder, ["finish recording", "stop recording"])  # end-of-record

    # Warmups (reduce first-call latency)
    _ = transcribe(wake_asr, np.zeros(int(0.3 * SR), dtype=np.float32), WAKE_VAD, 5)
    _ = transcribe(main_asr, np.zeros(int(0.3 * SR), dtype=np.float32), {"min_silence_duration_ms": 450, "speech_pad_ms": 120}, 12)
    _ = embedder.encode(["warmup"], normalize_embeddings=True)

    # Optional LLM assist for schedule parsing
    qwen_pipe = build_qwen()

    # ---------- Vision ----------
    cam_ind = 0
    w, h = 1280, 720
    mirror = True

    pose_model = "heavy"
    num_hands = 1

    pose_model_path = Path("models") / f"pose_landmarker_{pose_model}.task"
    hand_model_path = Path("models") / "hand_landmarker.task"

    ensure_model(
        pose_model_path,
        f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{pose_model}/float16/1/pose_landmarker_{pose_model}.task",
    )
    ensure_model(
        hand_model_path,
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    )

    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    hand_opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=max(1, int(num_hands)),
    )

    pose_lm = PoseLandmarker.create_from_options(pose_opts)
    hand_lm = HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(cam_ind, cv2.CAP_DSHOW if os.name == "nt" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cv2.namedWindow("Gordon", cv2.WINDOW_NORMAL)

    # ---------- Threads ----------
    running = True
    quit_flag = False

    # Vision shared state
    last_frame_bgr = None
    last_frame_lock = threading.Lock()
    latest_annotated_bgr = None
    annot_lock = threading.Lock()

    # Recording shared state
    recorded_frames = []
    rec_lock = threading.Lock()

    # Phases
    phase = "idle"
    prev_phase = phase

    # Wake & utterance buffers
    wake_rb = AudioRing(2.2, SR)
    cap_capture = UtteranceCapture(SR)

    # Wake timing
    last_wake_eval = 0.0
    wake_cooldown_until = 0.0

    # Scheduling
    actions = {}
    pending_action_frames = None
    scheduled_job = None

    # UI flashes
    wake_flash_until = 0.0
    mode_flash_text = ""
    mode_flash_until = 0.0
    schedule_flash_text = ""
    schedule_flash_until = 0.0

    # playback
    playback_start = 0.0
    playback_idx = 0

    def set_phase(p):
        nonlocal phase
        if p != phase:
            print(f"[PHASE] {phase} -> {p}", flush=True)
        phase = p

    def camera_loop():
        nonlocal last_frame_bgr
        last_cam_tick = 0.0
        while running:
            tick = time.perf_counter()
            dt = tick - last_cam_tick
            if dt < TARGET_DT:
                time.sleep(TARGET_DT - dt)
            last_cam_tick = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            with last_frame_lock:
                last_frame_bgr = frame_bgr

    def vision_loop():
        nonlocal latest_annotated_bgr, recorded_frames, playback_idx
        last_vis_tick = 0.0
        t0 = time.perf_counter()
        while running:
            now_tick = time.perf_counter()
            dt = now_tick - last_vis_tick
            if dt < TARGET_DT:
                time.sleep(TARGET_DT - dt)
            last_vis_tick = time.perf_counter()

            # Only run detectors when mirroring or recording or playing back
            active_for_vis = phase in ("mirror", "record", "playback")
            if not active_for_vis:
                with annot_lock:
                    latest_annotated_bgr = None
                time.sleep(0.002)
                continue

            with last_frame_lock:
                frame_bgr = None if last_frame_bgr is None else last_frame_bgr.copy()
            if frame_bgr is None:
                time.sleep(0.002)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.perf_counter() - t0) * 1000)

            pose_res = pose_lm.detect_for_video(mp_image, timestamp_ms)
            hand_res = hand_lm.detect_for_video(mp_image, timestamp_ms)
            pose_list = pose_res.pose_landmarks if hasattr(pose_res, "pose_landmarks") else []
            hand_list = hand_res.hand_landmarks if hasattr(hand_res, "hand_landmarks") else []
            hand_handedness = getattr(hand_res, "handedness", None)

            annotated_rgb = draw_overlays(frame_rgb, pose_list, hand_list, hand_handedness)
            out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            with annot_lock:
                latest_annotated_bgr = out_bgr

            if phase == "record":
                t_rel = int((time.perf_counter() - (playback_start or time.perf_counter())) * 1000)
                pose_pts = []
                if pose_list:
                    p0 = pose_list[0]
                    pose_pts = [[lm.x, lm.y, lm.z] for lm in p0]
                hands_pts = []
                for hnd in (hand_list or []):
                    hands_pts.append([[lm.x, lm.y, lm.z] for lm in hnd])
                with rec_lock:
                    recorded_frames.append({"t": t_rel, "pose": pose_pts, "hands": hands_pts})

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    vis_thread = threading.Thread(target=vision_loop, daemon=True)

    executor = ThreadPoolExecutor(max_workers=2)
    asr_future = None
    wake_future = None

    # Start
    stream.start()
    cam_thread.start()
    vis_thread.start()

    fps = 0.0
    last_disp = time.perf_counter()

    def start_playback_from(frames, label=None):
        nonlocal recorded_frames, playback_start, playback_idx, mode_flash_text, mode_flash_until
        with rec_lock:
            recorded_frames = [{"t": f["t"], "pose": [p[:] for p in f.get("pose", [])], "hands": [[hpt[:] for hpt in h] for h in f.get("hands", [])]} for f in frames]
        playback_start = time.perf_counter()
        playback_idx = 0
        set_phase("playback")
        mode_flash_text = f"PLAY: {label}" if label else "PLAYBACK"
        mode_flash_until = time.monotonic() + 1.2
        print(f"[PLAYBACK] start label='{label}' frames={len(recorded_frames)}", flush=True)

    def save_action_from_pending(name_text):
        nonlocal pending_action_frames, mode_flash_text, mode_flash_until
        if not pending_action_frames:
            set_phase("idle")
            mode_flash_text = "IDLE"; mode_flash_until = time.monotonic() + 1.0
            return
        display = (name_text or "").strip() or "unnamed"
        name_key = norm_text(display)
        emb = embedder.encode([name_key], normalize_embeddings=True)[0]
        with rec_lock:
            stored = [{"t": f["t"], "pose": [p[:] for p in f.get("pose", [])], "hands": [[hpt[:] for hpt in h] for h in f.get("hands", [])]} for f in pending_action_frames]
        actions[name_key] = {"display": display, "emb": emb, "frames": stored}
        pending_action_frames = None
        set_phase("idle")
        mode_flash_text = f"SAVED: {display}"; mode_flash_until = time.monotonic() + 1.5
        print(f"[ACTION_SAVED] name='{display}' key='{name_key}' frames={len(actions[name_key]['frames'])}", flush=True)

    def best_action_match(text):
        if not actions:
            return None, 0.0
        qv = embedder.encode([norm_text(text or "")], normalize_embeddings=True)[0]
        best_name, best_sim = None, 0.0
        for k, v in actions.items():
            sim = float(np.dot(qv, v["emb"]))
            if sim > best_sim:
                best_name, best_sim = k, sim
        return best_name, best_sim

    try:
        while running and not quit_flag:
            loop_start = time.perf_counter()

            # ===== AUDIO INGEST =====
            try:
                block = q.get(timeout=0.01)
            except queue.Empty:
                block = None

            if block is not None:
                wake_rb.push(block)
                # Only collect utterances in phases that expect speech
                if phase in ("command", "record", "name"):
                    cap_capture.feed(block)

            now_mono = time.monotonic()

            # ===== WAKE CHECK =====
            if (
                phase == "idle"
                and now_mono >= wake_cooldown_until
                and (now_mono - last_wake_eval) >= WAKE_CHECK_INTERVAL
                and wake_future is None
            ):
                # quick energy pre-gate to avoid ASR on silence
                tail = wake_rb.as_array()
                if rms(tail[-int(0.5*SR):] if len(tail) > int(0.5*SR) else tail) >= WAKE_ENERGY_THRESH:
                    last_wake_eval = now_mono
                    wake_future = executor.submit(lambda b: transcribe(wake_asr, b, WAKE_VAD, 5), tail.copy())

            if wake_future is not None and wake_future.done():
                win_text = wake_future.result()
                wake_future = None
                if win_text:
                    sim = best_sim_against(embedder, wake_ref, win_text, ngram_max=5)
                    print(f"[WAKE_ASR] '{win_text}' sim={sim:.3f}", flush=True)
                    if sim >= 0.82:
                        wake_cooldown_until = time.monotonic() + WAKE_REFRAC_S
                        wake_rb.clear()
                        set_phase("command")
                        mode_flash_text = "WAKE DETECTED — SAY A COMMAND"
                        mode_flash_until = time.monotonic() + 1.5
                        wake_flash_until = time.monotonic() + 1.5
                        cap_capture.reset()

            # ===== UTTERANCE CLOSE & ASR =====
            # Per-phase timing params
            if phase == "command":
                eos_ms, min_s, max_s = 400, 0.35, 4.0
            elif phase == "name":
                eos_ms, min_s, max_s = 500, 0.35, 4.0
            elif phase == "record":
                eos_ms, min_s, max_s = 700, 0.5, 20.0
            else:
                eos_ms = min_s = max_s = None

            if eos_ms is not None and asr_future is None and cap_capture.ready(min_s, eos_ms, max_s):
                audio_arr = cap_capture.pop()
                if phase in ("command", "name"):
                    asr_future = executor.submit(lambda a: transcribe(wake_asr, a, WAKE_VAD, 5), audio_arr)
                else:
                    asr_future = executor.submit(lambda a: transcribe(main_asr, a, {"min_silence_duration_ms": 450, "speech_pad_ms": 120}, 12), audio_arr)

            if asr_future is not None and asr_future.done():
                text = asr_future.result()
                asr_future = None
                print(f"[ASR][{phase}] '{text}'", flush=True)

                if phase == "command":
                    nt = norm_text(text or "")
                    label = INTENTS_NORM_MAP.get(nt)
                    chosen_from = "exact"
                    if label is None:
                        sims = {k: best_sim_against(embedder, v, text) for k, v in intent_refs.items()}
                        label = max(sims, key=sims.get) if sims else "idle"
                        chosen_from = f"fuzzy({sims.get(label,0.0):.2f})"
                        if sims.get(label, 0.0) < 0.70:
                            label = "idle"
                            chosen_from = "fallback-idle"
                    print(f"[INTENT] '{text}' -> {label} via {chosen_from}", flush=True)

                    if label == "mirror":
                        set_phase("mirror"); mode_flash_text = "MIRROR MODE"; mode_flash_until = time.monotonic() + 1.2
                    elif label == "record":
                        set_phase("record"); mode_flash_text = "RECORD MODE"; mode_flash_until = time.monotonic() + 1.2
                        with rec_lock:
                            recorded_frames = []
                        playback_start = time.perf_counter()
                    elif label == "idle":
                        set_phase("idle"); mode_flash_text = "IDLE"; mode_flash_until = time.monotonic() + 1.0
                        scheduled_job = None
                        print("[SCHEDULE] cleared", flush=True)
                    else:
                        act_name, sim = best_action_match(text or "")
                        print(f"[ACTION_MATCH] top='{act_name}' sim={sim:.3f}", flush=True)
                        if act_name and sim >= ACTION_SIM_THRESHOLD:
                            sched = decode_action_schedule(qwen_pipe, text or "")
                            if sched["kind"] == "count":
                                scheduled_job = {"kind":"count","frames":actions[act_name]["frames"],"label":actions[act_name]["display"],"count_remaining":int(sched["count"]),"waiting_to_fire":True}
                                schedule_flash_text = f"{actions[act_name]['display']}: {scheduled_job['count_remaining']} times"
                                schedule_flash_until = time.monotonic() + 1.8
                                print(f"[SCHEDULE] kind=count count={scheduled_job['count_remaining']} label='{scheduled_job['label']}'", flush=True)
                            elif sched["kind"] == "periodic":
                                period_s = float(sched["period_s"] or 0.0)
                                duration_s = sched["duration_s"]
                                end_time = (time.monotonic() + float(duration_s)) if duration_s is not None else None
                                scheduled_job = {"kind":"periodic","frames":actions[act_name]["frames"],"label":actions[act_name]["display"],"period_s":period_s,"end_time":end_time,"next_fire":time.monotonic(),"waiting_to_fire":True}
                                schedule_flash_text = f"{actions[act_name]['display']}: every {int(period_s)}s" + (f" for {int(end_time - time.monotonic())}s" if end_time else "")
                                schedule_flash_until = time.monotonic() + 1.8
                                print(f"[SCHEDULE] kind=periodic period_s={period_s} end={'INF' if end_time is None else int(end_time-time.monotonic())}s label='{scheduled_job['label']}'", flush=True)
                            elif sched["kind"] == "duration":
                                end_time = time.monotonic() + float(sched["duration_s"])
                                scheduled_job = {"kind":"duration","frames":actions[act_name]["frames"],"label":actions[act_name]["display"],"end_time":end_time,"waiting_to_fire":True}
                                schedule_flash_text = f"{actions[act_name]['display']}: for {int(end_time - time.monotonic())}s"
                                schedule_flash_until = time.monotonic() + 1.8
                                print(f"[SCHEDULE] kind=duration seconds={int(end_time-time.monotonic())} label='{scheduled_job['label']}'", flush=True)
                            else:
                                scheduled_job = {"kind":"count","frames":actions[act_name]["frames"],"label":actions[act_name]["display"],"count_remaining":1,"waiting_to_fire":True}
                                schedule_flash_text = f"{actions[act_name]['display']}: once"
                                schedule_flash_until = time.monotonic() + 1.6
                                print(f"[SCHEDULE] kind=count count=1 label='{scheduled_job['label']}'", flush=True)
                            set_phase("idle")
                        else:
                            set_phase("idle"); mode_flash_text = "IDLE"; mode_flash_until = time.monotonic() + 1.0

                elif phase == "record":
                    if text:
                        nt = norm_text(text)
                        is_finish = nt in FINISH_NORM_SET
                        if not is_finish:
                            fin_sim = best_sim_against(embedder, finish_refs, text, ngram_max=5)
                            is_finish = fin_sim >= 0.75
                        print(f"[RECORD_ASR] '{text}' finish={is_finish}", flush=True)
                        if is_finish:
                            with rec_lock:
                                pending_action_frames = [{"t": f["t"], "pose": [p[:] for p in f.get("pose", [])], "hands": [[hpt[:] for hpt in h] for h in f.get("hands", [])]} for f in recorded_frames]
                            print(f"[RECORD] captured frames={len(pending_action_frames)}", flush=True)
                            set_phase("name"); mode_flash_text = "NAME THIS ACTION — SAY A NAME"; mode_flash_until = time.monotonic() + 1.8
                            cap_capture.reset()
                        else:
                            try:
                                path = os.path.join(REC_DIR, f"utt_{int(time.time()*1000)}.wav")
                                save_wav_file(path, np.zeros(1, dtype=np.float32), SR)
                            except Exception:
                                pass
                            with open(SESSION_TXT, "a", encoding="utf-8") as f:
                                f.write(text + "\n")

                elif phase == "name":
                    save_action_from_pending((text or "").strip())

            # ===== SCHEDULE FIRING =====
            if scheduled_job and phase not in ("record", "name"):
                if scheduled_job["kind"] == "count":
                    if scheduled_job["count_remaining"] <= 0:
                        print("[SCHEDULE] done (count complete)", flush=True)
                        scheduled_job = None
                    elif phase != "playback" and scheduled_job["waiting_to_fire"]:
                        print(f"[SCHEDULE] fire (count left {scheduled_job['count_remaining']})", flush=True)
                        start_playback_from(scheduled_job["frames"], scheduled_job["label"])
                        scheduled_job["count_remaining"] -= 1
                        scheduled_job["waiting_to_fire"] = False
                elif scheduled_job["kind"] == "periodic":
                    if scheduled_job["end_time"] is not None and now_mono > scheduled_job["end_time"]:
                        print("[SCHEDULE] done (periodic duration over)", flush=True)
                        scheduled_job = None
                    elif phase != "playback" and scheduled_job["waiting_to_fire"] and now_mono >= scheduled_job.get("next_fire", now_mono):
                        print("[SCHEDULE] fire (periodic)", flush=True)
                        start_playback_from(scheduled_job["frames"], scheduled_job["label"])
                        scheduled_job["waiting_to_fire"] = False
                        scheduled_job["next_fire"] = now_mono + float(scheduled_job.get("period_s", 0.0))
                elif scheduled_job["kind"] == "duration":
                    if now_mono > scheduled_job["end_time"]:
                        print("[SCHEDULE] done (duration over)", flush=True)
                        scheduled_job = None
                    elif phase != "playback" and scheduled_job["waiting_to_fire"]:
                        print("[SCHEDULE] fire (duration)", flush=True)
                        start_playback_from(scheduled_job["frames"], scheduled_job["label"])
                        scheduled_job["waiting_to_fire"] = False

            # ===== RENDER =====
            with last_frame_lock:
                base = None if last_frame_bgr is None else last_frame_bgr.copy()

            if base is not None:
                if phase == "playback":
                    canvas = np.zeros_like(base)
                    with rec_lock:
                        rf_len = len(recorded_frames)
                        if rf_len:
                            elapsed = int((time.perf_counter() - playback_start) * 1000)
                            while playback_idx + 1 < rf_len and recorded_frames[playback_idx + 1]["t"] <= elapsed:
                                playback_idx += 1
                            rf = recorded_frames[min(playback_idx, rf_len - 1)]
                            pose_list = [rf.get("pose", [])] if rf.get("pose") else []
                            hand_list = rf.get("hands", [])
                        else:
                            pose_list, hand_list = [], []
                    out_rgb = draw_overlays(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), pose_list, hand_list)
                    frame = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                    if rf_len and playback_idx >= rf_len - 1 and elapsed >= recorded_frames[-1]["t"]:
                        playback_idx = 0
                        with rec_lock:
                            recorded_frames = []
                        mode_flash_text = "IDLE"; mode_flash_until = time.monotonic() + 1.0; set_phase("idle")
                else:
                    with annot_lock:
                        ann = None if latest_annotated_bgr is None else latest_annotated_bgr.copy()
                    frame = ann if (phase in ("mirror", "record") and ann is not None) else base

                now = time.perf_counter()
                dt_disp = now - last_disp
                last_disp = now
                fps = (0.9 * fps + 0.1 * (1.0 / dt_disp)) if dt_disp > 0 else fps

                cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"MODE: {phase.upper()}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                if time.monotonic() < wake_flash_until:
                    cv2.putText(frame, "WAKE PHRASE DETECTED", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
                if time.monotonic() < mode_flash_until and mode_flash_text:
                    cv2.putText(frame, mode_flash_text, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
                if time.monotonic() < schedule_flash_until and schedule_flash_text:
                    cv2.putText(frame, schedule_flash_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
                if phase == "name":
                    cv2.putText(frame, "SAY A SHORT NAME FOR THIS ACTION", (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)

                cv2.imshow("Gordon", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')):
                    quit_flag = True
                    break

            # ===== POST-PLAYBACK SCHEDULER TICK =====
            just_finished = (prev_phase == "playback" and phase != "playback")
            if just_finished and scheduled_job:
                if scheduled_job["kind"] == "count":
                    if scheduled_job["count_remaining"] > 0:
                        scheduled_job["waiting_to_fire"] = True
                        print(f"[SCHEDULE] ready next (count left {scheduled_job['count_remaining']})", flush=True)
                    else:
                        print("[SCHEDULE] complete (count)", flush=True)
                        scheduled_job = None
                elif scheduled_job["kind"] == "periodic":
                    scheduled_job["waiting_to_fire"] = True
                    print(f"[SCHEDULE] next at +{scheduled_job['period_s']}s", flush=True)
                elif scheduled_job["kind"] == "duration":
                    if time.monotonic() <= scheduled_job["end_time"]:
                        scheduled_job["waiting_to_fire"] = True
                        print("[SCHEDULE] repeat during duration", flush=True)
                    else:
                        print("[SCHEDULE] complete (duration)", flush=True)
                        scheduled_job = None

            prev_phase = phase

            # pacing
            loop_elapsed = time.perf_counter() - loop_start
            if loop_elapsed < TARGET_DT:
                time.sleep(TARGET_DT - loop_elapsed)

    finally:
        running = False
        try:
            stream.stop(); stream.close()
        except Exception:
            pass
        try:
            cam_thread.join(timeout=1.0); vis_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            pose_lm.close(); hand_lm.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
