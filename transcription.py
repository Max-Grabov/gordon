import os, re, time, queue
import numpy as np
import torch
import sounddevice as sd
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

SR = 16000
BLOCK_MS = 40
BEAM = 5

WAKE_PHRASE = "hey gordon"
WAKE_MODEL_NAME = "tiny.en"
WAKE_WINDOW_S = 2.0
WAKE_STEP_S = 0.25
WAKE_MIN_RMS = 0.003
WAKE_SIM_THRESHOLD = 0.82
WAKE_VAD = {"min_silence_duration_ms": 250, "speech_pad_ms": 80}

CMD_END_SIL_MS = 400
CMD_MIN_S = 0.35
CMD_MAX_S = 4.0
CMD_NGRAM_MAX = 5

MAIN_MODEL_NAME = "large-v3"
MAIN_CHUNK_LEN = 12
MAIN_VAD = {"min_silence_duration_ms": 450, "speech_pad_ms": 120}
UTT_END_SIL_MS = 700
UTT_MIN_S = 0.5
UTT_MAX_S = 20.0

INTENTS = {
    "mirror": [
        "mirror mode", "start mirroring", "mirror"
    ],
    "record": [
        "record mode", "record", "begin recording"
    ],
    "idle": [
        "idle mode", "go idle", "stop", "sleep", "standby", "cancel"
    ],
}
FINISH_RECORDING_PHRASES = [
    "finish recording", "stop recording", "that is all", "we are done recording",
    "end dictation", "save and finish", "wrap up recording"
]
INTENT_SIM_THRESHOLD = 0.70
FINISH_SIM_THRESHOLD = 0.75

REC_DIR = "recordings"
SESSION_TXT = os.path.join(REC_DIR, "session.txt")
SAVE_WAV = True

def norm_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(x * x)))

def save_wav(path, audio, sr):
    try:
        import soundfile as sf
        sf.write(path, audio, sr)
    except Exception:
        import wave
        y = np.clip(audio, -1.0, 1.0)
        y = (y * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(y.tobytes())

def best_sim_against(embedder, refs_emb, text: str, ngram_max=1) -> float:
    t = norm_text(text)
    if not t: return 0.0
    toks = t.split()
    cands = [t]
    for n in range(2, min(ngram_max, len(toks)) + 1):
        cands.append(" ".join(toks[-n:]))
    embs = embedder.encode(cands, normalize_embeddings=True)
    sims = np.dot(embs, refs_emb.T)
    return float(np.max(sims))

def embed_bank(embedder, phrases):
    embs = embedder.encode([norm_text(p) for p in phrases], normalize_embeddings=True)
    return embs

q = queue.Queue()
def audio_cb(indata, frames, time_info, status):
    if status: pass
    q.put(indata.copy().reshape(-1))

blocksize = int(SR * (BLOCK_MS / 1000.0))
stream = sd.InputStream(channels=1, samplerate=SR, dtype="float32",
                        blocksize=blocksize, callback=audio_cb)

wake_asr = WhisperModel(WAKE_MODEL_NAME, device="cuda", compute_type="float16")
main_asr = WhisperModel(MAIN_MODEL_NAME, device="cuda", compute_type="float16")
 
embed_device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embed_device)

wake_ref = embed_bank(embedder, [WAKE_PHRASE])
intent_refs = {k: embed_bank(embedder, v) for k, v in INTENTS.items()}
finish_refs = embed_bank(embedder, FINISH_RECORDING_PHRASES)

_ = list(wake_asr.transcribe(np.zeros(int(0.3 * SR), dtype=np.float32),
                             language="en", beam_size=BEAM, word_timestamps=False,
                             vad_filter=True, vad_parameters=WAKE_VAD,
                             condition_on_previous_text=False, chunk_length=5)[0])
_ = list(main_asr.transcribe(np.zeros(int(0.3 * SR), dtype=np.float32),
                             language="en", beam_size=BEAM, word_timestamps=False,
                             vad_filter=True, vad_parameters=MAIN_VAD,
                             condition_on_previous_text=False, chunk_length=MAIN_CHUNK_LEN)[0])
_ = embedder.encode(["warmup"], normalize_embeddings=True)

phase = "idle"
wake_buf = np.zeros(0, dtype=np.float32)
samples_since_wake_check = 0
cap_buf = np.zeros(0, dtype=np.float32)
last_voice_t = None
utt_idx = 0

os.makedirs(REC_DIR, exist_ok=True)
stream.start()

dev = stream.device
in_index = dev[0] if isinstance(dev, tuple) else dev
try:
    dinfo = sd.query_devices(in_index)
    hostapi = sd.query_hostapis(dinfo['hostapi'])['name']
    print(f"Using input device #{in_index}: {dinfo['name']} [{hostapi}] @ {SR} Hz")
except Exception: pass

print("Idle. Say the wake phrase to start…")

try:
    while True:
        block = q.get()

        wake_buf = np.concatenate([wake_buf, block])
        if len(wake_buf) > int(WAKE_WINDOW_S * SR):
            wake_buf = wake_buf[-int(WAKE_WINDOW_S * SR):]
        samples_since_wake_check += len(block)

        if samples_since_wake_check >= int(WAKE_STEP_S * SR):
            samples_since_wake_check = 0
            if rms(wake_buf) >= WAKE_MIN_RMS:
                segs, _ = wake_asr.transcribe(
                    wake_buf, language="en", beam_size=BEAM, word_timestamps=False,
                    vad_filter=True, vad_parameters=WAKE_VAD,
                    condition_on_previous_text=False, chunk_length=5,
                )
                win_text = " ".join(s.text.strip() for s in segs).strip()
                if win_text:
                    sim = best_sim_against(embedder, wake_ref, win_text, ngram_max=CMD_NGRAM_MAX)
                    if sim >= WAKE_SIM_THRESHOLD:
                        print("\nWake detected. Say a command…")
                        phase = "command"
                        cap_buf = np.zeros(0, dtype=np.float32)
                        last_voice_t = time.monotonic()
                        continue

        if phase == "idle":
            continue

        cap_buf = np.concatenate([cap_buf, block])
        if rms(block) > WAKE_MIN_RMS:
            last_voice_t = time.monotonic()

        dur = len(cap_buf) / SR
        silent_for = (time.monotonic() - last_voice_t) if last_voice_t else 0.0

        if phase == "command":
            eos_ms, min_s, max_s = CMD_END_SIL_MS, CMD_MIN_S, CMD_MAX_S
        else:
            eos_ms, min_s, max_s = UTT_END_SIL_MS, UTT_MIN_S, UTT_MAX_S

        should_close = (dur >= min_s) and (silent_for >= eos_ms / 1000.0 or dur >= max_s)
        if not should_close:
            continue

        audio = cap_buf.copy()
        if len(audio) > int(0.1 * SR):
            audio = audio[:-int(0.05 * SR)]

        if phase == "command":
            segs, _ = wake_asr.transcribe(
                audio, language="en", beam_size=BEAM, word_timestamps=False,
                vad_filter=True, vad_parameters=WAKE_VAD,
                condition_on_previous_text=False, chunk_length=5,
            )
            cmd_text = " ".join(s.text.strip() for s in segs).strip()

            sims = {k: best_sim_against(embedder, v, cmd_text) for k, v in intent_refs.items()}
            label = max(sims, key=sims.get)
            score = sims[label]
            if score < INTENT_SIM_THRESHOLD:
                label = "idle"

            if label == "mirror":
                print("Mirror mode ON. Say wake phrase then 'idle' to exit.")
                phase = "mirror"
            elif label == "record":
                print("Record mode ON. Say wake phrase then 'idle' to exit, or say 'finish recording'.")
                phase = "record"
            else:
                print("Back to idle")
                phase = "idle"

            cap_buf = np.zeros(0, dtype=np.float32)
            last_voice_t = None
            continue

        elif phase == "mirror":
            segs, _ = main_asr.transcribe(
                audio, language="en", beam_size=BEAM, word_timestamps=False,
                vad_filter=True, vad_parameters=MAIN_VAD,
                condition_on_previous_text=False, chunk_length=MAIN_CHUNK_LEN,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
            if text:
                print(text, flush=True)
            cap_buf = np.zeros(0, dtype=np.float32)
            last_voice_t = None
            continue

        elif phase == "record":
            segs, _ = main_asr.transcribe(
                audio, language="en", beam_size=BEAM, word_timestamps=False,
                vad_filter=True, vad_parameters=MAIN_VAD,
                condition_on_previous_text=False, chunk_length=MAIN_CHUNK_LEN,
            )
            text = " ".join(s.text.strip() for s in segs).strip()

            if text:
                fin_sim = best_sim_against(embedder, finish_refs, text, ngram_max=CMD_NGRAM_MAX)
                if fin_sim >= FINISH_SIM_THRESHOLD:
                    print("recorded!")
                    if SAVE_WAV:
                        os.makedirs(REC_DIR, exist_ok=True)
                        path = os.path.join(REC_DIR, f"fin_{int(time.time())}.wav")
                        save_wav(path, audio, SR)
                    phase = "idle"
                    cap_buf = np.zeros(0, dtype=np.float32)
                    last_voice_t = None
                    print("Back to idle")
                    continue

            if text:
                print(text, flush=True)
                os.makedirs(REC_DIR, exist_ok=True)
                if SAVE_WAV:
                    path = os.path.join(REC_DIR, f"utt_{int(time.time()*1000)}.wav")
                    save_wav(path, audio, SR)
                with open(SESSION_TXT, "a", encoding="utf-8") as f:
                    f.write(text + "\n")

            cap_buf = np.zeros(0, dtype=np.float32)
            last_voice_t = None
            continue

except KeyboardInterrupt:
    pass
finally:
    stream.stop()
    stream.close()
