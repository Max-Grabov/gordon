import threading, time, numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Deque
import sounddevice as sd
from .utils import rms

@dataclass
class AudioPacket:
    data: np.ndarray
    timestamp: float
    rms_level: float

class AudioRing:
    def __init__(self, max_seconds: float, sr: int):
        self.sr = sr
        self.max_samples = int(max_seconds * sr)
        self.chunks: Deque[np.ndarray] = deque()
        self.samples = 0
        self.lock = threading.Lock()

    def push(self, block: np.ndarray):
        b = block.astype(np.float32, copy=False)
        with self.lock:
            self.chunks.append(b.copy())
            self.samples += len(b)
            while self.samples > self.max_samples and self.chunks:
                old = self.chunks.popleft()
                self.samples -= len(old)

    def get_tail(self, seconds: float) -> np.ndarray:
        with self.lock:
            if not self.chunks: return np.zeros(0, dtype=np.float32)
            arr = np.concatenate(list(self.chunks))
            tail = int(seconds * self.sr)
            return arr[-tail:] if len(arr) > tail else arr

    def clear(self):
        with self.lock:
            self.chunks.clear(); self.samples = 0

class UtteranceCapture:
    def __init__(self, sr: int):
        self.sr = sr; self.blocks = []; self.samples = 0
        self.last_voice_t = None; self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.blocks.clear(); self.samples = 0; self.last_voice_t = None

    def feed(self, block: np.ndarray, voice_thresh=0.003):
        with self.lock:
            self.blocks.append(block.copy()); self.samples += len(block)
            if rms(block) > voice_thresh: self.last_voice_t = time.monotonic()

    def ready(self, min_s: float, eos_ms: int, max_s: float) -> bool:
        with self.lock:
            dur = self.samples / float(self.sr)
            silent_for = (time.monotonic() - self.last_voice_t) if self.last_voice_t else 0.0
            return (dur >= min_s) and ((silent_for >= (eos_ms/1000.0)) or (dur >= max_s))

    def pop(self) -> np.ndarray:
        with self.lock:
            arr = np.concatenate(self.blocks) if self.blocks else np.zeros(0, dtype=np.float32)
            self.blocks.clear(); self.samples = 0; self.last_voice_t = None
            if len(arr) > int(0.05 * self.sr): arr = arr[:-int(0.025 * self.sr)]
            return arr

def make_audio_cb(q):
    from .audio_io import AudioPacket  # local import to avoid cycle when packaging
    def audio_cb(indata, frames, time_info, status):
        if status: print(f"[AUDIO] {status}", flush=True)
        data = indata.copy().reshape(-1)
        q.put(AudioPacket(data=data, timestamp=time.monotonic(), rms_level=rms(data)))
    return audio_cb

def make_input_stream(sr: int, blocksize: int, callback):
    sd.default.channels = 1; sd.default.samplerate = sr
    return sd.InputStream(dtype="float32", blocksize=blocksize, callback=callback)

