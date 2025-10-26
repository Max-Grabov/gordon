import torch, numpy as np
from . import config

class VADDetector:
    def __init__(self, sr: int):
        self.sr = sr
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad',
            force_reload=False, onnx=False
        )
        (self.get_speech_timestamps, *_) = utils
        self.model.eval()

    def is_speech_present(self, audio: np.ndarray) -> bool:
        if len(audio) == 0: return False
        audio_tensor = torch.from_numpy(audio).float()
        ts = self.get_speech_timestamps(
            audio_tensor, self.model,
            threshold=config.SILERO_THRESHOLD,
            sampling_rate=self.sr,
            min_speech_duration_ms=config.SILERO_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=config.SILERO_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=config.SILERO_SPEECH_PAD_MS,
            return_seconds=False
        )
        return len(ts) > 0