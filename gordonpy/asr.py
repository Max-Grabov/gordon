import numpy as np
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from .utils import norm_text

class AsrEngine:
    def __init__(self, device="cpu"):
        self.whisper = WhisperModel("tiny.en", device=device, compute_type = "float16" if device == "cuda" else "float32")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def warmup(self, sr: int):
        _ = self.transcribe(np.zeros(int(0.3*sr), dtype=np.float32), vad_params={"min_silence_duration_ms":250,"speech_pad_ms":80})
        _ = self.embedder.encode(["warmup"], normalize_embeddings=True)

    def transcribe(self, audio, vad_params, chunk_len=5):
        segs, _ = self.whisper.transcribe(
            audio, language="en", beam_size=5, word_timestamps=False,
            vad_filter=True, vad_parameters=vad_params,
            condition_on_previous_text=False, chunk_length=chunk_len,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    def embed_bank(self, phrases):
        return self.embedder.encode([norm_text(p) for p in phrases], normalize_embeddings=True)

    def best_sim_against(self, refs_emb, text: str, ngram_max=1) -> float:
        t = norm_text(text)
        if not t: return 0.0
        toks = t.split()
        cands = [t]
        for n in range(2, min(ngram_max, len(toks))+1): cands.append(" ".join(toks[-n:]))
        embs = self.embedder.encode(cands, normalize_embeddings=True)
        sims = np.dot(embs, refs_emb.T)
        return float(np.max(sims))
