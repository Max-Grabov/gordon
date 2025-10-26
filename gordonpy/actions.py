from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import time, json, re

from .utils import norm_text

@dataclass
class ActionFrame:
    t_ms: int
    motors: Any
    claw_open: Optional[bool]

@dataclass
class ReplaySpec:
    period_s: Optional[float] = None
    duration_s: Optional[float] = None
    count: Optional[int] = None

class ActionRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = None
        self.frames: List[ActionFrame] = []

    def start(self):
        self.reset()
        self.t0 = time.perf_counter()

    def feed(self, motors, claw_open):
        if self.t0 is None or motors is None:
            return
        t_ms = int((time.perf_counter() - self.t0) * 1000.0)
        self.frames.append(ActionFrame(t_ms=t_ms, motors=motors, claw_open=claw_open))

    def stop(self) -> List[ActionFrame]:
        return list(self.frames)

class ActionLibrary:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all()

    @staticmethod
    def _safe_name(s: str) -> str:
        s = norm_text(s)
        s = re.sub(r"[^a-z0-9\-_\s]+", "", s).strip()
        s = re.sub(r"\s+", "_", s)
        return s or "action"

    def _load_all(self):
        self._cache.clear()
        for p in self.root.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                name = p.stem
                self._cache[name] = data.get("frames", [])
            except Exception:
                pass

    def list_actions(self) -> List[str]:
        return sorted(self._cache.keys())

    def has(self, name: str) -> bool:
        return name in self._cache

    def save(self, raw_name: str, frames: List[ActionFrame]) -> str:
        base = self._safe_name(raw_name)
        name = base
        i = 2
        while self.has(name):
            name = f"{base}_{i}"
            i += 1
        frames_j = [dict(t_ms=f.t_ms, motors=f.motors, claw_open=f.claw_open) for f in frames]
        (self.root / f"{name}.json").write_text(json.dumps({"frames": frames_j}, indent=2))
        self._cache[name] = frames_j
        return name

    def get(self, name: str) -> Optional[List[ActionFrame]]:
        j = self._cache.get(name)
        if j is None:
            return None
        return [ActionFrame(**f) for f in j]

# ---- Actuation stub ----------------------------------------------------------

def send_motors(motors, claw_open: Optional[bool]):
    """Replace with your hardware call."""
    print("[REPLAY] motors:", motors, "claw_open:", claw_open)

# ---- Replay runner -----------------------------------------------------------

def replay_action(frames: List[ActionFrame]):
    if not frames:
        return
    t0 = time.perf_counter()
    start_ms = frames[0].t_ms
    for f in frames:
        target = (f.t_ms - start_ms) / 1000.0
        now = time.perf_counter() - t0
        if target > now:
            time.sleep(target - now)
        send_motors(f.motors, f.claw_open)

def run_replay(frames: List[ActionFrame], spec: ReplaySpec):
    if not frames:
        return

    def _once():
        replay_action(frames)

    if spec.period_s is None and spec.duration_s is None and spec.count is None:
        _once(); return

    if spec.count is not None and spec.period_s is not None:
        for _ in range(max(0, int(spec.count))):
            _once(); time.sleep(spec.period_s)
        return

    if spec.count is not None and spec.period_s is None:
        for _ in range(max(0, int(spec.count))):
            _once()
        return

    if spec.period_s is not None and spec.duration_s is not None:
        end_t = time.perf_counter() + max(0.0, spec.duration_s)
        while time.perf_counter() < end_t:
            _once(); time.sleep(spec.period_s)
        return

    if spec.period_s is not None:
        _once(); time.sleep(spec.period_s); return

    if spec.duration_s is not None:
        end_t = time.perf_counter() + max(0.0, spec.duration_s)
        while time.perf_counter() < end_t:
            _once()
        return


class ScheduleParser:
    def __init__(self):
        self._model = None
        self._tok = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                model_id = "Qwen/Qwen2.5-1.5B-Instruct"
                self._tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
                )
            except Exception as e:
                print("[LLM] Could not load Qwen2.5-1.5B-Instruct:", e)
                self._model = None

    @staticmethod
    def _regex_parse(text: str) -> ReplaySpec:
        t = norm_text(text)

        def num(s):
            try: return float(s)
            except: return None

        units = {
            "second": 1, "seconds": 1, "sec": 1, "secs": 1,
            "minute": 60, "minutes": 60, "min": 60, "mins": 60,
            "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
        }
        period_s = None; duration_s = None; count = None

        m = re.search(r"every\s+(\d+(?:\.\d+)?)\s+([a-z]+)", t)
        if m and m.group(2) in units:
            n = num(m.group(1)); period_s = n * units[m.group(2)] if n is not None else None

        m = re.search(r"for\s+(\d+(?:\.\d+)?)\s+([a-z]+)", t)
        if m and m.group(2) in units:
            n = num(m.group(1)); duration_s = n * units[m.group(2)] if n is not None else None

        m = re.search(r"(\d+)\s+times?", t)
        if m:
            count = int(m.group(1))

        return ReplaySpec(period_s=period_s, duration_s=duration_s, count=count)

    def parse(self, full_command: str) -> ReplaySpec:
        self._ensure_model()
        if self._model is None:
            return self._regex_parse(full_command)

        import torch, json
        prompt = (
            "You extract a schedule from a command. Return ONLY JSON with keys "
            "period_s, duration_s, count. Missing fields are null. Examples: "
            "'stir every 5 minutes' -> {\"period_s\":300,\"duration_s\":null,\"count\":null}; "
            "'stir every 5 minutes for an hour' -> {\"period_s\":300,\"duration_s\":3600,\"count\":null}; "
            "'stir 6 times' -> {\"period_s\":null,\"duration_s\":null,\"count\":6}.\nNow parse:\n" + full_command
        )
        messages = [
            {"role": "system", "content": "You output only JSON with keys period_s, duration_s, count."},
            {"role": "user", "content": prompt},
        ]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**input_ids, max_new_tokens=96)
        gen = self._tok.decode(out[0][input_ids["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        gen = gen.splitlines()[0].strip()
        try:
            j = re.search(r"\{.*\}", gen)
            if j:
                obj = json.loads(j.group(0))
                def _get(x): return None if x is None else float(x)
                return ReplaySpec(
                    period_s=_get(obj.get("period_s")),
                    duration_s=_get(obj.get("duration_s")),
                    count=int(obj["count"]) if obj.get("count") is not None else None,
                )
        except Exception as e:
            print("[LLM] JSON parse error, falling back:", e)
        return self._regex_parse(full_command)
