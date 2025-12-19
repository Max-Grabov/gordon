import os, time, queue, threading, cv2, numpy as np
from pathlib import Path

from gordonpy import config
from gordonpy.utils import norm_text
from gordonpy.intents import INTENTS, INTENTS_NORM_MAP, match_intent_keyword
from gordonpy.audio_io import AudioRing, UtteranceCapture, make_audio_cb, make_input_stream
from gordonpy.vad import VADDetector
from gordonpy.asr import AsrEngine
from gordonpy.ui import UIState, draw_enhanced_ui
from gordonpy.vision import MirrorProcessor
from gordonpy.actions import (
    ActionRecorder, ActionLibrary, ScheduleParser,
    ReplaySpec, run_replay
)

import torch

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
ACTIONS_DIR = RECORDINGS_DIR / "actions"
ACTIONS_DIR.mkdir(parents=True, exist_ok=True)

q_audio = queue.Queue()
asr_in   = queue.Queue(maxsize=5)
asr_out  = queue.Queue()

ui_state = UIState()
state_lock = threading.Lock()
state = {
    "running": True, "quit_flag": False, "phase": "idle",
    "last_frame_bgr": None, "latest_annotated_bgr": None,
    "command_start_time": None,
}

# Recording / Replay singletons in state
state.update({
    "recorder": ActionRecorder(),
    "awaiting_name": False,
    "pending_frames": None,
    "actions": ActionLibrary(ACTIONS_DIR),
    "schedule_parser": ScheduleParser(),
})

ring = AudioRing(config.AUDIO_BUFFER_S, config.SR)
utter = UtteranceCapture(config.SR)
audio_cb = make_audio_cb(q_audio)
stream = make_input_stream(config.SR, int(config.SR*0.04), audio_cb)

device = "cuda" if torch.cuda.is_available() else "cpu"
asr = AsrEngine(device=device)
asr.warmup(config.SR)
vad = VADDetector(config.SR)
mirror = MirrorProcessor()

wake_ref = asr.embed_bank(["gordon"])  # wake word
intent_refs = {k: asr.embed_bank(v) for k, v in INTENTS.items()}
print("Ready!")

# Camera
cap = cv2.VideoCapture(config.CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except: pass
cv2.namedWindow("Gordon", cv2.WINDOW_NORMAL)

# -----------------------------------------------------------------------------
# Phase helper
# -----------------------------------------------------------------------------
def set_phase(p):
    with state_lock:
        old = state["phase"]
        if p != old:
            print(f"[PHASE] {old} -> {p}")
            state["phase"] = p
            if p == "command":
                state["command_start_time"] = time.monotonic()
                utter.reset(); ui_state.is_listening=True
            elif p == "idle":
                state["command_start_time"] = None
                ui_state.is_listening=False; ui_state.is_processing=False
    ui_state.phase = p

# -----------------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------------
def asr_worker():
    while True:
        with state_lock:
            if not state["running"]: break
        try:
            task_type, audio_data = asr_in.get(timeout=0.1)
        except queue.Empty:
            continue
        ui_state.is_processing = True; ui_state.is_listening = False
        try:
            text = asr.transcribe(audio_data, config.WAKE_VAD, chunk_len=5)
        except Exception as e:
            print(f"[ASR] error: {e}"); text = ""
        asr_out.put((task_type, text)); ui_state.is_processing = False

def camera_loop():
    while True:
        with state_lock:
            if not state["running"]: break
        ok, frame_bgr = cap.read()
        if not ok:
            time.sleep(0.005); continue
        if config.MIRROR_FLIP: frame_bgr = cv2.flip(frame_bgr, 1)
        with state_lock:
            state["last_frame_bgr"] = frame_bgr


def vision_loop():
    t0 = time.perf_counter()
    while True:
        with state_lock:
            if not state["running"]: break
            active = state["phase"] in ("mirror", "record")
        if not active:
            with state_lock: state["latest_annotated_bgr"] = None
            time.sleep(0.002); continue
        with state_lock:
            src = None if state["last_frame_bgr"] is None else state["last_frame_bgr"].copy()
        if src is None:
            time.sleep(0.002); continue
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        ts_ms = int((time.perf_counter() - t0) * 1000)
        out_bgr, info = mirror.process(rgb, ts_ms)
        # Optional: observe mirror output
        # if info and info.get("motors") is not None:
        #     print(info.get("motors"), info.get("claw_open"))

        # Feed recorder if in record mode
        with state_lock:
            state["latest_annotated_bgr"] = out_bgr
            if state["phase"] == "record" and info is not None:
                motors = info.get("motors", None)
                claw = info.get("claw_open", None)
                state["recorder"].feed(motors, claw)

# Start threads
stream.start()
threads = [
    threading.Thread(target=asr_worker, daemon=True),
    threading.Thread(target=camera_loop, daemon=True),
    threading.Thread(target=vision_loop, daemon=True),
]
for t in threads: t.start()

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
wake_cooldown_until = 0.0
last_vad_check = 0.0
vad_check_interval = 0.2
vad_window_s = 1.0
fps, last_disp = 0.0, time.perf_counter()

try:
    while True:
        with state_lock:
            if not state["running"] or state["quit_flag"]: break
            phase = state["phase"]

        # ingest audio
        latest_level = 0.0
        for _ in range(10):
            try:
                pkt = q_audio.get(timeout=0.001)
            except queue.Empty:
                break
            ring.push(pkt.data); latest_level = max(latest_level, pkt.rms_level)
            if phase == "command": utter.feed(pkt.data)
        ui_state.audio_level = latest_level

        # command timeout
        now_mono = time.monotonic()
        if phase == "command":
            with state_lock: t0 = state["command_start_time"]
            if t0 and (now_mono - t0) > config.COMMAND_TIMEOUT_S:
                set_phase("idle"); ui_state.flash_message="COMMAND TIMEOUT"; ui_state.flash_until=now_mono+1.0

        # wake detection
        if (phase == "idle") and (now_mono >= wake_cooldown_until) and ((now_mono - last_vad_check) >= vad_check_interval):
            last_vad_check = now_mono
            tail = ring.get_tail(vad_window_s)
            if len(tail) > config.SR * 0.3:
                from gordonpy.utils import rms
                if rms(tail) >= config.SPEAKING_RMS and vad.is_speech_present(tail):
                    try:
                        asr_in.put_nowait(("wake", tail.copy())); ui_state.is_processing=True
                    except queue.Full:
                        pass

        # ASR results
        try:
            while True:
                task_type, text = asr_out.get_nowait()
                print(f"[ASR][{task_type}] '{text}'")
                ui_state.is_processing=False
                if task_type == "wake" and text:
                    nt = norm_text(text); triggered=False
                    if "gordon" in nt.split():
                        triggered=True; reason="keyword"; sim=1.0
                    else:
                        sim = asr.best_sim_against(wake_ref, nt, ngram_max=5); reason=f"cosine={sim:.3f}"
                        if sim >= 0.82: triggered=True
                    if triggered:
                        wake_cooldown_until = time.monotonic() + 2.0
                        ring.clear(); set_phase("command")
                        ui_state.wake_detected=True; ui_state.wake_until=time.monotonic()+1.5
                        ui_state.flash_message="LISTENING FOR COMMAND"; ui_state.flash_until=time.monotonic()+1.5
                        ui_state.last_transcript=text; ui_state.transcript_until=time.monotonic()+3.0
                elif task_type == "command" and phase == "command":
                    nt = norm_text(text or "")
                    ui_state.last_transcript = text if text else "(no speech)"
                    ui_state.transcript_until = time.monotonic() + 4.0

                    # 1) Naming flow takes precedence
                    with state_lock:
                        if state["awaiting_name"]:
                            name_raw = (text or "action").strip()
                            saved_name = state["actions"].save(name_raw, state["pending_frames"] or [])
                            state["awaiting_name"] = False
                            state["pending_frames"] = None
                            set_phase("idle")
                            ui_state.flash_message = f'SAVED "{saved_name}"'
                            ui_state.flash_until = time.monotonic() + 2.0
                            continue

                    # 2) Normal intent matching
                    label = match_intent_keyword(nt) or INTENTS_NORM_MAP.get(nt)
                    if label is None:
                        sims = {k: asr.best_sim_against(v, nt) for k, v in intent_refs.items()}
                        label = max(sims, key=sims.get) if sims else "idle"
                        if sims.get(label, 0.0) < 0.70:
                            label = "idle"

                    # 3) In record mode, allow bare "stop" to map to stop_record
                    with state_lock:
                        current_phase = state["phase"]
                    if current_phase == "record" and ("stop" in nt or "finish" in nt or "end recording" in nt):
                        label = "stop_record"

                    # 4) Handle intents
                    if label == "mirror":
                        set_phase("mirror")
                        ui_state.flash_message="MIRROR MODE ACTIVATED"; ui_state.flash_until=time.monotonic()+1.5

                    elif label == "record":
                        with state_lock:
                            state["recorder"].start()
                        set_phase("record")
                        ui_state.flash_message="RECORDING STARTED"; ui_state.flash_until=time.monotonic()+1.5

                    elif label == "stop_record":
                        with state_lock:
                            frames = state["recorder"].stop()
                            state["pending_frames"] = frames
                            state["awaiting_name"] = True
                        set_phase("command")  # capture next utterance as name
                        ui_state.flash_message="SAY A NAME FOR THIS ACTION"; ui_state.flash_until=time.monotonic()+2.5

                    else:
                        # 5) Playback by action name (LLM parses schedule)
                        with state_lock:
                            names = state["actions"].list_actions()
                        chosen = None
                        for nm in sorted(names, key=len, reverse=True):  # prefer longest
                            if nm in nt:
                                chosen = nm; break
                        if chosen:
                            with state_lock:
                                frames = state["actions"].get(chosen)
                                parser = state["schedule_parser"]
                            spec = parser.parse(text or "")
                            ui_state.flash_message=f'REPLAY "{chosen}"'; ui_state.flash_until=time.monotonic()+1.5
                            set_phase("idle")

                            def _runner():
                                try: run_replay(frames, spec)
                                except Exception as e: print("[REPLAY] error:", e)
                            threading.Thread(target=_runner, daemon=True).start()
                        else:
                            set_phase("idle")
                            ui_state.flash_message="IDLE MODE"; ui_state.flash_until=time.monotonic()+1.0
        except queue.Empty:
            pass

        # utterance endpointing
        if phase == "command":
            if utter.ready(min_s=0.35, eos_ms=400, max_s=4.0):
                try:
                    asr_in.put_nowait(("command", utter.pop()))
                    ui_state.is_listening=False; ui_state.is_processing=True
                except queue.Full:
                    pass

        # display
        with state_lock:
            base = None if state["last_frame_bgr"] is None else state["last_frame_bgr"].copy()
            ann  = None if state["latest_annotated_bgr"] is None else state["latest_annotated_bgr"].copy()
        if base is not None:
            frame = ann if (phase=="mirror" and ann is not None) or (phase=="record" and ann is not None) else base
            now = time.perf_counter(); dt = now - last_disp; last_disp = now
            if dt > 0: fps = 0.9*fps + 0.1*(1.0/dt)
            ui_state.fps = fps
            frame = draw_enhanced_ui(frame, ui_state)
            cv2.imshow("Gordon", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                with state_lock: state["quit_flag"] = True
                break
finally:
    with state_lock: state["running"]=False
    try:
        stream.stop(); stream.close()
    except: pass
    try: mirror.close()
    except: pass
    try: cap.release()
    except: pass
    cv2.destroyAllWindows()
    print("Bye Bye!! (otaku style).")

