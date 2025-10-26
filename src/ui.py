from dataclasses import dataclass
import time, numpy as np, cv2
from . import config

@dataclass
class UIState:
    phase: str = 'idle'
    fps: float = 0.0
    audio_level: float = 0.0
    is_listening: bool = False
    is_processing: bool = False
    flash_message: str = ''
    flash_until: float = 0.0
    wake_detected: bool = False
    wake_until: float = 0.0
    last_transcript: str = ''
    transcript_until: float = 0.0

def draw_enhanced_ui(frame, ui_state: UIState):
    h, w = frame.shape[:2]; now = time.monotonic()
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, f"FPS: {ui_state.fps:5.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    phase_colors = {'idle':(128,128,128),'command':(0,215,255),'mirror':(0,255,0),'record':(0,0,255)}
    cv2.putText(frame, f"MODE: {ui_state.phase.upper()}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_colors.get(ui_state.phase,(255,255,255)), 2, cv2.LINE_AA)

    # Audio meter
    meter_x, meter_y, meter_w, meter_h = 10, 85, 300, 20
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x+meter_w, meter_y+meter_h), (50,50,50), -1)
    level_norm = min(ui_state.audio_level / config.MAX_RMS_DISPLAY, 1.0); level_w = int(meter_w * level_norm)
    level_color = (100,100,100) if ui_state.audio_level < config.SILENCE_RMS else ((0,0,255) if ui_state.audio_level > config.MAX_RMS_DISPLAY*0.8 else (0,255,0))
    if level_w > 0: cv2.rectangle(frame, (meter_x, meter_y), (meter_x+level_w, meter_y+meter_h), level_color, -1)
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x+meter_w, meter_y+meter_h), (255,255,255), 1)
    cv2.putText(frame, f"MIC: {ui_state.audio_level*100:.1f}%", (meter_x+meter_w+10, meter_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Recording badge
    if ui_state.phase == "record":
        rec_text = "REC \u25CF"
        cv2.putText(frame, rec_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

    if ui_state.is_listening or ui_state.is_processing:
        pulse_x, pulse_y = w-60, 40; pulse_t = time.time()
        pulse_scale = 0.7 + 0.3 * abs(np.sin(pulse_t * 3))
        pulse_color = (0,165,255) if ui_state.is_processing else (0,255,0)
        pulse_text  = "PROCESSING" if ui_state.is_processing else "LISTENING"
        r = int(20 * pulse_scale)
        cv2.circle(frame, (pulse_x, pulse_y), r, pulse_color, -1)
        cv2.circle(frame, (pulse_x, pulse_y), r+2, (255,255,255), 2)
        sz = cv2.getTextSize(pulse_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, pulse_text, (pulse_x - sz[0] - 30, pulse_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pulse_color, 2, cv2.LINE_AA)

    if now < ui_state.wake_until:
        cv2.putText(frame, "WAKE WORD DETECTED!", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    if now < ui_state.flash_until and ui_state.flash_message:
        alpha = (ui_state.flash_until - now)/1.5; color_intensity = int(255*alpha)
        cv2.putText(frame, ui_state.flash_message, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,color_intensity,255), 2, cv2.LINE_AA)

    if now < ui_state.transcript_until and ui_state.last_transcript:
        text = f'"{ui_state.last_transcript}"'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        box_x, box_y = 10, h - 60; box_w, box_h = min(text_size[0]+20, w-20), 45
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x+box_w, box_y+box_h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        if text_size[0] > w - 40:
            max_chars = int((w - 60) / (text_size[0] / len(text))); text = text[:max_chars] + '..."'
        cv2.putText(frame, text, (box_x+10, box_y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return frame
