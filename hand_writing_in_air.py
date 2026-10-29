"""
Air Writing -> Text -> Speech
-----------------------------
Write in the air with your index finger in front of a webcam.
The drawing is recognised as text (OCR) and read aloud (TTS).

Gestures
    Index finger only up ............ DRAW
    Index + middle fingers up ....... PAUSE (move without drawing)
                                      Hover over a colour box in PAUSE
                                      mode to pick that colour.

Keys
    1-7 . pick colour
    u ... undo last stroke
    y ... redo
    s ... save drawing as PNG (white background, in your chosen colours)
    r ... recognise the drawing and speak it
    c ... clear the canvas (can be undone with u)
    q ... quit

Install
    pip install opencv-python mediapipe easyocr pyttsx3 numpy
"""

import threading
import time
from datetime import datetime

import cv2
import easyocr
import mediapipe as mp
import numpy as np
import pyttsx3

# ----------------------------- Configuration -----------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 540
BRUSH_THICKNESS = 10
SMOOTHING = 0.5          # 0 = no smoothing, closer to 1 = heavier smoothing
OCR_LANGS = ["en"]       # e.g. ["mr", "en"] for Marathi (Devanagari)

# Palette (name, BGR colour). Keys 1..7 select them.
COLORS = [
    ("Blue",   (255, 80, 0)),
    ("Red",    (0, 0, 230)),
    ("Green",  (0, 170, 0)),
    ("Orange", (0, 140, 255)),
    ("Purple", (200, 0, 160)),
    ("Cyan",   (220, 200, 0)),
    ("Black",  (0, 0, 0)),
]
BOX = 46                 # palette box size
BOX_GAP = 10
PALETTE_X, PALETTE_Y = 15, 75

# ----------------------------- Setup -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

print("Loading OCR model (first run downloads it)...")
reader = easyocr.Reader(OCR_LANGS, gpu=False)


def speak(text: str) -> None:
    """Speak text in a background thread so the video loop never freezes."""
    def _run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=_run, daemon=True).start()


def fingers_up(lm) -> dict:
    """Return which fingers are extended (tip above the PIP joint)."""
    return {
        "index": lm[8].y < lm[6].y,
        "middle": lm[12].y < lm[10].y,
        "ring": lm[16].y < lm[14].y,
        "pinky": lm[20].y < lm[18].y,
    }


# ----------------------------- Stroke model -----------------------------
# A stroke is {"color": (b, g, r), "pts": [(x, y), ...]}.
# Undo/redo simply move whole strokes between two stacks, and the canvas
# is re-rendered from the stroke list.

def render(strokes, current):
    """Return (color_canvas BGR, mask uint8) built from all strokes."""
    color = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    for s in strokes + ([current] if current else []):
        pts = s["pts"]
        if not pts:
            continue
        if len(pts) == 1:
            cv2.circle(color, pts[0], BRUSH_THICKNESS // 2, s["color"], -1)
            cv2.circle(mask, pts[0], BRUSH_THICKNESS // 2, 255, -1)
        else:
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(color, [arr], False, s["color"], BRUSH_THICKNESS,
                          cv2.LINE_AA)
            cv2.polylines(mask, [arr], False, 255, BRUSH_THICKNESS,
                          cv2.LINE_AA)
    return color, mask


def save_drawing(color_canvas, mask) -> str:
    """Save the strokes in their colours on a white background."""
    if not mask.any():
        return ""
    ys, xs = np.where(mask > 0)
    pad = 30
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, FRAME_W)
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, FRAME_H)

    white = np.full_like(color_canvas, 255)
    m = mask > 0
    white[m] = color_canvas[m]
    out = white[y1:y2, x1:x2]

    name = datetime.now().strftime("air_writing_%Y%m%d_%H%M%S.png")
    cv2.imwrite(name, out)
    return name


def recognise(canvas: np.ndarray) -> str:
    """Crop the strokes, convert to black-on-white, and run OCR."""
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0:
        return ""
    pad = 20
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, canvas.shape[1])
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, canvas.shape[0])
    crop = canvas[y1:y2, x1:x2]

    # 1. join small gaps between strokes
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 2. normalise height to 96 px (keep aspect ratio)
    h, w = crop.shape
    new_w = max(int(w * 96 / h), 1)
    crop = cv2.resize(crop, (new_w, 96), interpolation=cv2.INTER_AREA)

    # 3. black text on white, binary, with white border
    img = cv2.bitwise_not(crop)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.copyMakeBorder(img, 30, 30, 30, 30,
                             cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite("debug_ocr.png", img)

    H, W = img.shape
    # 4. try recogniser only (skips the detector), then fall back
    try:
        out = reader.recognize(img, horizontal_list=[[0, W, 0, H]],
                               free_list=[], detail=0)
    except Exception:
        out = []
    if not "".join(out).strip():
        out = reader.readtext(img, detail=0, paragraph=True)
    return " ".join(out).strip()


def palette_box(i):
    x = PALETTE_X + i * (BOX + BOX_GAP)
    return x, PALETTE_Y, x + BOX, PALETTE_Y + BOX


def palette_hit(pt):
    """Return the palette index under pt, or None."""
    for i in range(len(COLORS)):
        x1, y1, x2, y2 = palette_box(i)
        if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
            return i
    return None


def draw_palette(frame, selected):
    for i, (_, bgr) in enumerate(COLORS):
        x1, y1, x2, y2 = palette_box(i)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
        border = (255, 255, 255) if i == selected else (120, 120, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border,
                      4 if i == selected else 1)
        cv2.putText(frame, str(i + 1), (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main() -> None:
    cap = cv2.VideoCapture(CAM_INDEX)
    cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Air Writing", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    strokes = []          # finished strokes
    redo_stack = []       # undone strokes
    current = None        # stroke being drawn right now
    clear_backup = None   # lets 'c' be undone
    color_idx = 0

    smooth_pt = None
    result_text = ""
    mode = "NO HAND"
    status_msg, status_until = "", 0.0

    def finish_stroke():
        nonlocal current
        if current and current["pts"]:
            strokes.append(current)
            redo_stack.clear()
        current = None

    def notify(msg, secs=2.0):
        nonlocal status_msg, status_until
        status_msg, status_until = msg, time.time() + secs

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                    # mirror view
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            tip = np.array([lm[8].x * FRAME_W, lm[8].y * FRAME_H])
            smooth_pt = tip if smooth_pt is None else (
                SMOOTHING * smooth_pt + (1 - SMOOTHING) * tip
            )
            pt = (int(smooth_pt[0]), int(smooth_pt[1]))

            f = fingers_up(lm)
            if f["index"] and not f["middle"]:
                mode = "DRAW"
                if current is None:
                    current = {"color": COLORS[color_idx][1], "pts": []}
                current["pts"].append(pt)
                cv2.circle(frame, pt, 8, COLORS[color_idx][1], -1)
            else:
                mode = "PAUSE"
                finish_stroke()                       # lift the pen
                cv2.circle(frame, pt, 8, (0, 165, 255), 2)
                hit = palette_hit(pt)                 # hover to pick colour
                if hit is not None and hit != color_idx:
                    color_idx = hit
                    notify(f"Colour: {COLORS[color_idx][0]}")
        else:
            mode, smooth_pt = "NO HAND", None
            finish_stroke()

        # Render strokes (in colour) over the video
        color_canvas, mask = render(strokes, current)
        m = mask > 0
        blended = cv2.addWeighted(color_canvas, 0.9, frame, 0.1, 0)
        frame[m] = blended[m]

        # HUD
        draw_palette(frame, color_idx)
        cv2.putText(frame, f"Mode: {mode}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame,
                    "1-7: colour | u: undo | y: redo | s: save | "
                    "r: recognise | c: clear | q: quit",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        cv2.putText(frame, f"Text: {result_text}", (15, FRAME_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
        if time.time() < status_until:
            cv2.putText(frame, status_msg, (15, FRAME_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Air Writing", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            finish_stroke()
            if strokes:
                clear_backup = list(strokes)
                strokes.clear()
                redo_stack.clear()
                redo_stack.extend(reversed(clear_backup))  # 'y' restores
                notify("Cleared (press y to restore)")
            result_text = ""

        elif key == ord("u"):
            finish_stroke()
            if strokes:
                redo_stack.append(strokes.pop())
                notify("Undo")
            else:
                notify("Nothing to undo")

        elif key == ord("y"):
            finish_stroke_keep = current is None
            if redo_stack and finish_stroke_keep:
                strokes.append(redo_stack.pop())
                notify("Redo")
            else:
                notify("Nothing to redo")

        elif key == ord("s"):
            finish_stroke()
            color_canvas, mask = render(strokes, None)
            name = save_drawing(color_canvas, mask)
            if name:
                print("Saved:", name)
                notify(f"Saved {name}", 3.0)
            else:
                notify("Nothing to save")

        elif key == ord("r"):
            finish_stroke()
            _, mask = render(strokes, None)
            result_text = recognise(mask)
            print("Recognised:", result_text)
            if result_text:
                speak(result_text)

        elif ord("1") <= key <= ord("0") + len(COLORS):
            color_idx = key - ord("1")
            notify(f"Colour: {COLORS[color_idx][0]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
