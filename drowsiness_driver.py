"""
Driver Drowsiness Detection with Siren + Voice Alert
----------------------------------------------------
Detects: eyes closed too long, yawning, and head drooping (nodding).
Alert  : loud siren + voice saying "Wake up, wake up".

Install:
    pip install opencv-python mediapipe numpy pygame pyttsx3

Run:
    python drowsiness_alert.py        (press 'q' to quit, 'f' for fullscreen)
"""

import math
import subprocess
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---------------- Tunable settings ----------------
EAR_THRESHOLD = 0.21      # below this = eye closed
EYE_CLOSED_SECONDS = 1.5  # eyes closed this long = drowsy
MAR_THRESHOLD = 0.60      # above this = mouth wide open (yawn)
YAWN_SECONDS = 1.5        # mouth open this long = yawn
HEAD_DROP_RATIO = 0.42    # nose position within forehead-chin span; lower = head tilted down
HEAD_DROP_SECONDS = 1.5
CAMERA_INDEX = 0
WINDOW_NAME = "Driver Drowsiness Detection"
VOICE_TEXT = "Wake up! Wake up!"
# --------------------------------------------------

# MediaPipe FaceMesh landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]   # upper lip, lower lip, left corner, right corner
NOSE, CHIN, FOREHEAD = 1, 152, 10

# Speech runs in a separate process. pyttsx3 often goes silent when
# runAndWait() is called repeatedly from a thread, so a fresh process
# each time is far more reliable.
SPEAK_CODE = (
    "import pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.setProperty('rate', 150)\n"
    "e.setProperty('volume', 1.0)\n"
    f"e.say({VOICE_TEXT!r})\n"
    "e.runAndWait()\n"
)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(p):
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]))


def mouth_aspect_ratio(p):
    return dist(p[0], p[1]) / dist(p[2], p[3])


# ---------------- Alarm (siren + voice) ----------------
class Alarm:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        self.siren = self._make_siren()
        self.active = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _make_siren(duration=1.2, sr=44100):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 900 + 500 * np.sin(2 * np.pi * 2 * t)          # wailing 400-1400 Hz
        wave = 0.8 * np.sin(2 * np.pi * np.cumsum(freq) / sr)
        samples = (wave * 32767).astype(np.int16)
        # The mixer may open in stereo even if mono was requested, so match it
        channels = pygame.mixer.get_init()[2]
        if channels > 1:
            samples = np.ascontiguousarray(np.column_stack([samples] * channels))
        return pygame.sndarray.make_sound(samples)

    def _speak(self):
        try:
            proc = subprocess.Popen([sys.executable, "-c", SPEAK_CODE])
        except Exception as exc:
            print("Voice error:", exc)
            return
        # Wait for speech to finish, but cut it off if the alarm is cancelled
        while proc.poll() is None and self.active.is_set():
            time.sleep(0.05)
        if proc.poll() is None:
            proc.terminate()

    def _loop(self):
        while True:
            self.active.wait()
            self.siren.play()
            time.sleep(1.2)
            if self.active.is_set():
                self._speak()

    def start(self):
        self.active.set()

    def stop(self):
        self.active.clear()
        self.siren.stop()


# ---------------- Main ----------------
def main():
    alarm = Alarm()
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    eye_start = yawn_start = head_start = None

    # Full-screen window (press 'f' to toggle, 'q' to quit)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()
        reasons = []

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            pt = lambda i: (lm[i].x * w, lm[i].y * h)

            # 1) Eyes closed (detection only - nothing is drawn on the eyes)
            ear = (eye_aspect_ratio([pt(i) for i in RIGHT_EYE]) +
                   eye_aspect_ratio([pt(i) for i in LEFT_EYE])) / 2
            eye_closed = ear < EAR_THRESHOLD
            eye_color = (0, 0, 255) if eye_closed else (0, 255, 0)
            cv2.putText(frame, "EYES: CLOSED" if eye_closed else "EYES: OPEN",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)

            if eye_closed:
                eye_start = eye_start or now
                if now - eye_start >= EYE_CLOSED_SECONDS:
                    reasons.append("EYES CLOSED")
            else:
                eye_start = None

            # 2) Yawning
            mar = mouth_aspect_ratio([pt(i) for i in MOUTH])
            if mar > MAR_THRESHOLD:
                yawn_start = yawn_start or now
                if now - yawn_start >= YAWN_SECONDS:
                    reasons.append("YAWNING")
            else:
                yawn_start = None

            # 3) Head drooping (nose moves toward the forehead-chin line's upper part)
            face_h = dist(pt(FOREHEAD), pt(CHIN))
            ratio = (pt(NOSE)[1] - pt(FOREHEAD)[1]) / face_h
            if ratio < HEAD_DROP_RATIO:
                head_start = head_start or now
                if now - head_start >= HEAD_DROP_SECONDS:
                    reasons.append("HEAD DROOPING")
            else:
                head_start = None

            cv2.putText(frame, f"EAR:{ear:.2f} MAR:{mar:.2f} HEAD:{ratio:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            eye_start = yawn_start = head_start = None
            cv2.putText(frame, "No face detected", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if reasons:
            alarm.start()
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            cv2.putText(frame, "DROWSY: " + ", ".join(reasons), (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            alarm.stop()

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):                      # toggle fullscreen / normal window
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
