from __future__ import annotations

import sys
import time
import types
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Workaround: mediapipe's __init__ unconditionally imports
# mediapipe.tasks.python which tries to import tensorflow.  If the
# installed TensorFlow is broken (e.g. protobuf / h5py / numpy ABI
# mismatch), the import crashes with a hard error that mediapipe's own
# try/except cannot catch.
#
# Fix: inject a lightweight shim for the tensorflow.tools.docs module
# into sys.modules *before* importing mediapipe.  mediapipe only needs
# ``doc_controls`` from there, so a trivial stub is enough.  This
# approach does NOT modify any installed packages and does NOT remove
# TensorFlow; it just prevents mediapipe from triggering TF's full
# import chain.
# ---------------------------------------------------------------------------
if "tensorflow" not in sys.modules:
    _tf = types.ModuleType("tensorflow")
    _tf_tools = types.ModuleType("tensorflow.tools")
    _tf_docs = types.ModuleType("tensorflow.tools.docs")

    class _FakeDocControls:
        @staticmethod
        def do_not_generate_docs(obj):
            return obj

        @staticmethod
        def do_not_doc_inheritable(obj):
            return obj

    _tf_docs.doc_controls = _FakeDocControls
    _tf.tools = _tf_tools

    sys.modules.setdefault("tensorflow", _tf)
    sys.modules.setdefault("tensorflow.tools", _tf_tools)
    sys.modules.setdefault("tensorflow.tools.docs", _tf_docs)
    sys.modules.setdefault("tensorflow.tools.docs.doc_controls", _tf_docs)

import mediapipe as mp  # noqa: E402  — must come after the shim

HandData = dict[int, tuple[float, float]]


class HandTracker:
    """MediaPipe hand-tracking wrapper (Nodes 0, 4, 8, 12)."""

    REQUIRED_NODES = (0, 4, 8, 12)

    def __init__(
        self,
        camera_index: int = 0,
        frame_size: tuple[int, int] = (640, 480),
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
    ) -> None:
        self.frame_width, self.frame_height = frame_size
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self._previous_time: Optional[float] = None

    def read_frame(self) -> tuple[bool, object]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()

    def process_frame(self, frame) -> Optional[HandData]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return None

        rgb_frame = frame[:, :, ::-1].copy()
        results = self.hands.process(rgb_frame)
        self._draw_fps(frame)

        if not results.multi_hand_landmarks:
            self._draw_no_hand_warning(frame)
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
        )

        return {
            node_id: (
                hand_landmarks.landmark[node_id].x,
                hand_landmarks.landmark[node_id].y,
            )
            for node_id in self.REQUIRED_NODES
        }

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_fps(self, frame) -> None:
        current_time = time.time()
        fps = 0
        if self._previous_time is not None:
            delta = current_time - self._previous_time
            fps = 1 / delta if delta > 0 else 0
        self._previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    @staticmethod
    def _draw_no_hand_warning(frame) -> None:
        overlay = frame.copy()
        overlay[:] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        text = "WARNING: NO HAND DETECTED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2
        text_width, text_height = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (frame.shape[1] - text_width) // 2
        y = (frame.shape[0] + text_height) // 2
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), thickness)


if __name__ == "__main__":
    tracker = HandTracker()

    while True:
        ok, frame = tracker.read_frame()
        if not ok:
            break

        nodes = tracker.process_frame(frame)
        print("Nodes:", nodes)

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.release()
    cv2.destroyAllWindows()
