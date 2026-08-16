from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# TensorFlow shim — must run before any mediapipe import (including
# transitive imports via hand_tracker).  See hand_tracker.py for the
# full explanation.
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

# ---------------------------------------------------------------------------
# Ensure the src/ directory is on the path so sibling modules resolve.
# ---------------------------------------------------------------------------
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from geometry_processor import GeometryProcessor  # noqa: E402
from gesture_controller import ActionType, GestureController  # noqa: E402
from hand_tracker import HandTracker  # noqa: E402
from system_actuator import SystemActuator  # noqa: E402


def main() -> None:
    tracker = HandTracker(camera_index=0, frame_size=(640, 480))
    geometry = GeometryProcessor(frame_size=(640, 480), screen_size=(1920, 1080))
    gestures = GestureController(geometry_processor=geometry)
    actuator = SystemActuator(screen_height=geometry.screen_height)

    try:
        while True:
            ok, frame = tracker.read_frame()
            if not ok:
                break

            hand_data = tracker.process_frame(frame)
            action_type, data = gestures.evaluate_state(hand_data)

            if hand_data is not None:
                actuator.execute_action(action_type, data)

            _draw_active_box(frame, geometry)
            _draw_state_overlay(frame, hand_data, action_type)

            cv2.imshow("Air Cursor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.release()
        cv2.destroyAllWindows()


def _draw_active_box(frame, geometry: GeometryProcessor) -> None:
    frame_h, frame_w = frame.shape[:2]
    x_min = int(geometry.x_min * frame_w)
    y_min = int(geometry.y_min * frame_h)
    x_max = int(geometry.x_max * frame_w)
    y_max = int(geometry.y_max * frame_h)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)


def _draw_state_overlay(frame, hand_data, action_type: ActionType) -> None:
    color = _state_color(action_type)
    frame_h, frame_w = frame.shape[:2]

    if hand_data:
        for node_id, (norm_x, norm_y) in hand_data.items():
            center = (int(norm_x * frame_w), int(norm_y * frame_h))
            cv2.circle(frame, center, 10, color, -1)
            cv2.putText(
                frame,
                str(node_id),
                (center[0] - 15, center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    label = f"Action: {action_type.name}"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = frame_w - text_size[0] - 10
    cv2.putText(frame, label, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def _state_color(action_type: ActionType) -> tuple[int, int, int]:
    if action_type == ActionType.MOVE:
        return 0, 255, 0
    if action_type == ActionType.CLICK:
        return 0, 0, 255
    if action_type == ActionType.ACTION_ENGAGED:
        return 255, 0, 0
    return 200, 200, 200


if __name__ == "__main__":
    main()
