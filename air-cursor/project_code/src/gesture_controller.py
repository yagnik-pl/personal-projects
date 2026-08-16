from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Optional

from geometry_processor import GeometryProcessor, Point


HandData = dict[int, Point]


class ActionType(Enum):
    MOVE = "MOVE"
    CLICK = "CLICK"
    ACTION_ENGAGED = "ACTION_ENGAGED"
    NONE = "NONE"


class GestureController:
    CLICK_THRESHOLD = 0.04
    ACTION_THRESHOLD = 0.03

    def __init__(self, geometry_processor: Optional[GeometryProcessor] = None) -> None:
        self.geom_processor = geometry_processor or GeometryProcessor()

    def evaluate_state(self, hand_data: Optional[HandData]) -> tuple[ActionType, list[float]]:
        if not self._has_required_nodes(hand_data):
            self.geom_processor.reset_ema_state()
            return ActionType.NONE, []

        assert hand_data is not None
        thumb_tip = hand_data[4]
        index_tip = hand_data[8]
        middle_tip = hand_data[12]

        mapped_x, mapped_y = self.geom_processor.map_to_screen(*index_tip)
        smoothed_x, smoothed_y = self.geom_processor.update_ema(mapped_x, mapped_y)

        thumb_index_distance = GeometryProcessor.distance(thumb_tip, index_tip)
        thumb_middle_distance = GeometryProcessor.distance(thumb_tip, middle_tip)

        if thumb_index_distance < self.ACTION_THRESHOLD:
            return ActionType.ACTION_ENGAGED, [smoothed_x, smoothed_y]
        if thumb_middle_distance < self.CLICK_THRESHOLD:
            return ActionType.CLICK, [smoothed_x, smoothed_y]
        return ActionType.MOVE, [smoothed_x, smoothed_y]

    @staticmethod
    def _has_required_nodes(hand_data: Optional[HandData]) -> bool:
        if hand_data is None:
            return False

        for node_id in (4, 8, 12):
            point = hand_data.get(node_id)
            if point is None or len(point) != 2:
                return False
            if not all(isfinite(value) for value in point):
                return False
        return True


if __name__ == "__main__":
    controller = GestureController()

    action, _ = controller.evaluate_state(None)
    assert action == ActionType.NONE

    action, data = controller.evaluate_state({
        4: (0.1, 0.1),
        8: (0.5, 0.5),
        12: (0.5, 0.1),
    })
    print("Move action:", action, data)
    assert action == ActionType.MOVE

    action, data = controller.evaluate_state({
        4: (0.5, 0.5),
        8: (0.5, 0.51),
        12: (0.1, 0.1),
    })
    print("Action engaged:", action, data)
    assert action == ActionType.ACTION_ENGAGED

    action, data = controller.evaluate_state({
        4: (0.5, 0.5),
        8: (0.1, 0.1),
        12: (0.5, 0.53),
    })
    print("Click action:", action, data)
    assert action == ActionType.CLICK

    print("GestureController tests passed")
