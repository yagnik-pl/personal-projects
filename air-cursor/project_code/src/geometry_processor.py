from __future__ import annotations

from typing import Optional

import numpy as np


Point = tuple[float, float]


class GeometryProcessor:
    def __init__(
        self,
        frame_size: tuple[int, int] = (640, 480),
        screen_size: tuple[int, int] = (1920, 1080),
        margin: int = 100,
        ema_alpha: float = 0.4,
    ) -> None:
        self.frame_width, self.frame_height = frame_size
        self.screen_width, self.screen_height = screen_size
        self.margin = margin
        self.ema_alpha = ema_alpha

        if margin * 2 >= self.frame_width or margin * 2 >= self.frame_height:
            raise ValueError("Margin is too large for the configured frame size")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in the range (0, 1]")

        self.x_min = margin / self.frame_width
        self.x_max = (self.frame_width - margin) / self.frame_width
        self.y_min = margin / self.frame_height
        self.y_max = (self.frame_height - margin) / self.frame_height
        self._previous_point: Optional[Point] = None

    def map_to_screen(self, norm_x: float, norm_y: float) -> Point:
        clipped_x = float(np.clip(norm_x, self.x_min, self.x_max))
        clipped_y = float(np.clip(norm_y, self.y_min, self.y_max))

        relative_x = (clipped_x - self.x_min) / (self.x_max - self.x_min)
        relative_y = (clipped_y - self.y_min) / (self.y_max - self.y_min)

        screen_x = float(np.clip(relative_x * self.screen_width, 0, self.screen_width))
        screen_y = float(np.clip(relative_y * self.screen_height, 0, self.screen_height))
        return screen_x, screen_y

    def update_ema(self, new_x: float, new_y: float) -> Point:
        if self._previous_point is None:
            self._previous_point = (new_x, new_y)
            return new_x, new_y

        previous_x, previous_y = self._previous_point
        smoothed_x = self.ema_alpha * new_x + (1 - self.ema_alpha) * previous_x
        smoothed_y = self.ema_alpha * new_y + (1 - self.ema_alpha) * previous_y
        self._previous_point = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    def reset_ema_state(self) -> None:
        self._previous_point = None

    @staticmethod
    def distance(p1: Point, p2: Point) -> float:
        return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


if __name__ == "__main__":
    processor = GeometryProcessor()

    assert np.isclose(GeometryProcessor.distance((0, 0), (3, 4)), 5.0)
    assert processor.map_to_screen(processor.x_min, processor.y_min) == (0.0, 0.0)

    right, bottom = processor.map_to_screen(processor.x_max, processor.y_max)
    assert np.isclose(right, processor.screen_width)
    assert np.isclose(bottom, processor.screen_height)

    processor.reset_ema_state()
    assert processor.update_ema(100, 100) == (100, 100)
    second = processor.update_ema(200, 200)
    assert np.isclose(second[0], 140)
    assert np.isclose(second[1], 140)

    print("GeometryProcessor tests passed")
