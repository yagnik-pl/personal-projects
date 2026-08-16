from __future__ import annotations

import time
from typing import Optional

import pyautogui

try:
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
except Exception:
    AudioUtilities = None
    CLSCTX_ALL = None
    IAudioEndpointVolume = None
    POINTER = None
    cast = None


class SystemActuator:
    CLICK_DEBOUNCE_SECONDS = 0.4
    VOLUME_STEP_INTERVAL_SECONDS = 0.12
    VOLUME_DEADZONE_PX = 12

    def __init__(self, screen_height: int = 1080) -> None:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        self.screen_height = screen_height
        self.last_click_time = 0.0
        self.last_volume_time = 0.0
        self._last_action_y: Optional[float] = None
        self.volume = None
        self.min_volume = None
        self.max_volume = None
        self._init_volume_control()

    def execute_action(self, action_type, data=None) -> None:
        action = action_type.value if hasattr(action_type, "value") else str(action_type)

        try:
            if action == "MOVE":
                self._move(data)
            elif action == "CLICK":
                self._click()
            elif action == "ACTION_ENGAGED":
                self._adjust_volume(data)
            else:
                self._last_action_y = None
        except Exception as exc:
            print(f"Error executing action {action}: {exc}")

    def _move(self, data) -> None:
        self._last_action_y = None
        point = self._extract_point(data)
        if point is not None:
            pyautogui.moveTo(point[0], point[1])

    def _click(self) -> None:
        self._last_action_y = None
        now = time.time()
        if now - self.last_click_time >= self.CLICK_DEBOUNCE_SECONDS:
            pyautogui.click()
            self.last_click_time = now

    def _adjust_volume(self, data) -> None:
        point = self._extract_point(data)
        if point is None:
            return

        y = point[1]
        if self.volume is not None:
            # Map screen Y to volume: top of screen = max volume, bottom = min
            normalized_y = max(0.0, min(1.0, y / self.screen_height))
            volume_scalar = 1.0 - normalized_y  # invert: hand up = louder
            try:
                self.volume.SetMasterVolumeLevelScalar(volume_scalar, None)
            except Exception as exc:
                print(f"Volume set error: {exc}")
            self._last_action_y = y
            return

        # Fallback: media keys
        now = time.time()
        if self._last_action_y is None:
            self._last_action_y = y
            return
        if now - self.last_volume_time < self.VOLUME_STEP_INTERVAL_SECONDS:
            return

        delta_y = y - self._last_action_y
        if delta_y < -self.VOLUME_DEADZONE_PX:
            pyautogui.press("volumeup")
            self.last_volume_time = now
        elif delta_y > self.VOLUME_DEADZONE_PX:
            pyautogui.press("volumedown")
            self.last_volume_time = now
        self._last_action_y = y


    def _init_volume_control(self) -> None:
        if not all((AudioUtilities, CLSCTX_ALL, IAudioEndpointVolume, POINTER, cast)):
            return

        try:
            speakers = AudioUtilities.GetSpeakers()

            # Try the standard Activate path first (older pycaw / raw IMMDevice)
            interface = None
            if hasattr(speakers, "Activate"):
                try:
                    interface = speakers.Activate(
                        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                    )
                except Exception:
                    pass

            # Newer pycaw wraps the device; try EndpointVolume property
            if interface is None and hasattr(speakers, "EndpointVolume"):
                self.volume = speakers.EndpointVolume
                self.min_volume, self.max_volume, _ = self.volume.GetVolumeRange()
                print(f"Volume control initialized (EndpointVolume, range: {self.min_volume:.1f} to {self.max_volume:.1f} dB)")
                return

            # Try accessing the underlying COM device object
            if interface is None and hasattr(speakers, "_dev"):
                interface = speakers._dev.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )

            if interface is not None:
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.min_volume, self.max_volume, _ = self.volume.GetVolumeRange()
        except Exception as exc:
            print(f"pycaw volume control unavailable, using media keys fallback: {exc}")
            self.volume = None
            self.min_volume = None
            self.max_volume = None

    @staticmethod
    def _extract_point(data) -> Optional[tuple[float, float]]:
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            first, second = data[0], data[1]
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                return float(first), float(second)
        return None


if __name__ == "__main__":
    actuator = SystemActuator()

    print("Testing MOVE to (100, 100)")
    actuator.execute_action("MOVE", (100, 100))
    time.sleep(1)

    print("Testing CLICK debounce")
    actuator.execute_action("CLICK")
    actuator.execute_action("CLICK")
    time.sleep(0.5)
    actuator.execute_action("CLICK")

    print("Testing ACTION_ENGAGED volume mapping")
    actuator.execute_action("ACTION_ENGAGED", (960, 300))
    actuator.execute_action("ACTION_ENGAGED", (960, 700))
    print("Done testing")
