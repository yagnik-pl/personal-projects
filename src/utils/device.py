"""
Device resolution and hardware telemetry utilities.
"""
import logging
from typing import Any, Dict, Optional, Union
import torch

logger = logging.getLogger("AdaptiveRetriever.Device")


def resolve_device(device_str: Optional[str] = "auto") -> torch.device:
    """
    Resolves the execution device with safe fallback and telemetry logging.

    Args:
        device_str: "auto", "cuda", "cuda:0", "mps", "cpu", or None.

    Returns:
        torch.device instance.
    """
    if device_str is None or device_str.lower() == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif device_str.lower().startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device_str)
        else:
            logger.warning(f"CUDA device '{device_str}' requested but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif device_str.lower() == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logger.warning("MPS device requested but MPS is not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif device_str.lower() == "cpu":
        device = torch.device("cpu")
    else:
        logger.warning(f"Unknown device specification '{device_str}'. Defaulting to CPU.")
        device = torch.device("cpu")

    return device


def get_device_info(device: torch.device) -> Dict[str, Any]:
    """
    Returns hardware telemetry details for the resolved device.
    """
    info: Dict[str, Any] = {
        "type": device.type,
        "index": device.index,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_idx)
        info["device_name"] = props.name
        info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["multi_processor_count"] = props.multi_processor_count
        info["major_capability"] = props.major
        info["minor_capability"] = props.minor
    elif device.type == "cpu":
        info["device_name"] = "Host CPU"
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon MPS"

    return info
