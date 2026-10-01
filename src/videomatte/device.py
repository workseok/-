"""디바이스 감지: CUDA -> MPS(Apple Silicon) -> CPU."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def configure_environment() -> None:
    """torch import 전에 호출해야 하는 환경 설정.

    MPS 에서 아직 구현되지 않은 연산은 CPU 로 폴백하도록 한다.
    이미 사용자가 값을 지정했다면 덮어쓰지 않는다.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def select_device(requested: str = "auto") -> str:
    """사용할 디바이스 문자열을 돌려준다. 요청한 디바이스를 쓸 수 없으면 예외."""
    import torch

    requested = (requested or "auto").lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if requested == "auto":
        device = "cuda" if cuda_ok else "mps" if mps_ok else "cpu"
    elif requested.split(":")[0] == "cuda":
        if not cuda_ok:
            raise RuntimeError("--device cuda 를 요청했지만 CUDA 를 사용할 수 없습니다 (CPU 전용 PyTorch 이거나 드라이버 문제).")
        device = requested
    elif requested == "mps":
        if not mps_ok:
            raise RuntimeError("--device mps 를 요청했지만 MPS 를 사용할 수 없습니다 (Apple Silicon + macOS 12.3 이상 필요).")
        device = "mps"
    elif requested == "cpu":
        device = "cpu"
    else:
        raise RuntimeError(f"알 수 없는 디바이스: {requested!r} (auto|cuda|mps|cpu)")
    log.info("device: %s", device)
    return device


def autocast_dtype(device: str):
    """CUDA 에서만 bfloat16 autocast 를 쓴다. MPS/CPU 는 float32 (안정성 우선)."""
    import torch

    if device.startswith("cuda"):
        return torch.bfloat16
    return None


def describe() -> str:
    import torch

    parts = [f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        parts.append(f"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        parts.append("MPS available")
    parts.append(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    return ", ".join(parts)
