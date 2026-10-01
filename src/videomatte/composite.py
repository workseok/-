"""전경색 추정, despill, 배경 소스, 합성."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from .errors import ConfigError
from .io_video import read_background_video

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 전경색 추정
def estimate_foreground(image: np.ndarray, alpha: np.ndarray, mode: str = "ml") -> np.ndarray:
    """경계의 배경색 오염 제거. image uint8 RGB, alpha float -> float32 RGB 0..1.

    ml   : pymatting.estimate_foreground_ml (정확, CPU 에서 느림)
    fast : 알파 가중 블러로 안쪽 색을 바깥으로 번지게 하는 근사 (빠름)
    none : 원본 색 그대로
    """
    img = image.astype(np.float32) / 255.0
    if mode == "none":
        return img
    if mode == "ml":
        from pymatting import estimate_foreground_ml

        return np.clip(estimate_foreground_ml(img, alpha.astype(np.float64)), 0, 1).astype(np.float32)
    if mode == "fast":
        k = max(3, int(min(image.shape[:2]) * 0.01) | 1)
        core = np.clip((alpha - 0.5) * 2.0, 0, 1) ** 2  # 확실한 전경 픽셀만 색 근원으로 사용
        num = cv2.GaussianBlur(img * core[..., None], (0, 0), k)
        den = cv2.GaussianBlur(core, (0, 0), k)[..., None]
        est = num / np.maximum(den, 1e-4)
        w = core[..., None]
        out = np.where(den > 1e-3, w * img + (1 - w) * est, img)
        return np.clip(out, 0, 1).astype(np.float32)
    raise ConfigError(f"--fg-estimate 는 ml|fast|none: {mode!r}")


def despill(fg: np.ndarray, mode: str, strength: float = 1.0) -> np.ndarray:
    """그린/블루 스필 제거 (평균 기반): 스필 채널을 나머지 두 채널 평균 이하로 제한."""
    if mode in (None, "none") or strength <= 0:
        return fg
    ch = {"green": 1, "blue": 2}.get(mode)
    if ch is None:
        raise ConfigError(f"--despill 은 none|green|blue: {mode!r}")
    others = [i for i in range(3) if i != ch]
    limit = (fg[..., others[0]] + fg[..., others[1]]) / 2.0
    out = fg.copy()
    out[..., ch] = fg[..., ch] - strength * np.maximum(fg[..., ch] - limit, 0.0)
    return out


# ---------------------------------------------------------------- 알파 후처리
def feather_alpha(alpha: np.ndarray, px: float) -> np.ndarray:
    if px <= 0:
        return alpha
    return cv2.GaussianBlur(alpha, (0, 0), px)


# ---------------------------------------------------------------- 색 매칭
def color_match(fg: np.ndarray, alpha: np.ndarray, bg: np.ndarray, strength: float) -> np.ndarray:
    """Reinhard 방식(Lab 평균/표준편차)으로 전경 톤을 배경 쪽으로 이동. strength 0..1."""
    if strength <= 0:
        return fg
    sel = alpha > 0.5
    if sel.sum() < 100:
        return fg
    f_lab = cv2.cvtColor(fg.astype(np.float32), cv2.COLOR_RGB2LAB)
    b_lab = cv2.cvtColor(bg.astype(np.float32), cv2.COLOR_RGB2LAB)
    f_mu, f_sd = f_lab[sel].mean(0), f_lab[sel].std(0) + 1e-3
    b_mu, b_sd = b_lab.reshape(-1, 3).mean(0), b_lab.reshape(-1, 3).std(0) + 1e-3
    matched = (f_lab - f_mu) * (b_sd / f_sd) + b_mu
    out = f_lab * (1 - strength) + matched * strength
    return np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


# ---------------------------------------------------------------- 배경 소스
class Background(Protocol):
    def frames(self, w: int, h: int, fps_frac: str, start_frame: int, fps: float) -> Iterator[np.ndarray | None]:
        """float32 RGB 0..1 배경 프레임을 무한히 yield. None 이면 'blur' (전경 프레임에서 생성)."""


class ColorBg:
    def __init__(self, hexcolor: str):
        m = re.fullmatch(r"#?([0-9a-fA-F]{6})", hexcolor)
        if not m:
            raise ConfigError(f"색상 형식 오류: {hexcolor!r} (예: color:#00ff00)")
        v = m.group(1)
        self.rgb = np.array([int(v[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0

    def frames(self, w, h, fps_frac, start_frame, fps):
        frame = np.broadcast_to(self.rgb, (h, w, 3)).copy()
        while True:
            yield frame


class ImageBg:
    def __init__(self, path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ConfigError(f"배경 이미지를 읽을 수 없습니다: {path}")
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def frames(self, w, h, fps_frac, start_frame, fps):
        frame = cover_resize(self.img, w, h).astype(np.float32) / 255.0
        while True:
            yield frame


class VideoBg:
    def __init__(self, path: Path):
        self.path = path

    def frames(self, w, h, fps_frac, start_frame, fps):
        for f in read_background_video(self.path, w, h, fps_frac, start_seconds=start_frame / fps):
            yield f.astype(np.float32) / 255.0


class BlurBg:
    """원본 프레임을 강하게 블러한 배경 (프레임 단위로 호출 측이 생성)."""

    def __init__(self, strength: float = 0.03):
        self.strength = strength

    def frames(self, w, h, fps_frac, start_frame, fps):
        while True:
            yield None


def cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


def parse_background(spec: str) -> Background:
    if spec == "blur":
        return BlurBg()
    if spec.startswith("color:"):
        return ColorBg(spec[len("color:"):])
    path = Path(spec).expanduser()
    if not path.is_file():
        raise ConfigError(f"--bg 를 해석할 수 없습니다: {spec!r} (이미지/영상 경로, color:#RRGGBB, blur)")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return ImageBg(path)
    return VideoBg(path)


def blur_background(frame_rgb: np.ndarray, strength: float = 0.03) -> np.ndarray:
    k = max(3, int(min(frame_rgb.shape[:2]) * strength) | 1)
    return cv2.GaussianBlur(frame_rgb.astype(np.float32) / 255.0, (0, 0), k)


def compose(fg: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0, 1)[..., None]
    return fg * a + bg * (1 - a)
