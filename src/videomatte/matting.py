"""트라이맵 생성, 알파 정밀화(none / vitmatte / birefnet), 시간적 안정화."""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

from .models import MODELS, model_path

log = logging.getLogger(__name__)


def make_trimap(mask: np.ndarray, erode_px: int, dilate_px: int | None = None) -> np.ndarray:
    """mask(bool/0-1 float) -> 트라이맵 uint8 (0=배경, 128=unknown, 255=전경)."""
    dilate_px = erode_px if dilate_px is None else dilate_px
    m = (mask > 0.5).astype(np.uint8)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2) if erode_px > 0 else None
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2) if dilate_px > 0 else None
    fg = cv2.erode(m, ek) if ek is not None else m
    bg_complement = cv2.dilate(m, dk) if dk is not None else m
    tri = np.full(m.shape, 128, np.uint8)
    tri[fg == 1] = 255
    tri[bg_complement == 0] = 0
    return tri


def auto_trimap_width(h: int, w: int) -> int:
    """해상도에 비례한 기본 unknown 밴드 폭(px): 짧은 변의 약 2%, 최소 4."""
    return max(4, int(round(min(h, w) * 0.02)))


class Refiner(Protocol):
    def refine(self, image_rgb: np.ndarray, mask: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """float32 알파 (H,W) 0..1"""


class NoRefiner:
    """정밀화 없음: SAM 마스크를 unknown 밴드에서만 가볍게 블러해 소프트 알파로 만든다."""

    def refine(self, image_rgb, mask, trimap):
        m = (mask > 0.5).astype(np.float32)
        sigma = max(0.8, (int((trimap == 128).sum()) / max(1, trimap.shape[0] * trimap.shape[1])) ** 0.5 * 6)
        soft = cv2.GaussianBlur(m, (0, 0), sigma)
        alpha = np.where(trimap == 255, 1.0, np.where(trimap == 0, 0.0, soft)).astype(np.float32)
        return alpha


class ViTMatteRefiner:
    def __init__(self, device: str, models_dir=None, variant: str = "small"):
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        name = f"vitmatte-{variant}"
        path = model_path(name, models_dir)
        if not MODELS[name].license_verified:
            log.warning("ViTMatte 가중치 라이선스(학습 데이터 포함)는 미확인입니다 (THIRD_PARTY_LICENSES.md).")
        self._torch, self.device = torch, device
        self.processor = VitMatteImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = VitMatteForImageMatting.from_pretrained(path, local_files_only=True).to(device).eval()

    def refine(self, image_rgb, mask, trimap):
        from PIL import Image

        h, w = trimap.shape
        inputs = self.processor(images=Image.fromarray(image_rgb), trimaps=Image.fromarray(trimap), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.inference_mode():
            alpha = self.model(**inputs).alphas[0, 0, :h, :w].float().cpu().numpy()
        # trimap 이 확정한 영역은 모델 출력과 무관하게 고정
        alpha = np.where(trimap == 255, 1.0, np.where(trimap == 0, 0.0, alpha))
        return np.clip(alpha, 0, 1).astype(np.float32)


class BiRefNetRefiner:
    """BiRefNet 은 트라이맵 입력이 없으므로 unknown 밴드 안에서만 출력을 채택한다."""

    def __init__(self, device: str, models_dir=None, size: int = 1024):
        import torch
        from transformers import AutoModelForImageSegmentation

        path = model_path("birefnet", models_dir)
        if not MODELS["birefnet"].license_verified:
            log.warning("BiRefNet 가중치 라이선스는 HF 카드 미확인입니다 (THIRD_PARTY_LICENSES.md).")
        self._torch, self.device, self.size = torch, device, size
        # trust_remote_code: models/birefnet 안에 받아둔 모델 코드를 실행한다 (원격 호출 아님).
        self.model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model.to(device).eval()
        self._mean = np.array([0.485, 0.456, 0.406], np.float32)
        self._std = np.array([0.229, 0.224, 0.225], np.float32)

    def refine(self, image_rgb, mask, trimap):
        torch = self._torch
        h, w = trimap.shape
        x = cv2.resize(image_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = torch.from_numpy(((x - self._mean) / self._std).transpose(2, 0, 1)[None]).to(self.device)
        with torch.inference_mode():
            pred = self.model(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
        a = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.where(trimap == 255, 1.0, np.where(trimap == 0, 0.0, a)).astype(np.float32)


def build_refiner(name: str, device: str, models_dir=None, variant: str = "small") -> Refiner:
    if name == "none":
        return NoRefiner()
    if name == "vitmatte":
        return ViTMatteRefiner(device, models_dir, variant)
    if name == "birefnet":
        return BiRefNetRefiner(device, models_dir)
    raise ValueError(f"unknown refiner: {name}")


class TemporalSmoother:
    """알파 EMA. 움직임이 큰 픽셀은 잔상(ghosting)을 막기 위해 가중치를 줄인다.

    out = prev + (cur - prev) * (1 - s * exp(-(|cur-prev|/tau)^2))
    s=0 이면 비활성, s 가 클수록 강한 평활. 움직임이 크면(|diff|>>tau) 현재 값을 그대로 따른다.
    """

    def __init__(self, strength: float = 0.0, tau: float = 0.3):
        if not 0.0 <= strength < 1.0:
            raise ValueError("temporal-smooth 는 0 이상 1 미만이어야 합니다.")
        self.s, self.tau = strength, tau
        self.prev: np.ndarray | None = None

    def reset(self, prev: np.ndarray | None = None) -> None:
        self.prev = prev

    def __call__(self, alpha: np.ndarray) -> np.ndarray:
        if self.s <= 0:
            return alpha
        if self.prev is None or self.prev.shape != alpha.shape:
            self.prev = alpha
            return alpha
        diff = alpha - self.prev
        w = 1.0 - self.s * np.exp(-((diff / self.tau) ** 2))
        out = (self.prev + diff * w).astype(np.float32)
        self.prev = out
        return out
