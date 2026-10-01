"""첫 프레임에서 대상 지정: Grounding DINO(텍스트) 또는 수동 박스/포인트."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .errors import ConfigError, DetectionError, MissingModelError
from .models import MODELS, model_path

log = logging.getLogger(__name__)


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1,y1,x2,y2 (원본 프레임 좌표)
    score: float = 1.0
    label: str = ""

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


@dataclass
class Target:
    """SAM 2 에 줄 첫 프레임 프롬프트 (원본 좌표계)."""

    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    points: list[tuple[float, float]] = field(default_factory=list)
    point_labels: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"boxes": self.boxes, "points": self.points, "point_labels": self.point_labels}

    @classmethod
    def from_json(cls, d: dict) -> "Target":
        return cls([tuple(b) for b in d["boxes"]], [tuple(p) for p in d["points"]], list(d["point_labels"]))

    def scaled(self, sx: float, sy: float) -> "Target":
        return Target(
            [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in self.boxes],
            [(p[0] * sx, p[1] * sy) for p in self.points],
            list(self.point_labels),
        )


def parse_box(text: str) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
    except ValueError:
        raise ConfigError(f"--box 형식 오류: {text!r} (예: 100,50,600,700)") from None
    if x2 <= x1 or y2 <= y1:
        raise ConfigError(f"--box 는 x1<x2, y1<y2 여야 합니다: {text!r}")
    return x1, y1, x2, y2


def parse_points(text: str) -> tuple[list[tuple[float, float]], list[int]]:
    """'x,y;x,y' 또는 'x,y,label;...' (label 1=전경, 0=배경, 생략 시 1)."""
    pts, labels = [], []
    for item in filter(None, (s.strip() for s in text.split(";"))):
        vals = item.split(",")
        try:
            if len(vals) not in (2, 3):
                raise ValueError
            pts.append((float(vals[0]), float(vals[1])))
            labels.append(int(vals[2]) if len(vals) == 3 else 1)
        except ValueError:
            raise ConfigError(f"--points 형식 오류: {item!r} (예: 320,240;400,300)") from None
    if not pts:
        raise ConfigError("--points 가 비어 있습니다.")
    return pts, labels


def select_detections(dets: list[Detection], mode: str) -> list[Detection]:
    """largest | all | <index>. index 는 면적 내림차순 정렬 기준(0 = 가장 큼)."""
    if not dets:
        return []
    ranked = sorted(dets, key=lambda d: d.area, reverse=True)
    if mode == "largest":
        return ranked[:1]
    if mode == "all":
        return ranked
    try:
        idx = int(mode)
    except ValueError:
        raise ConfigError(f"--select 는 largest|all|<정수> 여야 합니다: {mode!r}") from None
    if not 0 <= idx < len(ranked):
        raise DetectionError(f"--select {idx}: 검출된 객체는 {len(ranked)}개입니다 (0..{len(ranked) - 1}).")
    return [ranked[idx]]


class GroundingDinoDetector:
    """HuggingFace transformers 의 Grounding DINO 구현 사용 (models/ 의 로컬 스냅샷만 사용)."""

    def __init__(self, device: str, models_dir=None, variant: str = "tiny",
                 box_threshold: float = 0.35, text_threshold: float = 0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        path = model_path(f"grounding-dino-{variant}", models_dir)
        if not MODELS[f"grounding-dino-{variant}"].license_verified:
            log.warning("Grounding DINO 가중치 라이선스는 아직 미확인입니다 (THIRD_PARTY_LICENSES.md).")
        self._torch = torch
        self.device = device
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(device).eval()

    def detect(self, frame_rgb: np.ndarray, prompt: str) -> list[Detection]:
        from PIL import Image

        text = prompt.strip().lower()
        if not text.endswith("."):
            text += "."
        image = Image.fromarray(frame_rgb)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = res.get("text_labels", res.get("labels", []))
        return [
            Detection(tuple(float(v) for v in b.tolist()), float(s), str(l))
            for b, s, l in zip(res["boxes"], res["scores"], labels)
        ]
