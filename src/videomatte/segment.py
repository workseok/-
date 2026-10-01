"""SAM 2.1 video predictor 로 첫 프레임 프롬프트를 전체 프레임에 전파."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .detect import Target
from .errors import MissingModelError, VideomatteError
from .models import SAM_SIZES, model_path

log = logging.getLogger(__name__)


class Segmenter(Protocol):
    def propagate(self, frames_dir: Path, n_frames: int, target: Target | None,
                  seed_mask: np.ndarray | None) -> np.ndarray:
        """frames_dir 의 JPEG 시퀀스에 대해 (n_frames, H, W) bool 마스크를 돌려준다.

        첫 청크는 target(박스/포인트)으로, 이후 청크는 이전 청크 마지막 마스크(seed_mask)로 시작한다.
        """


class Sam2Segmenter:
    def __init__(self, device: str, size: str = "small", models_dir=None):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:
            raise VideomatteError(
                "sam2 패키지를 import 할 수 없습니다. PyTorch 를 먼저 설치한 뒤\n"
                '  pip install "git+https://github.com/facebookresearch/sam2.git"\n'
                "를 실행하세요. 'Failed to build the SAM 2 CUDA extension' 경고는 무시해도 됩니다 "
                f"(docs/TROUBLESHOOTING.md). 원인: {e}"
            ) from e
        import torch

        if size not in SAM_SIZES:
            raise VideomatteError(f"--sam-model 은 {sorted(SAM_SIZES)} 중 하나여야 합니다.")
        ckpt = model_path(f"sam2.1_hiera_{size}", models_dir)
        self._torch = torch
        self.device = device
        self.predictor = build_sam2_video_predictor(SAM_SIZES[size], str(ckpt), device=device)

    @classmethod
    def from_predictor(cls, predictor, device: str) -> "Sam2Segmenter":
        """미리 만든 predictor 주입 (테스트/배선 검증용)."""
        import torch

        self = cls.__new__(cls)
        self._torch, self.device, self.predictor = torch, device, predictor
        return self

    def propagate(self, frames_dir, n_frames, target, seed_mask):
        torch = self._torch
        p = self.predictor
        ac = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.startswith("cuda") else _nullctx()
        out = np.zeros((n_frames,) + self._frame_hw(frames_dir), bool)
        with torch.inference_mode(), ac:
            # 프레임/상태를 CPU 로 offload 해서 VRAM 사용량을 청크 크기와 무관하게 낮춘다.
            state = p.init_state(video_path=str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
            try:
                if seed_mask is not None:
                    p.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(seed_mask.astype(bool)))
                else:
                    self._add_prompts(state, target)
                for fi, _ids, logits in p.propagate_in_video(state):
                    out[fi] = (logits > 0.0).any(dim=0)[0].cpu().numpy()
            finally:
                p.reset_state(state)
        return out

    def _add_prompts(self, state, target: Target | None) -> None:
        if target is None or (not target.boxes and not target.points):
            raise VideomatteError("첫 프레임 프롬프트(박스/포인트)가 없습니다.")
        p = self.predictor
        # 각 박스를 별도 obj_id 로 등록하고 결과는 합집합(any)으로 사용한다.
        for i, box in enumerate(target.boxes, start=1):
            kw = {}
            if i == 1 and target.points:  # 박스 + 포인트 동시 지정은 첫 객체에 결합
                kw = dict(points=np.array(target.points, np.float32), labels=np.array(target.point_labels, np.int32))
            p.add_new_points_or_box(state, frame_idx=0, obj_id=i, box=np.array(box, np.float32), **kw)
        if not target.boxes:
            p.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                    points=np.array(target.points, np.float32),
                                    labels=np.array(target.point_labels, np.int32))

    @staticmethod
    def _frame_hw(frames_dir: Path) -> tuple[int, int]:
        from . import imgio

        first = sorted(Path(frames_dir).glob("*.jpg"))[0]
        h, w = imgio.imread(first).shape[:2]
        return h, w


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
