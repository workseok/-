"""SAM 2 API 배선 검증: 가중치 없이 무작위 초기화된 tiny 모델로 실제 sam2 코드 경로를 실행한다.

마스크 품질은 의미 없음(가중치 없음). init_state/add_new_points_or_box/add_new_mask/propagate/reset_state
호출이 예외 없이 동작하고 출력 shape 이 맞는지만 확인한다.
"""

import numpy as np
import pytest

sam2 = pytest.importorskip("sam2")

from videomatte.config import Config  # noqa: E402
from videomatte.pipeline import Pipeline  # noqa: E402
from videomatte.segment import Sam2Segmenter  # noqa: E402

from conftest import FakeDetector, H, W  # noqa: E402


@pytest.fixture(scope="module")
def predictor():
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", None, device="cpu")


def test_sam2_random_init_through_pipeline(clip, tmp_path, predictor):
    seg = Sam2Segmenter.from_predictor(predictor, "cpu")
    out = tmp_path / "s.mp4"
    cfg = Config(input=clip, output=out, fg_estimate="none", chunk_size=4, max_frames=8, encoder="mpeg4",
                 work_dir=tmp_path / "w", keep_work=True, sam_max_side=160)
    Pipeline(cfg, FakeDetector(), seg).run()
    assert out.exists()
    work = next((tmp_path / "w").iterdir())
    m0, m1 = np.load(work / "masks_00000.npz")["m"], np.load(work / "masks_00001.npz")["m"]
    assert m0.shape == (4, 90, 160) and m1.shape == (4, 90, 160) and m0.dtype == bool


def test_sam2_points_prompt(clip, tmp_path, predictor):
    seg = Sam2Segmenter.from_predictor(predictor, "cpu")
    cfg = Config(input=clip, output=tmp_path / "p.mp4", fg_estimate="none", chunk_size=4, max_frames=4,
                 encoder="mpeg4", points="100,95;5,5,0", work_dir=tmp_path / "w", sam_max_side=160)
    Pipeline(cfg, FakeDetector(), seg).run()
    assert (tmp_path / "p.mp4").exists()
