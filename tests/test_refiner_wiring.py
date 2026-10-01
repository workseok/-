"""ViTMatte 배선 검증: transformers 의 무작위 초기화 tiny 모델을 models/ 구조로 저장해 로더/전처리/출력 크롭을 확인.
(가중치 품질은 검증 대상 아님)"""

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from videomatte import matting  # noqa: E402


def test_vitmatte_random_tiny(tmp_path):
    from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor

    bb = VitDetConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=64, pretrain_image_size=64,
                      patch_size=16, window_size=2, window_block_indices=[0], residual_block_indices=[1],
                      out_features=["stage2"], use_relative_position_embeddings=True, num_channels=4)
    cfg = VitMatteConfig(backbone_config=bb, hidden_size=32, convstream_hidden_sizes=[8, 16, 32],
                         fusion_hidden_sizes=[32, 16, 8, 4])
    d = tmp_path / "vitmatte-small"
    VitMatteForImageMatting(cfg).save_pretrained(d)
    VitMatteImageProcessor().save_pretrained(d)

    r = matting.ViTMatteRefiner("cpu", models_dir=tmp_path, variant="small")
    img = np.random.RandomState(0).randint(0, 255, (50, 70, 3), np.uint8)  # 32 의 배수가 아닌 크기
    m = np.zeros((50, 70), np.float32)
    m[10:40, 15:55] = 1
    tri = matting.make_trimap(m, 4)
    a = r.refine(img, m, tri)
    assert a.shape == (50, 70) and a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1
    assert (a[tri == 255] == 1).all() and (a[tri == 0] == 0).all()
