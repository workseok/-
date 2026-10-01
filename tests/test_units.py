import numpy as np
import pytest

from videomatte import composite as comp
from videomatte import matting
from videomatte.config import load_config
from videomatte.detect import Detection, parse_box, parse_points, select_detections
from videomatte.errors import ConfigError, DetectionError


def test_parse_box_points():
    assert parse_box("1,2,30,40") == (1, 2, 30, 40)
    with pytest.raises(ConfigError):
        parse_box("1,2,0,40")
    with pytest.raises(ConfigError):
        parse_box("a,b")
    assert parse_points("1,2;3,4,0") == ([(1, 2), (3, 4)], [1, 0])
    with pytest.raises(ConfigError):
        parse_points("1;2")


def test_select_detections():
    dets = [Detection((0, 0, 10, 10)), Detection((0, 0, 50, 50)), Detection((0, 0, 30, 30))]
    assert select_detections(dets, "largest")[0].box[2] == 50
    assert len(select_detections(dets, "all")) == 3
    assert select_detections(dets, "1")[0].box[2] == 30
    assert select_detections([], "largest") == []
    with pytest.raises(DetectionError):
        select_detections(dets, "7")


def test_trimap_structure():
    m = np.zeros((100, 100), bool)
    m[30:70, 30:70] = True
    t = matting.make_trimap(m, 5)
    assert set(np.unique(t)) == {0, 128, 255}
    assert t[50, 50] == 255 and t[0, 0] == 0 and t[30, 50] == 128 and t[26, 50] == 128 and t[20, 50] == 0


def test_no_refiner_alpha_range_and_locked_regions():
    m = np.zeros((100, 100), np.float32)
    m[30:70, 30:70] = 1
    t = matting.make_trimap(m, 5)
    a = matting.NoRefiner().refine(np.zeros((100, 100, 3), np.uint8), m, t)
    assert a.dtype == np.float32 and 0 <= a.min() and a.max() <= 1
    assert (a[t == 255] == 1).all() and (a[t == 0] == 0).all()


def test_temporal_smoother_reduces_flicker_but_follows_motion():
    s = matting.TemporalSmoother(0.7)
    base = np.full((8, 8), 0.5, np.float32)
    s(base)
    flick = s(base + 0.05)  # 작은 변화 -> 억제
    assert abs(flick.mean() - 0.5) < 0.03
    jump = s(np.ones((8, 8), np.float32))  # 큰 변화(움직임) -> 거의 즉시 추종
    assert jump.mean() > 0.9
    assert (matting.TemporalSmoother(0.0)(base) is base)
    with pytest.raises(ValueError):
        matting.TemporalSmoother(1.0)


def test_despill_green_reduces_green_only_when_excess():
    fg = np.array([[[0.2, 0.9, 0.2], [0.8, 0.5, 0.7]]], np.float32)
    out = comp.despill(fg, "green")
    assert out[0, 0, 1] == pytest.approx(0.2)  # 초과분 제거
    assert out[0, 1, 1] == pytest.approx(0.5)  # 초과 없음 -> 유지
    assert comp.despill(fg, "none") is fg
    blue = comp.despill(np.array([[[0.2, 0.2, 0.9]]], np.float32), "blue")
    assert blue[0, 0, 2] == pytest.approx(0.2)


def test_compose_and_feather_and_color_bg():
    fg = np.ones((4, 4, 3), np.float32)
    bg = np.zeros((4, 4, 3), np.float32)
    a = np.full((4, 4), 0.25, np.float32)
    assert comp.compose(fg, a, bg).mean() == pytest.approx(0.25)
    assert comp.feather_alpha(a, 0) is a
    c = comp.ColorBg("#ff0000")
    f = next(c.frames(4, 2, "25/1", 0, 25))
    assert f.shape == (2, 4, 3) and f[0, 0].tolist() == [1, 0, 0]
    with pytest.raises(ConfigError):
        comp.ColorBg("red")
    with pytest.raises(ConfigError):
        comp.parse_background("/no/such/file.png")


def test_estimate_foreground_modes():
    img = np.random.RandomState(0).randint(0, 255, (40, 40, 3), np.uint8)
    a = np.zeros((40, 40), np.float32)
    a[10:30, 10:30] = 1
    a[9, 10:30] = 0.5
    for mode in ("ml", "fast", "none"):
        out = comp.estimate_foreground(img, a, mode)
        assert out.shape == (40, 40, 3) and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
    with pytest.raises(ConfigError):
        comp.estimate_foreground(img, a, "bad")


def test_color_match_moves_toward_bg():
    fg = np.full((50, 50, 3), 0.8, np.float32)
    bg = np.full((50, 50, 3), 0.2, np.float32)
    a = np.ones((50, 50), np.float32)
    assert comp.color_match(fg, a, bg, 1.0).mean() < comp.color_match(fg, a, bg, 0.3).mean() < fg.mean()
    assert comp.color_match(fg, a, bg, 0) is fg


def test_config_priority_cli_over_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("refiner: birefnet\nbg: blur\nchunk-size: 40\ntemporal_smooth: 0.1\n", encoding="utf-8")
    cfg = load_config(["in.mp4", "--config", str(y), "--bg", "color:#000000", "--despill", "green"])
    assert cfg.refiner == "birefnet" and cfg.bg == "color:#000000" and cfg.chunk_size == 40
    assert cfg.temporal_smooth == 0.1 and cfg.despill == "green" and cfg.resume is True
    assert load_config(["in.mp4", "--no-resume"]).resume is False
    assert load_config(["in.mp4"]).refiner == "none"
    assert load_config(["in.mp4", "--export-alpha"]).export_alpha == "both"
    y.write_text("bogus: 1\n")
    with pytest.raises(ConfigError):
        load_config(["in.mp4", "--config", str(y)])
    with pytest.raises(ConfigError):
        load_config(["in.mp4", "--select", "x"])


def test_device_auto_is_valid_and_forced_cpu():
    from videomatte.device import select_device

    assert select_device("auto") in ("cuda", "mps", "cpu")
    assert select_device("cpu") == "cpu"
    with pytest.raises(RuntimeError):
        select_device("tpu")
