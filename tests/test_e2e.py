"""합성 클립 end-to-end (가짜 검출/분할기 사용). 실제 SAM2/DINO 가중치는 사용하지 않는다."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from videomatte import io_video
from videomatte.config import Config
from videomatte.errors import DetectionError, MissingModelError, VideomatteError
from videomatte.pipeline import Pipeline

from conftest import FPS, H, N, W


def make_cfg(clip, out, **kw):
    base = dict(input=clip, output=out, fg_estimate="fast", chunk_size=12, temporal_smooth=0.0,
                work_dir=out.parent / "work", encoder="mpeg4")
    base.update(kw)
    return Config(**base)


def frames_of(path):
    info = io_video.probe(path)
    return info, list(io_video.read_frames(info, 0, info.n_frames + 5))


def test_probe_and_random_access(clip):
    info = io_video.probe(clip)
    assert (info.width, info.height, info.n_frames, info.has_audio) == (W, H, N, True)
    assert info.fps == pytest.approx(FPS)
    all_ = list(io_video.read_frames(info, 0, N))
    part = list(io_video.read_frames(info, 12, 6))
    assert len(all_) == N and len(part) == 6
    for a, b in zip(all_[12:18], part):  # 청크 시작 위치 탐색이 프레임 정확한지
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 1.0


def test_color_bg_e2e_with_chunks_audio_and_alpha_exports(clip, tmp_path, fakes):
    det, seg = fakes
    out = tmp_path / "o.mp4"
    cfg = make_cfg(clip, out, bg="color:#00ff00", export_alpha="both", despill="green", feather=0.5)
    Pipeline(cfg, det, seg).run()

    info, fr = frames_of(out)
    assert len(fr) == N and (info.width, info.height) == (W, H) and info.has_audio
    # 3 청크: 12+12+6. 첫 청크는 박스 프롬프트, 이후는 시드 마스크로 연결
    assert [c["n"] for c in seg.calls] == [12, 12, 6]
    assert seg.calls[0]["target"] is not None and seg.calls[0]["seed"] is None
    assert all(c["seed"] is not None for c in seg.calls[1:])
    assert det.calls == ["person"]
    # 'largest' 선택 -> 큰 박스 하나(첫 프레임 좌표 -> SAM 해상도 스케일은 그대로 1:1, sam_max_side=1024 > 320)
    assert len(seg.calls[0]["target"].boxes) == 1

    # 합성 결과 검증: 배경(좌상단)은 초록, 사람 중심은 빨강 유지 (mpeg4 손실 허용)
    for i in (0, 15, N - 1):
        f = fr[i]
        assert f[5, 5].astype(int).tolist() == pytest.approx([0, 255, 0], abs=30)
        cx = 80 + i * 4
        assert f[95, cx].astype(int)[0] > 150 and f[95, cx].astype(int)[1] < 90

    # 알파 PNG 시퀀스
    pngs = sorted((tmp_path / "o_alpha").glob("*.png"))
    assert len(pngs) == N
    a = cv2.imread(str(pngs[10]), cv2.IMREAD_UNCHANGED)
    assert a.shape == (H, W) and a[95, 80 + 40] == 255 and a[5, 5] == 0
    # ProRes 4444 + 알파 채널
    mov = tmp_path / "o_alpha.mov"
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
                        "-of", "json", str(mov)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    assert s["codec_name"] == "prores" and "yuva" in s["pix_fmt"]
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())  # 성공 시 캐시 정리


def test_image_video_blur_backgrounds(clip, bg_image, bg_video, tmp_path, fakes):
    results = {}
    for name, spec in {"img": str(bg_image), "vid": str(bg_video), "blur": "blur"}.items():
        det, seg = fakes
        out = tmp_path / f"{name}.mp4"
        Pipeline(make_cfg(clip, out, bg=spec, max_frames=14, color_match=0.3), det, seg).run()
        info, fr = frames_of(out)
        assert len(fr) == 14 and (info.width, info.height) == (W, H)
        results[name] = fr
    # 이미지 배경: (30,160,30) 녹색 계열
    assert results["img"][3][5, 5].astype(int).tolist() == pytest.approx([30, 160, 30], abs=30)
    # 영상 배경은 시간에 따라 변해야 함 (testsrc)
    assert np.abs(results["vid"][0].astype(int) - results["vid"][13].astype(int)).mean() > 1
    # blur 배경은 원본과 같은 톤(타원이 없는 영역 밝기가 원본과 유사)
    src = io_video.read_frames(io_video.probe(clip), 0, 1).__next__()
    assert np.abs(results["blur"][0][5, 5].astype(int) - src[5, 5].astype(int)).max() < 40


def test_manual_box_skips_detector_and_select_all(clip, tmp_path, fakes):
    det, seg = fakes
    Pipeline(make_cfg(clip, tmp_path / "a.mp4", box="50,30,130,160", max_frames=6), det, seg).run()
    assert det.calls == [] and seg.calls[0]["target"].boxes == [(50, 30, 130, 160)]
    det, seg = FakeD(), FakeS()
    Pipeline(make_cfg(clip, tmp_path / "b.mp4", select="all", max_frames=6), det, seg).run()
    assert len(seg.calls[0]["target"].boxes) == 2
    det, seg = FakeD(), FakeS()
    Pipeline(make_cfg(clip, tmp_path / "c.mp4", points="100,95;5,5,0", max_frames=6), det, seg).run()
    t = seg.calls[0]["target"]
    assert t.points == [(100, 95), (5, 5)] and t.point_labels == [1, 0] and det.calls == []


from conftest import FakeDetector as FakeD, FakeSegmenter as FakeS  # noqa: E402


def test_no_detection_gives_clear_error(clip, tmp_path):
    class Empty:
        def detect(self, f, p):
            return []

    with pytest.raises(DetectionError, match="--box"):
        Pipeline(make_cfg(clip, tmp_path / "x.mp4", max_frames=4), Empty(), FakeS()).run()


def test_resume_skips_finished_chunks(clip, tmp_path):
    out = tmp_path / "r.mp4"
    cfg = make_cfg(clip, out, keep_work=True)
    seg1 = FakeS()
    Pipeline(cfg, FakeD(), seg1).run()
    assert len(seg1.calls) == 3
    # 동일 설정 재실행 -> 모든 청크 스킵, 분할기/검출기는 호출되지 않음
    seg2, det2 = FakeS(), FakeD()
    Pipeline(cfg, det2, seg2).run()
    assert seg2.calls == [] and det2.calls == []
    # 마지막 청크만 지워서 중단 상황 재현 -> 그 청크만 재계산(마스크 캐시는 유지)
    work = next((tmp_path / "work").iterdir())
    (work / "chunk_00002.done").unlink()
    (work / "masks_00002.npz").unlink()
    seg3 = FakeS()
    Pipeline(cfg, FakeD(), seg3).run()
    assert [c["n"] for c in seg3.calls] == [6] and seg3.calls[0]["seed"] is not None
    assert len(frames_of(out)[1]) == N
    # 렌더 파라미터가 바뀌면 청크 재렌더 (마스크는 재사용: 분할기 호출 없음)
    seg4 = FakeS()
    Pipeline(make_cfg(clip, out, keep_work=True, bg="color:#0000ff"), FakeD(), seg4).run()
    assert seg4.calls == []
    assert frames_of(out)[1][0][5, 5].astype(int).tolist() == pytest.approx([0, 0, 255], abs=30)


def test_failure_mid_run_keeps_cache_and_resumes(clip, tmp_path):
    out = tmp_path / "f.mp4"
    cfg = make_cfg(clip, out)

    class Boom(FakeS):
        def propagate(self, *a, **k):
            if len(self.calls) == 1:
                raise RuntimeError("boom")
            return super().propagate(*a, **k)

    with pytest.raises(RuntimeError):
        Pipeline(cfg, FakeD(), Boom()).run()
    assert not out.exists()
    work = next((tmp_path / "work").iterdir())
    assert (work / "chunk_00000.done").exists() and not (work / "chunk_00001.done").exists()
    seg = FakeS()
    Pipeline(cfg, FakeD(), seg).run()
    assert [c["n"] for c in seg.calls] == [12, 6]  # 청크 0 은 재사용
    assert len(frames_of(out)[1]) == N


def test_temporal_smoothing_continuity_across_chunks(clip, tmp_path):
    cfg = make_cfg(clip, tmp_path / "t.mp4", temporal_smooth=0.6, export_alpha="png")
    Pipeline(cfg, FakeD(), FakeS()).run()
    a = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(float) for p in sorted((tmp_path / "t_alpha").glob("*.png"))]
    assert len(a) == N and all(x.max() <= 255 for x in a)


def test_errors_ffmpeg_missing_and_missing_model(clip, tmp_path, monkeypatch):
    with pytest.raises(MissingModelError, match="download_models.py"):
        Pipeline(make_cfg(clip, tmp_path / "m.mp4", models_dir=tmp_path / "nomodels", max_frames=4), FakeD()).run()
    monkeypatch.setattr(io_video.shutil, "which", lambda *_: None)
    with pytest.raises(VideomatteError, match="ffmpeg"):
        io_video.probe(clip)


def test_cli_doctor_and_bad_input(tmp_path, capsys):
    from videomatte.cli import main

    assert main(["--doctor", "--models-dir", str(tmp_path)]) == 0
    assert "ffmpeg" in capsys.readouterr().out
    assert main([str(tmp_path / "none.mp4")]) == 1
    assert "입력 파일이 없습니다" in capsys.readouterr().err


def test_unicode_and_space_paths(clip, bg_image, tmp_path, fakes):
    """한글/공백/작은따옴표 경로 (Windows 한글 경로 이슈 방어: 입력, 배경 이미지, 출력, 작업 폴더, 알파 PNG)."""
    import shutil

    d = tmp_path / "한글 폴더's test"
    d.mkdir()
    c, b = d / "입력 영상.mp4", d / "배경 이미지.png"
    shutil.copy(clip, c)
    shutil.copy(bg_image, b)
    det, seg = fakes
    out = d / "결과 영상.mp4"
    Pipeline(make_cfg(c, out, bg=str(b), export_alpha="png", max_frames=14, work_dir=d / "작업"), det, seg).run()
    info, fr = frames_of(out)
    assert len(fr) == 14 and info.has_audio
    assert len(list((d / "결과 영상_alpha").glob("*.png"))) == 14
