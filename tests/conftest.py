"""합성 테스트 클립 + 모델 없이 파이프라인을 검증하기 위한 가짜 검출기/분할기."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from videomatte.detect import Detection, Target

W, H, FPS, N = 320, 180, 25, 30
PERSON_RGB = (220, 40, 40)

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg 필요")


def synth_frame(i: int) -> np.ndarray:
    """회색 그라디언트 배경 + 좌->우로 움직이는 빨간 타원('사람')."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([60 + x * 0.1, 90 + y * 0.2, 120 + 0 * x], -1).astype(np.uint8)
    cx = 80 + i * 4
    cv2.ellipse(img, (cx, 95), (28, 60), 0, 0, 360, PERSON_RGB, -1)
    return img


@pytest.fixture(scope="session")
def clip(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("clip")
    raw = d / "raw.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "pipe:0", "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(raw)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(synth_frame(i).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    out = d / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def bg_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bg") / "bg.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=1",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.fixture()
def bg_image(tmp_path) -> Path:
    p = tmp_path / "bg.png"
    cv2.imwrite(str(p), np.full((90, 160, 3), (30, 160, 30), np.uint8))
    return p


class FakeDetector:
    def __init__(self):
        self.calls = []

    def detect(self, frame_rgb, prompt):
        self.calls.append(prompt)
        return [Detection((50, 30, 130, 160), 0.9, "person"), Detection((200, 100, 240, 150), 0.6, "person")]


class FakeSegmenter:
    """색상 임계값 분할. 첫 프레임 프롬프트/시드를 기록해 청크 연결을 검증한다."""

    def __init__(self):
        self.calls = []

    def propagate(self, frames_dir, n_frames, target, seed_mask):
        self.calls.append({"n": n_frames, "target": target, "seed": seed_mask})
        files = sorted(Path(frames_dir).glob("*.jpg"))
        assert len(files) == n_frames
        out = []
        for f in files:
            rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB).astype(int)
            out.append(np.abs(rgb - np.array(PERSON_RGB)).sum(-1) < 90)
        return np.stack(out)


@pytest.fixture()
def fakes():
    return FakeDetector(), FakeSegmenter()
