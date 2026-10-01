"""ffmpeg/ffprobe subprocess 기반 영상 입출력. OS 의존 코드 없음 (pathlib + 리스트 인자)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .errors import FFmpegNotFoundError, VideomatteError

log = logging.getLogger(__name__)


def require_ffmpeg() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FFmpegNotFoundError(
            "ffmpeg/ffprobe 를 PATH 에서 찾을 수 없습니다. 설치 후 새 터미널을 여세요.\n"
            "  macOS: brew install ffmpeg / Windows: winget install Gyan.FFmpeg (docs/SETUP_*.md 참고)"
        )
    return ffmpeg, ffprobe


def _run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, **kw)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    fps_frac: str
    n_frames: int
    duration: float
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    _, ffprobe = require_ffmpeg()
    path = Path(path)
    if not path.is_file():
        raise VideomatteError(f"입력 파일이 없습니다: {path}")
    r = _run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    if r.returncode != 0:
        raise VideomatteError(f"ffprobe 실패: {r.stderr.decode(errors='replace').strip()}")
    info = json.loads(r.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise VideomatteError(f"비디오 스트림이 없습니다: {path}")
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1"
    if rate in ("0/0", "0/1"):
        rate = v.get("r_frame_rate", "30/1")
    fps = float(Fraction(rate))
    duration = float(v.get("duration") or info.get("format", {}).get("duration") or 0)
    n = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else int(round(duration * fps))
    return VideoInfo(
        path=path, width=int(v["width"]), height=int(v["height"]), fps=fps, fps_frac=rate,
        n_frames=n, duration=duration, has_audio=any(s["codec_type"] == "audio" for s in info["streams"]),
    )


def _scaled(w: int, h: int, max_side: int) -> tuple[int, int]:
    if not max_side or max(w, h) <= max_side:
        return w, h
    s = max_side / max(w, h)
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def read_frames(info: VideoInfo, start: int, count: int) -> Iterator[np.ndarray]:
    """[start, start+count) 프레임을 RGB uint8 (H,W,3) 로 순서대로 yield (원본 해상도)."""
    ffmpeg, _ = require_ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start / info.fps:.6f}"]
    cmd += ["-i", info.path, "-frames:v", count, "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "pipe:1"]
    yield from _read_raw(cmd, info.width, info.height, 3)


def _read_raw(cmd: Sequence[str], w: int, h: int, ch: int) -> Iterator[np.ndarray]:
    size = w * h * ch
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, ch).copy()
    finally:
        proc.stdout.close()
        proc.kill() if proc.poll() is None else None
        proc.wait()


def extract_jpegs(info: VideoInfo, start: int, count: int, out_dir: Path, max_side: int = 1024) -> tuple[int, int, int]:
    """SAM 2 입력용 JPEG (00000.jpg ...) 추출. (n_extracted, w, h) 반환."""
    ffmpeg, _ = require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    w, h = _scaled(info.width, info.height, max_side)
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start / info.fps:.6f}"]
    cmd += ["-i", info.path, "-frames:v", count, "-vsync", "0", "-vf", f"scale={w}:{h}", "-q:v", "2",
            "-start_number", "0", out_dir / "%05d.jpg"]
    r = _run(cmd)
    if r.returncode != 0:
        raise VideomatteError(f"프레임 추출 실패: {r.stderr.decode(errors='replace').strip()}")
    return len(list(out_dir.glob("*.jpg"))), w, h


def available_encoders() -> set[str]:
    ffmpeg, _ = require_ffmpeg()
    r = _run([ffmpeg, "-hide_banner", "-encoders"])
    names = set()
    for line in r.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


H264_PREFERENCE = ["libx264", "h264_nvenc", "h264_videotoolbox", "h264_mf", "h264_qsv", "h264_amf", "mpeg4"]


def pick_h264_encoder(requested: str = "auto") -> str:
    enc = available_encoders()
    if requested != "auto":
        if requested not in enc:
            raise VideomatteError(f"이 ffmpeg 에는 인코더 '{requested}' 가 없습니다. (ffmpeg -encoders 로 확인)")
        return requested
    for name in H264_PREFERENCE:
        if name in enc:
            if name != "libx264":
                log.warning("libx264 가 없는 ffmpeg(LGPL 빌드 등)라 '%s' 인코더를 사용합니다.", name)
            return name
    raise VideomatteError("사용 가능한 H.264/MPEG-4 인코더가 없습니다. 다른 ffmpeg 빌드를 설치하세요.")


class RawWriter:
    """rawvideo 프레임을 stdin 으로 받아 ffmpeg 로 인코딩. with 문으로 사용."""

    def __init__(self, path: Path, w: int, h: int, fps_frac: str, pix_in: str, codec_args: Sequence[str]):
        ffmpeg, _ = require_ffmpeg()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(self.path.stem + ".partial" + self.path.suffix)
        self._w, self._h, self._ch = w, h, {"rgb24": 3, "rgba": 4}[pix_in]
        cmd = [ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}",
               "-r", fps_frac, "-i", "pipe:0", *codec_args, self._tmp]
        self._proc = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape == (self._h, self._w, self._ch) and frame.dtype == np.uint8, frame.shape
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            raise VideomatteError("ffmpeg 인코더가 종료되었습니다: " + self._finish_err()) from None
        self.n += 1

    def _finish_err(self) -> str:
        return self._proc.stderr.read().decode(errors="replace").strip()

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        err = self._finish_err()
        rc = self._proc.wait()
        if rc != 0:
            self._tmp.unlink(missing_ok=True)
            raise VideomatteError(f"ffmpeg 인코딩 실패(rc={rc}): {err}")
        self._tmp.replace(self.path)  # 원자적으로 완료 표시 -> resume 안전

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()
        self._tmp.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            self.abort()


def h264_args(encoder: str, crf: int = 18) -> list[str]:
    # yuv420p 는 짝수 해상도가 필요하므로 호출 측에서 짝수로 맞춘다.
    base = ["-c:v", encoder, "-pix_fmt", "yuv420p"]
    if encoder == "libx264":
        return base + ["-preset", "medium", "-crf", str(crf)]
    if encoder == "mpeg4":
        return base + ["-q:v", "2"]
    return base + ["-b:v", "20M"]


PRORES4444_ARGS = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]


def concat_and_mux(segments: Sequence[Path], audio_src: Path | None, out: Path, work: Path) -> None:
    """세그먼트들을 -c copy 로 이어붙이고, 원본 오디오를 다시 입힌다 (복사 불가 시 AAC)."""
    ffmpeg, _ = require_ffmpeg()
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = work / "concat.txt"
    def esc(p: Path) -> str:  # concat demuxer: 작은따옴표 이스케이프
        return Path(p).resolve().as_posix().replace("'", "'\\''")

    lst.write_text("".join(f"file '{esc(s)}'\n" for s in segments), encoding="utf-8")
    base = [ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", lst]
    tmp = out.with_name(out.stem + ".partial" + out.suffix)
    if audio_src is None:
        attempts = [base + ["-c", "copy", tmp]]
    else:
        m = ["-map", "0:v:0", "-map", "1:a?", "-shortest"]
        attempts = [
            base + ["-i", audio_src, *m, "-c:v", "copy", "-c:a", "copy", tmp],
            base + ["-i", audio_src, *m, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", tmp],
        ]
    last = None
    for cmd in attempts:
        r = _run(cmd)
        if r.returncode == 0:
            tmp.replace(out)
            return
        last = r.stderr.decode(errors="replace").strip()
        tmp.unlink(missing_ok=True)
    raise VideomatteError(f"세그먼트 병합/오디오 remux 실패: {last}")


def read_background_video(path: Path, w: int, h: int, fps_frac: str, start_seconds: float = 0.0) -> Iterator[np.ndarray]:
    """배경 영상을 출력 해상도/프레임레이트로 맞춰(cover+crop) 무한 루프로 yield."""
    ffmpeg, _ = require_ffmpeg()
    vf = f"fps={fps_frac},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-stream_loop", "-1"]
    if start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.6f}"]
    cmd += ["-i", path, "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    yield from _read_raw(cmd, w, h, 3)
