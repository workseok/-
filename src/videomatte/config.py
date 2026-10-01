"""설정: CLI 인자 + 선택적 config.yaml (우선순위: CLI > yaml > 기본값)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from .errors import ConfigError


@dataclass
class Config:
    input: Path | None = None
    output: Path | None = None
    # 대상 지정
    prompt: str = "person"
    box: str | None = None
    points: str | None = None
    select: str = "largest"
    detector_threshold: float = 0.35
    dino_variant: str = "tiny"
    # 분할
    sam_model: str = "small"
    sam_max_side: int = 1024
    # 정밀화
    refiner: str = "none"
    vitmatte_variant: str = "small"
    trimap_width: int | None = None
    refine_max_side: int = 1080
    temporal_smooth: float = 0.3
    # 합성
    fg_estimate: str = "ml"
    despill: str = "none"
    despill_strength: float = 1.0
    bg: str = "color:#00ff00"
    feather: float = 0.0
    color_match: float = 0.0
    # 출력
    export_alpha: str | None = None
    encoder: str = "auto"
    crf: int = 18
    # 실행
    device: str = "auto"
    chunk_size: int = 150
    work_dir: Path | None = None
    resume: bool = True
    keep_work: bool = False
    max_frames: int | None = None
    models_dir: Path | None = None
    verbose: bool = False
    doctor: bool = False


CHOICES = {
    "select_re": r"^(largest|all|\d+)$",
    "sam_model": ["tiny", "small", "base_plus", "large"],
    "refiner": ["none", "vitmatte", "birefnet"],
    "fg_estimate": ["ml", "fast", "none"],
    "despill": ["none", "green", "blue"],
    "export_alpha": ["png", "prores", "both"],
    "dino_variant": ["tiny", "base"],
    "vitmatte_variant": ["small", "base"],
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videomatte",
        description="영상에서 대상을 자동으로 찾아 알파 매트를 만들고 새 배경과 합성합니다.",
    )
    p.add_argument("input", nargs="?", type=Path, help="입력 영상 (mp4/mov)")
    p.add_argument("-o", "--output", type=Path, help="출력 mp4 (기본: <입력>_matte.mp4)")
    p.add_argument("--config", type=Path, help="config.yaml (CLI 인자가 우선)")

    g = p.add_argument_group("대상 지정 (첫 프레임)")
    g.add_argument("--prompt", help="Grounding DINO 텍스트 프롬프트 (기본: person)")
    g.add_argument("--box", help="수동 박스 x1,y1,x2,y2 (원본 픽셀 좌표). 지정 시 검출 생략")
    g.add_argument("--points", help="수동 포인트 'x,y;x,y' (끝에 ,0 을 붙이면 배경 포인트)")
    g.add_argument("--select", help="여러 객체 선택: largest | all | 인덱스(면적 내림차순, 0=가장 큼)")
    g.add_argument("--detector-threshold", type=float, help="검출 박스 임계값 (기본 0.35)")
    g.add_argument("--dino-variant", choices=CHOICES["dino_variant"])

    g = p.add_argument_group("분할 / 정밀화")
    g.add_argument("--sam-model", choices=CHOICES["sam_model"], help="SAM 2.1 크기 (기본 small)")
    g.add_argument("--sam-max-side", type=int, help="SAM 입력 긴 변 상한 px, 0=원본 (기본 1024)")
    g.add_argument("--refiner", choices=CHOICES["refiner"], help="알파 정밀화 (기본 none; 가중치 라이선스 검증 전)")
    g.add_argument("--vitmatte-variant", choices=CHOICES["vitmatte_variant"])
    g.add_argument("--trimap-width", type=int, help="트라이맵 unknown 밴드 폭 px (기본: 해상도의 약 2%%)")
    g.add_argument("--refine-max-side", type=int, help="정밀화 입력 긴 변 상한 px, 0=원본. 알파는 업샘플 (기본 1080)")
    g.add_argument("--temporal-smooth", type=float, help="알파 시간 평활 강도 0~0.95 (기본 0.3, 0=끔)")

    g = p.add_argument_group("합성")
    g.add_argument("--fg-estimate", choices=CHOICES["fg_estimate"], help="전경색 추정 (ml=pymatting, fast=근사, none)")
    g.add_argument("--despill", choices=CHOICES["despill"], help="그린/블루 스필 제거")
    g.add_argument("--despill-strength", type=float)
    g.add_argument("--bg", help="이미지/영상 경로 | color:#RRGGBB | blur (기본 color:#00ff00)")
    g.add_argument("--feather", type=float, help="알파 에지 페더링 sigma(px)")
    g.add_argument("--color-match", type=float, help="전경 색을 배경 톤에 맞추는 강도 0~1")

    g = p.add_argument_group("출력")
    g.add_argument("--export-alpha", nargs="?", const="both", choices=CHOICES["export_alpha"],
                   help="알파 내보내기: png(시퀀스) | prores(ProRes 4444 mov) | both (값 생략 시 both)")
    g.add_argument("--encoder", help="H.264 인코더 (기본 auto: libx264 > nvenc > videotoolbox > ...)")
    g.add_argument("--crf", type=int, help="libx264 품질 (기본 18)")

    g = p.add_argument_group("실행")
    g.add_argument("--device", help="auto | cuda | mps | cpu")
    g.add_argument("--chunk-size", type=int, help="청크당 프레임 수 (기본 150). 메모리가 부족하면 줄이세요")
    g.add_argument("--work-dir", type=Path, help="중간 결과 캐시 위치")
    g.add_argument("--no-resume", dest="resume", action="store_false", default=None, help="캐시를 무시하고 처음부터")
    g.add_argument("--keep-work", action="store_true", default=None, help="성공 후에도 중간 결과 유지")
    g.add_argument("--max-frames", type=int, help="앞쪽 N 프레임만 처리 (미리보기/테스트)")
    g.add_argument("--models-dir", type=Path, help="가중치 폴더 (기본: 저장소 models/ 또는 $VIDEOMATTE_MODELS_DIR)")
    g.add_argument("-v", "--verbose", action="store_true", default=None)
    g.add_argument("--doctor", action="store_true", default=None, help="환경 점검 (디바이스/ffmpeg/모델) 후 종료")
    return p


_PATH_KEYS = {"input", "output", "work_dir", "models_dir"}


def load_config(argv: list[str] | None = None) -> Config:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    valid = {f.name for f in fields(Config)}
    if pre.config:
        if not pre.config.is_file():
            raise ConfigError(f"config 파일이 없습니다: {pre.config}")
        data = yaml.safe_load(pre.config.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError("config.yaml 최상위는 매핑(key: value)이어야 합니다.")
        data = {str(k).replace("-", "_"): v for k, v in data.items()}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ConfigError(f"config.yaml 에 알 수 없는 키: {unknown}")
        for k in _PATH_KEYS & set(data):
            data[k] = Path(data[k]).expanduser() if data[k] is not None else None
        parser.set_defaults(**data)
    ns = parser.parse_args(argv)
    cfg = Config()
    for f in fields(Config):
        v = getattr(ns, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    import re

    if not re.match(CHOICES["select_re"], str(cfg.select)):
        raise ConfigError(f"--select 는 largest|all|<정수>: {cfg.select!r}")
    for name, choices in CHOICES.items():
        if name.endswith("_re"):
            continue
        v = getattr(cfg, name)
        if v is not None and v not in choices:
            raise ConfigError(f"{name}={v!r} 는 {choices} 중 하나여야 합니다.")
    if cfg.chunk_size < 2:
        raise ConfigError("--chunk-size 는 2 이상이어야 합니다.")
    if not 0.0 <= cfg.temporal_smooth < 1.0:
        raise ConfigError("--temporal-smooth 는 0 이상 1 미만이어야 합니다.")
