"""모델 레지스트리와 가중치 경로/로딩.

가중치는 저장소에 포함하지 않는다. scripts/download_models.py 로 models/ 에 내려받는다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import MissingModelError

CHECKSUM_FILE = "checksums.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "url" (단일 파일) | "hf" (Hugging Face 저장소 스냅샷)
    source: str  # URL 또는 HF repo id
    local: str  # models/ 아래 파일명 또는 디렉터리명
    license: str
    # 가중치 라이선스를 공식 출처에서 직접 확인했는가. False 면 다운로드에 --allow-unverified 필요.
    license_verified: bool
    note: str = ""
    sha256: str | None = None  # 고정된 해시를 알고 있을 때만 채운다 (현재는 없음)


_SAM_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

MODELS: dict[str, ModelSpec] = {
    f"sam2.1_hiera_{size}": ModelSpec(
        name=f"sam2.1_hiera_{size}",
        kind="url",
        source=f"{_SAM_BASE}/sam2.1_hiera_{size}.pt",
        local=f"sam2.1_hiera_{size}.pt",
        license="Apache-2.0",
        license_verified=True,
        note="sam2 README: 'model checkpoints ... are licensed under Apache 2.0' (2026-10-01 확인)",
    )
    for size in ("tiny", "small", "base_plus", "large")
}
MODELS.update(
    {
        "grounding-dino-tiny": ModelSpec(
            "grounding-dino-tiny", "hf", "IDEA-Research/grounding-dino-tiny", "grounding-dino-tiny",
            "미확인 (HF 모델 카드 확인 불가)", False,
            "코드(GroundingDINO 저장소, transformers)는 Apache-2.0 확인. 가중치 카드는 미확인.",
        ),
        "grounding-dino-base": ModelSpec(
            "grounding-dino-base", "hf", "IDEA-Research/grounding-dino-base", "grounding-dino-base",
            "미확인 (HF 모델 카드 확인 불가)", False, "위와 동일",
        ),
        "vitmatte-small": ModelSpec(
            "vitmatte-small", "hf", "hustvl/vitmatte-small-composition-1k", "vitmatte-small",
            "미확인 (코드는 MIT)", False,
            "학습 데이터(Composition-1k 계열)의 상업적 사용 가능 여부 불확실 -> 기본값 아님.",
        ),
        "vitmatte-base": ModelSpec(
            "vitmatte-base", "hf", "hustvl/vitmatte-base-composition-1k", "vitmatte-base",
            "미확인 (코드는 MIT)", False, "위와 동일",
        ),
        "birefnet": ModelSpec(
            "birefnet", "hf", "ZhengPeng7/BiRefNet", "birefnet",
            "미확인 (코드는 MIT, README상 공식 가중치는 제한 문구 없음)", False,
            "RMBG-2.0 계열 파인튜닝은 비상업 -> 반드시 공식 ZhengPeng7/BiRefNet 만 사용.",
        ),
    }
)

SAM_SIZES = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def models_dir(override: str | os.PathLike | None = None) -> Path:
    """우선순위: 인자 > $VIDEOMATTE_MODELS_DIR > 저장소의 models/ > ./models"""
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("VIDEOMATTE_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_models = Path(__file__).resolve().parents[2] / "models"
    if repo_models.is_dir():
        return repo_models
    return Path.cwd() / "models"


def model_path(name: str, override: str | os.PathLike | None = None) -> Path:
    spec = MODELS[name]
    path = models_dir(override) / spec.local
    if not path.exists():
        raise MissingModelError(
            f"모델 '{name}' 이(가) 없습니다: {path}\n"
            f"  -> python scripts/download_models.py {name}"
            + ("" if spec.license_verified else " --allow-unverified (라이선스 미확인 모델: THIRD_PARTY_LICENSES.md 확인)")
        )
    return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_checksums(root: Path) -> dict[str, str]:
    p = root / CHECKSUM_FILE
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_checksums(root: Path, data: dict[str, str]) -> None:
    (root / CHECKSUM_FILE).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
