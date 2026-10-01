"""videomatte CLI 진입점."""

from __future__ import annotations

import logging
import sys

from .device import configure_environment

configure_environment()  # torch import 전에 MPS 폴백 환경변수 설정

from .config import load_config  # noqa: E402
from .errors import OutOfMemoryHint, VideomatteError  # noqa: E402


def _doctor(cfg) -> int:
    import shutil

    from . import device as dev
    from .io_video import available_encoders
    from .models import MODELS, models_dir

    print("== videomatte doctor ==")
    ff = shutil.which("ffmpeg")
    print(f"ffmpeg : {ff or '없음 (설치 필요)'}")
    if ff:
        enc = available_encoders()
        print(f"  libx264={'libx264' in enc}  prores_ks={'prores_ks' in enc}  nvenc={'h264_nvenc' in enc}  "
              f"videotoolbox={'h264_videotoolbox' in enc}")
    try:
        print("torch  :", dev.describe())
        print("device :", dev.select_device(cfg.device))
    except ImportError:
        print("torch  : 설치되지 않음 (docs/SETUP_*.md)")
    except RuntimeError as e:
        print("device :", e)
    try:
        import sam2  # noqa: F401
        print("sam2   : OK")
    except ImportError as e:
        print(f"sam2   : import 실패 ({e})")
    root = models_dir(cfg.models_dir)
    print(f"models : {root}")
    for s in MODELS.values():
        present = (root / s.local).exists()
        print(f"  [{'x' if present else ' '}] {s.name:24s} {'' if s.license_verified else '(라이선스 미확인)'}")
    return 0


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def main(argv: list[str] | None = None) -> int:
    try:
        cfg = load_config(argv)
    except VideomatteError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.DEBUG if cfg.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if cfg.doctor:
        return _doctor(cfg)
    from .pipeline import Pipeline

    try:
        try:
            Pipeline(cfg).run()
        except RuntimeError as e:
            if _is_oom(e):
                raise OutOfMemoryHint(
                    "메모리(VRAM) 부족입니다. --chunk-size 를 줄이거나, --sam-max-side / --refine-max-side 를 낮추거나, "
                    "--sam-model tiny 를 쓰거나, --device cpu 를 시도하세요. 이미 끝난 청크는 다시 실행하면 이어서 처리됩니다."
                ) from e
            raise
        except MemoryError as e:
            raise OutOfMemoryHint("시스템 메모리 부족입니다. --chunk-size 를 줄이세요.") from e
    except VideomatteError as e:
        if cfg.verbose:
            logging.exception("실패")
        print(f"\n오류: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n중단됨. 같은 명령을 다시 실행하면 이어서 처리합니다.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
