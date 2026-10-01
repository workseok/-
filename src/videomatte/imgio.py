"""유니코드(한글) 경로에서도 동작하는 이미지 입출력.

cv2.imread/imwrite 는 Windows 에서 비ASCII 경로를 처리하지 못하므로
파일 입출력은 Python(pathlib)이 하고 OpenCV 는 메모리 버퍼만 인코딩/디코딩한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.frombuffer(Path(path).read_bytes(), np.uint8)
    return cv2.imdecode(data, flags)


def imwrite(path: Path, img: np.ndarray) -> None:
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise OSError(f"이미지 인코딩 실패: {path}")
    path.write_bytes(buf.tobytes())
