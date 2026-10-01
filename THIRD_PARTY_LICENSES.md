# 서드파티 라이선스 검증 결과

- 확인일: **2026-10-01**
- 확인 방법: 각 공식 GitHub 저장소의 `LICENSE`/`README` 원문을 직접 조회.
- **제약**: 작성 환경의 네트워크 정책이 `huggingface.co`, `ffmpeg.org` 를 차단하여 **Hugging Face 모델 카드(가중치 라이선스 필드)와 ffmpeg.org 는 조회하지 못했습니다.** 해당 칸은 "미확인"으로 표기했고, 추측으로 채우지 않았습니다.
- 이 문서는 법률 자문이 아닙니다. 상업 배포 전에 아래 "미확인" 항목을 직접 확인하세요.

## 1. 사용 후보

| 이름 | 버전/기준 | 코드 라이선스 | 가중치 라이선스 | 출처 | 상태 |
|---|---|---|---|---|---|
| SAM 2.1 (facebookresearch/sam2) | SAM 2.1 체크포인트(2024-09-30 릴리스) | Apache-2.0 | **Apache-2.0** (README: "model checkpoints ... are licensed under Apache 2.0") | https://github.com/facebookresearch/sam2 | 확인됨 |
| Grounding DINO (IDEA-Research/GroundingDINO) | - | Apache-2.0 (Copyright IDEA Research) | **미확인** (저장소 README에 가중치 라이선스 언급 없음, HF 카드 조회 불가) | https://github.com/IDEA-Research/GroundingDINO | 코드만 확인 |
| transformers (HF 구현체: Grounding DINO, ViTMatte) | 5.18.0 (테스트 환경) | Apache-2.0 | - | https://github.com/huggingface/transformers | 확인됨 |
| ViTMatte (hustvl/ViTMatte) | - | MIT (Hust Vision Lab) | **미확인** — README에 학습 데이터 명시 없음. 벤치마크(Composition-1k 등) 데이터셋 약관이 상업 이용을 제한할 가능성이 있으나 **이번에 검증하지 못함** | https://github.com/hustvl/ViTMatte | 코드만 확인 / **기본값 아님** |
| BiRefNet (ZhengPeng7/BiRefNet) | - | MIT (Copyright 2024 ZhengPeng) | README상 공식 가중치는 별도 제한 문구 없음. **HF 카드는 미확인.** README에 "RMBG-2.0 계열 파인튜닝 가중치는 비상업 전용" 문구가 있으므로 공식 `ZhengPeng7/BiRefNet` 외 변종 사용 금지 | https://github.com/ZhengPeng7/BiRefNet | 부분 확인 / **기본값 아님** |
| pymatting | 1.1.16 (테스트 환경) | MIT | - (모델 가중치 없음) | https://github.com/pymatting/pymatting | 확인됨 |
| FFmpeg | 6.1.1 (테스트 환경, 시스템 패키지) | 기본 LGPL-2.1+ (`--enable-gpl` 시 GPL-2+, `--enable-nonfree` 시 재배포 불가) | - | https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md | GitHub 사본으로 확인 (ffmpeg.org 미조회) |

### 이 저장소의 기본 동작
- 기본 `--refiner` 는 **`none`** (SAM 마스크 + 소프트 에지). ViTMatte/BiRefNet 가중치 라이선스가 검증되기 전에는 기본값으로 쓰지 않습니다.
- `scripts/download_models.py` 는 가중치 라이선스가 "미확인"인 모델(Grounding DINO, ViTMatte, BiRefNet)을 `--allow-unverified` 없이는 내려받지 않습니다.
- 기본 대상 검출(`--prompt`)은 Grounding DINO 를 사용하므로, 가중치 라이선스 확인 전에는 `--box` / `--points` 수동 지정만으로도 SAM 2.1 단독 사용이 가능합니다.

## 2. 사용 금지 (상업 이용 제약)

| 이름 | 라이선스 | 비고 |
|---|---|---|
| Robust Video Matting | GPL-3.0 | 사용하지 않음 |
| Ultralytics YOLO | AGPL-3.0 | 사용하지 않음 |
| BRIA RMBG 계열 | 비상업 조건 | 사용하지 않음 (BiRefNet RMBG-2.0 파인튜닝 가중치 포함) |
| MatAnyone | 비상업 | 사용하지 않음 |

## 3. FFmpeg 빌드별 주의 (subprocess 호출만 함)

이 도구는 FFmpeg 를 **외부 프로그램으로 실행만** 하며 링크/번들하지 않습니다. 따라서 FFmpeg 라이선스가 이 저장소 코드로 전염되는 구조는 아닙니다(별도 프로세스). 단, 다음은 사용자가 알아야 합니다.

- **LGPL 빌드**: `libx264` 가 없습니다. 이 도구는 자동으로 다른 H.264 인코더(`h264_nvenc`, `h264_videotoolbox`, `h264_mf` 등)나 `mpeg4` 로 대체하며 경고를 출력합니다.
- **GPL 빌드** (`--enable-gpl`, 보통 `libx264` 포함): 시스템에 설치된 ffmpeg 를 사용하는 것은 문제없으나, **ffmpeg 바이너리를 제품에 포함해 재배포**하면 GPL 의무(소스 제공 등)가 생깁니다.
- **nonfree 빌드** (`--enable-nonfree`): 재배포 불가. 재배포 목적이면 사용하지 마세요.
- 이 저장소는 ffmpeg 바이너리를 포함하지 않으며, 사용자가 직접 설치합니다.
- 현재 환경의 ffmpeg(6.1.1)는 `--enable-gpl` 빌드임을 `ffmpeg -buildconf` 로 확인했습니다. 본인 빌드는 `ffmpeg -buildconf` 로 확인하세요.

## 4. 기타 파이썬 의존성
numpy, opencv-python-headless, Pillow, tqdm, PyYAML, PyTorch, torchvision, huggingface_hub, hydra-core, iopath: **이번 세션에서 라이선스를 검증하지 않았습니다.** 배포 전에 `pip-licenses` 등으로 확인하세요.

## 5. 가중치 체크섬
`ModelSpec.sha256` 고정 해시는 현재 비어 있습니다 (다운로드 서버 접근 불가로 값을 확인하지 못함). `scripts/download_models.py` 는 최초 다운로드 시 SHA-256 을 `models/checksums.json` 에 기록하고 이후 대조합니다(최초 신뢰 방식). 공식 해시를 확인하면 `src/videomatte/models.py` 에 고정하세요.
