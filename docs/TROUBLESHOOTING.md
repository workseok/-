# 문제 해결

먼저 `videomatte --doctor` 를 실행하세요 (ffmpeg, torch, 디바이스, sam2, 모델 파일 상태 출력).
`-v` 를 붙이면 상세 로그/스택트레이스가 출력됩니다.

> 이 문서의 항목 중 **직접 재현해 확인한 것**은 "(재현 확인)" 으로 표시했습니다. 나머지는 코드/공식 문서 기준의 예상 조치이며 OS별 동작은 미검증입니다.

## 설치 / 환경
| 증상 | 조치 |
|---|---|
| `ffmpeg/ffprobe 를 PATH 에서 찾을 수 없습니다` | 설치 후 새 터미널. macOS `brew install ffmpeg`, Windows `winget install Gyan.FFmpeg` (docs/SETUP_*.md) |
| `sam2 패키지를 import 할 수 없습니다` | PyTorch 를 먼저 설치하고 `pip install "git+https://github.com/facebookresearch/sam2.git"`. venv 활성화 확인 |
| `Failed to build the SAM 2 CUDA extension` | 무시해도 됩니다 (SAM 2 README 안내). 실행 시 `cannot import name '_C'` 경고 후 후처리가 생략되며 CPU 에서 정상 동작 (재현 확인) |
| 인코더 경고 `libx264 가 없는 ffmpeg 라 ... 사용` | LGPL 빌드 ffmpeg. 동작하지만 화질/용량이 다를 수 있음. `--encoder` 로 지정 가능. [THIRD_PARTY_LICENSES.md §3](../THIRD_PARTY_LICENSES.md) |
| `이 ffmpeg 에는 인코더 '...' 가 없습니다` | `ffmpeg -encoders` 로 목록 확인 후 `--encoder` 수정 |
| ProRes 알파 export 실패 | ffmpeg 에 `prores_ks` 필요 (`ffmpeg -encoders \| grep prores`) |

## 모델
| 증상 | 조치 |
|---|---|
| `모델 'sam2.1_hiera_small' 이(가) 없습니다` (재현 확인) | `python scripts/download_models.py` |
| `[SKIP] ... 가중치 라이선스 미확인` | 해당 모델의 라이선스를 직접 확인한 뒤 `--allow-unverified` |
| 다운로드 실패 / 403 | 사내망·프록시가 `dl.fbaipublicfiles.com`, `huggingface.co` 를 차단했을 수 있음 (작성 환경에서 실제로 차단됨). 다른 네트워크에서 받아 `models/` 에 복사 |
| `[CHECKSUM MISMATCH]` | 파일 손상 또는 버전 변경. `--force` 로 재다운로드. 해시는 최초 다운로드 시 기록된 값(TOFU)임에 유의 |
| BiRefNet 이 코드 실행 관련 경고 | `trust_remote_code=True` 로 `models/birefnet` 안의 파이썬 파일을 실행합니다. 받은 파일을 신뢰할 수 있을 때만 사용 |

## 대상 검출
| 증상 | 조치 |
|---|---|
| `첫 프레임에서 'person' 를 찾지 못했습니다` (재현 확인) | `--detector-threshold 0.25`, 다른 `--prompt`, 또는 `--box x1,y1,x2,y2` / `--points`. 첫 프레임에 대상이 없으면 클립을 잘라서 사용 |
| 엉뚱한 사람이 선택됨 | `--select all` 또는 `--select 1` (면적 내림차순 인덱스), 또는 `--box` 로 직접 지정 |
| `--box` 좌표 기준 | **원본 영상 픽셀 좌표**(좌상단 원점). 내부에서 SAM 해상도로 자동 변환 |

## 메모리 / 속도
| 증상 | 조치 |
|---|---|
| `CUDA out of memory`, MPS 메모리 부족 | `--chunk-size 60`, `--sam-max-side 768`, `--refine-max-side 720`, `--sam-model tiny`, `--device cpu`. 이미 끝난 청크는 같은 명령 재실행 시 건너뜀 |
| 시스템 RAM 부족 | `--chunk-size` ↓ (청크 마스크가 메모리에 올라감) |
| 너무 느림 | `--fg-estimate fast`, `--refine-max-side` ↓, `--temporal-smooth` 는 비용 거의 없음. 먼저 `--max-frames 30` 로 시험 |
| 중간에 끊김 | 같은 명령 재실행 = 이어서 처리. 처음부터: `--no-resume`. 캐시: `<출력 폴더>/.videomatte_work/` |
| 디스크가 찬다 | 작업 폴더는 성공 시 삭제됨. 실패/중단 시 남은 `.videomatte_work` 를 지워도 됨 |

## 품질
| 증상 | 조치 |
|---|---|
| 경계에 배경색 테두리 | `--fg-estimate ml`(기본), `--despill green/blue`, `--feather 1` |
| 가장자리가 딱딱함 | `--refiner` 사용(라이선스 확인 후), `--trimap-width` 증가 |
| 알파가 깜빡임 | `--temporal-smooth 0.5` 이상. 빠른 움직임에서 번지면 낮추기 |
| 전경이 배경과 따로 노는 느낌 | `--color-match 0.3~0.6` |
| 중간에 대상을 놓침/엉뚱한 영역 추적 | SAM 은 첫 프레임 프롬프트 기반. `--sam-model base_plus/large`, 더 정확한 `--box`, 또는 `--points` 로 배경 포인트(`x,y,0`) 추가 |

## 영상 입출력
| 증상 | 조치 |
|---|---|
| 출력 해상도가 1px 작음 | yuv420p 는 짝수 해상도 필요. 정상 |
| 오디오가 없음 | 입력에 오디오 스트림이 없거나 `-shortest` 로 영상 길이에 맞춰 잘림. mov→mp4 에서 복사 불가 시 AAC 192k 로 재인코딩 |
| 가변 프레임레이트(VFR) 영상에서 청크 경계가 어긋남 | 알려진 한계. 먼저 `ffmpeg -i in.mp4 -vsync cfr -r 30 cfr.mp4` 로 고정 프레임레이트 변환 후 처리 |
| 한글/공백 경로 | 코드에서 방어(Linux 테스트 통과). Windows 에서 문제가 있으면 영문 경로로 이동 |

## MPS (macOS)
`NotImplementedError: ... MPS` → `PYTORCH_ENABLE_MPS_FALLBACK=1` (이 도구가 기본 설정. 이미 `0` 으로 지정해 둔 값은 덮어쓰지 않음).
