# videomatte — 영상 자동 누끼(알파 매트) + 배경 합성 CLI

영상에서 대상(기본: 사람)을 **자동으로 찾아** 마스크를 만들고, 알파 매트를 추출한 뒤 **새 배경과 합성**하는 명령줄 도구입니다.
코드베이스는 하나이며 macOS / Windows(/Linux)에서 동작하도록 작성했습니다. OS별 차이는 코드가 아니라 문서로 다룹니다.

> **현재 검증 상태 요약** (자세한 내용은 맨 아래 "검증 현황")
> - Linux(CPU)에서 합성 테스트 클립으로 파이프라인/청크/resume/export/한글 경로를 검증했습니다.
> - **실제 모델 가중치(SAM 2.1, Grounding DINO, ViTMatte, BiRefNet)로는 실행하지 못했습니다.** 작성 환경에서 가중치 서버가 차단되어 있었습니다.
> - macOS / Windows 에서는 **실행해 본 적이 없습니다.**

## 파이프라인

```
 영상(mp4/mov)
   │ ffmpeg (프레임 읽기, 청크 단위 탐색)
   ▼
 [1] 첫 프레임 대상 지정 ── --prompt "person" → Grounding DINO 박스
   │                         └ 또는 --box / --points (수동)   ── --select largest|all|N
   ▼
 [2] SAM 2.1 video predictor ── 첫 프레임 프롬프트를 전체 프레임에 전파 (청크마다 이전 청크 마지막 마스크로 이어받음)
   ▼
 [3] 트라이맵(erode/dilate) → 알파 정밀화 (--refiner none|vitmatte|birefnet)
   │   └ 시간 안정화 (--temporal-smooth: 알파 EMA, 움직임이 큰 픽셀은 잔상 방지)
   ▼
 [4] 전경색 추정 (pymatting estimate_foreground_ml) → despill(그린/블루) → 페더링
   ▼
 [5] 배경 합성 (--bg 이미지 | 영상 | color:#RRGGBB | blur), 색 매칭 옵션
   ▼
 [6] ffmpeg 인코딩 → 합성 mp4 (원본 오디오 유지)  [+ 알파 PNG 시퀀스 / ProRes 4444 mov]
```

## 사용 모델과 라이선스 요약

| 용도 | 구성요소 | 코드 | 가중치 | 상태 |
|---|---|---|---|---|
| 영상 마스크 전파 | SAM 2.1 | Apache-2.0 | Apache-2.0 | 확인됨 |
| 텍스트 → 박스 | Grounding DINO (HF transformers) | Apache-2.0 | **미확인** | 코드만 확인 |
| 알파 정밀화 | ViTMatte | MIT | **미확인** (학습 데이터 약관 불확실) | 기본값 아님 |
| 대안 정밀화 | BiRefNet (공식 가중치만) | MIT | **미확인** | 기본값 아님 |
| 전경색 추정 | pymatting | MIT | - | 확인됨 |
| 영상 입출력 | FFmpeg (subprocess 호출만) | LGPL/GPL (빌드에 따라) | - | 확인됨 |

사용 금지: Robust Video Matting(GPL-3.0), Ultralytics YOLO(AGPL-3.0), BRIA RMBG 계열, MatAnyone(비상업).
**상세 근거, 확인일, 미확인 사유: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**
이 저장소 자체 코드는 [MIT 라이선스](LICENSE)입니다. 서드파티 구성요소는 각자의 라이선스를 따릅니다.

## 설치

OS별 설치 절차는 문서로 분리했습니다. (둘 다 **미검증** 문서입니다 — 각 문서 상단 참고)

- macOS → [docs/SETUP_MAC.md](docs/SETUP_MAC.md)
- Windows → [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)
- 문제 해결 → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 퀵스타트 (공통)

OS별 문서에 따라 Python 가상환경, ffmpeg, PyTorch, SAM 2 를 설치했다고 가정합니다.

```bash
pip install -e .                                   # 이 저장소 설치 (PyTorch/SAM2 는 제외되어 있음)
python scripts/download_models.py                  # 기본: SAM 2.1 small
python scripts/download_models.py grounding-dino-tiny --allow-unverified   # 가중치 라이선스 미확인: 직접 확인 후 사용
videomatte --doctor                                # 환경/모델 점검
videomatte input.mp4 --bg color:#00ff00 -o out.mp4
```

> `--prompt` 자동 검출을 쓰지 않고 `--box`/`--points` 만 쓰면 Grounding DINO 가중치가 필요 없습니다.

## CLI 사용 예시

```bash
# 1) 기본: 사람 자동 누끼 (그린 배경에 합성)
videomatte input.mp4 -o out.mp4

# 2) 배경 영상 합성 (해상도/프레임레이트 자동 맞춤, 짧으면 반복) + 전경 톤 맞춤 + 에지 페더링
videomatte input.mp4 --bg background.mp4 --color-match 0.4 --feather 1.0 -o out.mp4

# 3) 수동 박스 지정 (원본 픽셀 좌표, 검출 모델 불필요) / 포인트
videomatte input.mp4 --box 420,80,1180,1040 --bg photo.jpg -o out.mp4
videomatte input.mp4 --points "800,500;100,100,0" -o out.mp4      # 끝에 ,0 은 배경 포인트

# 4) 알파 내보내기: PNG 시퀀스(out_alpha/) + 알파 포함 ProRes 4444 (out_alpha.mov)
videomatte input.mp4 --export-alpha -o out.mp4          # = --export-alpha both
videomatte input.mp4 --export-alpha png  -o out.mp4     # PNG 시퀀스만
videomatte input.mp4 --export-alpha prores -o out.mp4   # ProRes 4444 만

# 5) 긴 영상: 청크 단위 + 해상도 다운스케일 후 알파 업샘플, 중단되면 같은 명령으로 이어서
videomatte long.mp4 --chunk-size 100 --sam-max-side 768 --refine-max-side 720 --sam-model tiny -o out.mp4
# (Ctrl+C 로 중단했다면 같은 명령을 다시 실행하면 끝난 청크는 건너뜁니다. 처음부터: --no-resume)

# 6) 여러 객체 중 선택 / 설정 파일
videomatte input.mp4 --prompt "person" --select all
videomatte input.mp4 --select 1                 # 면적 내림차순 인덱스 (0 = 가장 큼)
videomatte input.mp4 --config config.example.yaml --bg blur
```

주요 옵션: `--refiner none|vitmatte|birefnet`(기본 none), `--temporal-smooth 0~0.95`(기본 0.3), `--fg-estimate ml|fast|none`,
`--despill none|green|blue`, `--device auto|cuda|mps|cpu`, `--chunk-size`, `--work-dir`, `--max-frames N`(앞부분만 미리보기).
전체: `videomatte --help`. 설정 파일 예시: [config.example.yaml](config.example.yaml) (우선순위: CLI > yaml > 기본값).

### 동작 방식 메모
- **디바이스**: CUDA → MPS → CPU 자동 선택. MPS 에서 미지원 연산은 CPU 폴백(`PYTORCH_ENABLE_MPS_FALLBACK=1` 을 코드에서 기본 설정).
- **메모리**: SAM 2 프레임/상태를 CPU 로 offload, 청크 단위 처리, 마스크는 SAM 해상도(기본 긴 변 1024)에서 만들고 원본으로 업샘플, 정밀화는 `--refine-max-side` 이하에서 수행 후 알파를 업샘플.
- **resume**: 작업 폴더(`<출력 폴더>/.videomatte_work/<입력명>-<해시>/`)에 청크별 마스크/세그먼트를 저장. 입력 파일·대상 지정·SAM 설정이 바뀌면 새 폴더, 렌더 옵션(배경, 정밀화 등)만 바뀌면 마스크는 재사용하고 청크만 다시 렌더. 성공하면 자동 삭제(`--keep-work` 로 유지).
- **출력 해상도**: yuv420p 때문에 홀수 해상도는 1px 잘려 짝수가 됩니다.
- **인코더**: `libx264` 우선, 없으면(LGPL 빌드 등) nvenc/videotoolbox/mf/… 또는 `mpeg4` 로 폴백(경고 출력). FFmpeg 빌드 차이는 [THIRD_PARTY_LICENSES.md §3](THIRD_PARTY_LICENSES.md).

## 하드웨어 권장 사양 / 속도

실제 모델로 측정하지 못해 대부분 **미측정**입니다. 아래 표의 "측정" 칸만 이 저장소 작성 환경(Linux, CPU 4코어, RAM 15GB, GPU 없음)에서 실제로 잰 값입니다.

| 항목 | 값 |
|---|---|
| 권장 GPU VRAM (SAM 2.1 small + ViTMatte, 1080p) | 미측정 |
| 권장 시스템 RAM | 미측정 |
| Apple Silicon (MPS) 처리 속도 | 미측정 |
| NVIDIA GPU 처리 속도 (fps) | 미측정 |
| SAM 2.1 + Grounding DINO + 정밀화 CPU 처리 속도 | 미측정 |
| **측정**: 전경색 추정 `--fg-estimate ml`, 1080p 1프레임, CPU 4코어 | 약 1.06초 (무작위 이미지 기준) |
| **측정**: 전경색 추정 `--fg-estimate fast`, 동일 조건 | 약 0.52초 |

참고(측정 아님): SAM 2 는 GPU 사용을 전제로 설계되었고 CPU 는 매우 느릴 수 있습니다. 긴 영상은 `--sam-model tiny`, 낮은 `--sam-max-side`, 작은 `--chunk-size` 부터 시도하세요.

## 개발 / 테스트

```bash
pip install -e . pytest
pytest -q        # 합성 클립 e2e, 단위, SAM2/ViTMatte 배선 테스트 (실제 가중치 불필요)
```

## 검증 현황

**검증됨** (Linux x86_64, Python 3.11.15, PyTorch 2.14.1 CPU, ffmpeg 6.1.1(GPL 빌드), 테스트 25개 통과)
- 합성 테스트 클립(오디오 포함) e2e: 청크 분할/병합, 오디오 유지, 색/이미지/영상/블러 배경, 수동 박스·포인트·select, 알파 PNG, ProRes 4444(알파 채널 확인)
- resume(정상 재실행 스킵, 중단 후 이어하기, 렌더 옵션 변경 시 재렌더), 청크 간 마스크 시드 연결
- 한글/공백/작은따옴표 경로 (입력/출력/배경/작업 폴더/알파 PNG)
- SAM 2 **API 배선**: 무작위 초기화 tiny 모델로 `init_state` / `add_new_points_or_box` / `add_new_mask` / `propagate_in_video` / `reset_state` 호출 및 출력 shape 확인 (마스크 품질은 검증 불가), CUDA 확장 없이 CPU 폴백 동작
- ViTMatte 로더/전처리/출력 크롭 배선 (무작위 초기화 tiny 모델)
- `--doctor`, 입력 없음/모델 없음/검출 실패 에러 메시지, config.yaml 우선순위

**미검증**
- 실제 가중치로 한 분할/검출/정밀화 **결과 품질** 전부 (SAM 2.1, Grounding DINO, ViTMatte, BiRefNet)
- Grounding DINO 후처리/ BiRefNet 추론 코드 경로 (실제 모델 없이 실행해 보지 못함)
- `scripts/download_models.py` 의 실제 다운로드/체크섬 성공 경로 (서버 차단; 목록/실패 경로만 실행)
- CUDA, MPS 디바이스 동작, VRAM 부족 처리 (OOM 메시지 변환은 코드만 있고 실제 OOM 미재현)
- macOS, Windows 전체 (설치 문서의 명령 포함)
- 가변 프레임레이트(VFR) 영상에서 청크 시작 위치의 프레임 정확도, 수 시간짜리 영상

**알려진 한계**
- 청크 시작 탐색은 `ffmpeg -ss` 시간 기반이라 VFR 영상에서 프레임이 어긋날 수 있습니다.
- 청크 경계에서 SAM 이 이전 청크 마지막 마스크를 시드로 쓰므로 객체가 크게 변하면 경계에서 오차가 누적될 수 있습니다. 새로 등장하는 객체는 추적하지 않습니다.
- 알파 시간 평활은 EMA 기반으로, 강하게 걸면 빠른 움직임에서 경계가 번질 수 있습니다(움직임 적응 가중치로 완화).
- 합성은 sRGB 공간(비선형)에서 수행합니다.
- 알파 PNG 는 8비트입니다. ProRes 4444 는 straight(비프리멀티플라이드) 알파로 출력합니다.
- 다수 객체의 `--select all` 은 합집합 마스크 하나로 처리합니다(객체별 알파 분리 없음).
- 배경이 영상일 때 청크 재개 시 배경 재생 위치는 시간 기반 탐색이라 약간 어긋날 수 있습니다.
