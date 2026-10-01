# macOS 설치 가이드

## 검증 상태

**미검증 - 작성자 환경에서 테스트되지 않음.**
이 문서의 명령은 macOS 에서 실행해 본 적이 없습니다. 일반적으로 알려진 절차를 정리한 것이며, 버전·패키지 이름·동작은 실제와 다를 수 있습니다.
(코드 자체는 Linux 에서만 테스트되었습니다. [README 검증 현황](../README.md#검증-현황) 참고.)
실행해 보고 달랐던 부분은 이 문서를 고쳐 주세요.

## 1. Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew --version
```
설치 끝에 출력되는 "Next steps"(PATH 설정 줄)를 그대로 실행해야 `brew` 가 잡힙니다. Apple Silicon 은 `/opt/homebrew`, Intel 은 `/usr/local` 에 설치됩니다.

## 2. Python
권장: **3.11** (이 저장소는 Linux 에서 3.11.15 로만 테스트됨. 3.10~3.12 는 지원 의도이나 미검증).
```bash
brew install python@3.11
python3.11 --version
```

## 3. 저장소 + 가상환경(venv)
```bash
git clone https://github.com/workseok/- videomatte && cd videomatte      # 저장소 주소/폴더명은 실제에 맞게
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 4. ffmpeg
```bash
brew install ffmpeg
ffmpeg -version
ffmpeg -hide_banner -encoders | grep -E "libx264|prores_ks|videotoolbox"
```
- Homebrew 의 ffmpeg 가 어떤 빌드(GPL/LGPL)인지 **확인하지 못했습니다.** `ffmpeg -buildconf | tr ' ' '\n' | grep enable-gpl` 로 직접 확인하세요.
- `libx264` 가 없으면 이 도구는 `h264_videotoolbox` 등으로 자동 폴백합니다 (경고 출력). 빌드별 라이선스 차이: [THIRD_PARTY_LICENSES.md §3](../THIRD_PARTY_LICENSES.md).
- ProRes 4444 알파 export 에는 `prores_ks` 인코더가 필요합니다.

## 5. PyTorch (MPS)
Apple Silicon(M1 이상) 용 기본 PyTorch 휠에 MPS 백엔드가 포함됩니다 (macOS 12.3 이상 필요로 알려짐).
```bash
pip install torch torchvision
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
`True` 면 `--device auto` 가 MPS 를 선택합니다. 정확한 설치 명령은 https://pytorch.org 의 선택기를 확인하세요 (버전별로 바뀝니다).

### Apple Silicon vs Intel
| | Apple Silicon (M1+) | Intel Mac |
|---|---|---|
| GPU 가속 | MPS 사용 가능 | **MPS 없음 → CPU** |
| 속도 | 미측정 | 미측정 (매우 느릴 것으로 예상) |
| PyTorch | 기본 휠 | 최신 PyTorch 가 Intel Mac 휠을 제공하지 않을 수 있음 (**미확인**; 설치 실패 시 구버전 PyTorch 가 필요하고 다른 의존성과 맞지 않을 수 있음) |

### MPS 폴백 환경변수
MPS 에 구현되지 않은 연산은 CPU 로 대체하도록 `PYTORCH_ENABLE_MPS_FALLBACK=1` 이 필요합니다.
**이 도구는 실행 시 이 값을 자동으로 설정합니다**(이미 지정한 값은 덮어쓰지 않음). 직접 쓰는 스크립트에서는:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 6. 이 저장소 + SAM 2 설치
```bash
pip install -e .
pip install "git+https://github.com/facebookresearch/sam2.git"
```
- 설치 중 `Failed to build the SAM 2 CUDA extension` 메시지는 SAM 2 README 에 "무시해도 된다"고 안내되어 있고, macOS 에는 CUDA 가 없으므로 정상입니다. (Linux CPU 에서 확장 없이 동작함은 확인했습니다.)
- `sam2` 설치 시 컴파일러가 필요하면 `xcode-select --install` (Command Line Tools). 필요 여부는 미확인.

## 7. 모델 다운로드
```bash
python scripts/download_models.py                                           # SAM 2.1 small
python scripts/download_models.py grounding-dino-tiny --allow-unverified    # 가중치 라이선스 미확인: 직접 확인 후
python scripts/download_models.py list
```
가중치는 `models/` 에 저장됩니다(.gitignore 처리). 자세한 라이선스 상태는 [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).

## 8. 동작 확인
```bash
videomatte --doctor            # ffmpeg / torch / device / sam2 / 모델 존재 여부
pytest -q                      # 실제 가중치 없이 합성 클립으로 돌리는 테스트
videomatte input.mp4 --max-frames 30 --box 100,50,600,700 -o preview.mp4   # 짧게 먼저
```
`device : mps` 가 표시되면 MPS 를 사용 중입니다.

## 9. 자주 나는 오류
| 증상 | 원인 / 조치 |
|---|---|
| `ffmpeg/ffprobe 를 PATH 에서 찾을 수 없습니다` | `brew install ffmpeg` 후 새 터미널. `which ffmpeg` 확인 |
| `NotImplementedError: ... not currently implemented for the MPS device` | `PYTORCH_ENABLE_MPS_FALLBACK=1` (이 도구는 기본 설정). 터미널에서 `export` 로 덮어쓴 값이 `0` 인지 확인 |
| MPS 메모리 부족 / 시스템 멈춤 | `--chunk-size` ↓, `--sam-max-side 768`, `--refine-max-side 720`, `--sam-model tiny`. 그래도 안 되면 `--device cpu` |
| `externally-managed-environment` | 시스템 Python 에 pip 하지 말고 3단계의 venv 안에서 설치 |
| `sam2 ... import 실패` | venv 활성화 확인 후 6단계 재실행 |
| 모델 없음 에러 | 7단계 실행. `videomatte --doctor` 로 확인 |

더 많은 항목: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
