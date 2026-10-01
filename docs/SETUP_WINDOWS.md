# Windows 설치 가이드

## 검증 상태

**미검증 - 작성자 환경에서 테스트되지 않음.**
이 문서의 명령은 Windows 에서 실행해 본 적이 없습니다. 일반적으로 알려진 절차를 정리한 것이며, 패키지 ID·URL·동작이 실제와 다를 수 있습니다.
(코드 자체는 Linux 에서만 테스트되었고, 한글/공백 경로는 코드에서 방어하여 Linux 에서 검증했으나 Windows 에서는 확인하지 못했습니다. [README 검증 현황](../README.md#검증-현황) 참고.)

> 명령은 **PowerShell** 기준입니다.

## 1. Python
권장 **3.11** (Linux 3.11.15 에서만 테스트됨).
- python.org 설치 프로그램: 설치 첫 화면에서 **"Add python.exe to PATH"** 체크.
- 또는: `winget install Python.Python.3.11`
```powershell
py -3.11 --version
```
Microsoft Store 의 `python` 별칭이 먼저 잡히면 "앱 실행 별칭 관리"에서 끄거나 `py -3.11` 을 사용하세요.

## 2. git
SAM 2 를 `pip install git+...` 로 설치하므로 git 이 필요합니다.
```powershell
winget install Git.Git
```

## 3. 저장소 + venv (PowerShell 실행 정책 이슈)
```powershell
git clone https://github.com/workseok/- videomatte    # 저장소 주소/폴더명은 실제에 맞게
cd videomatte
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
`이 시스템에서 스크립트를 실행할 수 없으므로 ... Activate.ps1 을 로드할 수 없습니다` 오류가 나면 실행 정책 때문입니다. 현재 사용자에 한해 허용:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
정책을 바꾸고 싶지 않다면 **cmd** 에서 `.venv\Scripts\activate.bat` 을 쓰거나, 활성화 없이 `.\.venv\Scripts\python.exe -m pip ...` 로 실행하세요.
```powershell
python -m pip install --upgrade pip
```

## 4. ffmpeg
```powershell
winget install Gyan.FFmpeg        # 패키지 ID 는 미검증. 안 되면 winget search ffmpeg
```
설치 후 **PowerShell 을 완전히 닫고 새로 열어야** PATH 가 반영됩니다.
```powershell
ffmpeg -version
ffmpeg -hide_banner -encoders | Select-String "libx264|prores_ks|nvenc"
ffmpeg -buildconf | Select-String "enable-gpl"      # GPL 빌드 여부 (미확인 항목)
```
PATH 에 안 잡히면: 설정 → 시스템 → 정보 → 고급 시스템 설정 → 환경 변수 → 사용자 `Path` 에 `ffmpeg.exe` 가 있는 `bin` 폴더 추가.
빌드(GPL/LGPL) 차이와 `libx264` 부재 시 폴백은 [THIRD_PARTY_LICENSES.md §3](../THIRD_PARTY_LICENSES.md).

## 5. PyTorch: NVIDIA 드라이버 / CUDA 버전과 wheel 선택
1. GPU 가 NVIDIA 인지 확인하고 드라이버 상태 확인:
   ```powershell
   nvidia-smi
   ```
   표 오른쪽 위의 `CUDA Version: XX.X` 는 **드라이버가 지원하는 최대 CUDA 버전**입니다.
2. **CUDA Toolkit 을 따로 설치할 필요는 없습니다.** PyTorch wheel 이 CUDA 런타임을 포함하는 것으로 알려져 있습니다 (미검증). 필요한 것은 충분히 새로운 NVIDIA 드라이버뿐입니다.
3. https://pytorch.org 의 "Get Started" 선택기에서 (Stable / Windows / Pip / Python / CUDA 버전)을 고르면 명령이 나옵니다. 형태는 다음과 같습니다 (**버전 태그 `cuXXX` 는 시기마다 달라지므로 선택기 값을 쓰세요**):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
   ```
   `cuXXX` 는 `nvidia-smi` 의 CUDA Version **이하**여야 합니다.
4. 확인:
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   `False` 면 CPU 전용 wheel 이 설치된 것입니다(`+cpu`). 제거 후 위 index-url 로 재설치하세요.

### CPU-only 대안 (NVIDIA GPU 없음)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
동작은 하지만 매우 느릴 수 있습니다 (속도 미측정). `--sam-model tiny --sam-max-side 768 --refine-max-side 720` 및 `--max-frames` 로 짧게 시험하세요. AMD/Intel GPU 는 지원하지 않습니다(CPU 로 동작).

## 6. 이 저장소 + SAM 2 설치
```powershell
pip install -e .
pip install "git+https://github.com/facebookresearch/sam2.git"
```

### Visual Studio Build Tools 가 필요한 경우 / 불필요한 경우
- **불필요 (대부분)**: 미리 빌드된 wheel 로 설치되는 PyTorch, numpy, opencv 등. 이 도구는 SAM 2 의 **선택적 CUDA 확장 없이도 동작**하도록 되어 있습니다(SAM 2 README: 확장 빌드 실패는 무시해도 됨; Linux CPU 에서 확장 없이 동작 확인).
- **필요할 수 있음**: SAM 2 의 CUDA 확장을 직접 빌드하고 싶을 때(CUDA Toolkit + "C++를 사용한 데스크톱 개발" 워크로드), 또는 wheel 이 없는 패키지를 소스로 빌드해야 할 때(오류에 `Microsoft Visual C++ 14.0 or greater is required` 가 보이면).
- 설치 로그의 `Failed to build the SAM 2 CUDA extension` 는 무시해도 됩니다.

## 7. 모델 다운로드
```powershell
python scripts/download_models.py
python scripts/download_models.py grounding-dino-tiny --allow-unverified     # 가중치 라이선스 미확인: 직접 확인 후
python scripts/download_models.py list
```
`models\` 에 저장됩니다(.gitignore 처리). 라이선스 상태는 [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).

## 8. 경로 길이 / 한글 경로
- 경로 처리는 `pathlib` 을 쓰고 ffmpeg 에는 인자 리스트로 전달하며, 이미지 읽기/쓰기는 OpenCV 의 파일 API 를 쓰지 않고 바이트 버퍼로 처리합니다. 그래서 한글/공백 경로를 코드에서 방어했고 **Linux 에서 한글·공백·작은따옴표 경로를 테스트했습니다. Windows 에서는 미검증입니다.**
- 그래도 문제가 생기면 영문 경로(예: `C:\work\videomatte`)로 옮겨 보세요.
- **260자 경로 제한**: 저장소를 깊은 폴더에 두거나 작업 폴더 이름이 길면 실패할 수 있습니다. 짧은 경로에 두세요. Windows 긴 경로 활성화(레지스트리 `LongPathsEnabled`, 관리자 권한 필요)와 `git config --global core.longpaths true` 를 쓸 수 있습니다. 작업 폴더는 `--work-dir C:\vm` 처럼 짧게 지정할 수 있습니다.
- 동영상 파일이 OneDrive/바탕화면 동기화 폴더에 있으면 동기화 잠금으로 실패할 수 있으니 로컬 폴더에서 처리하세요.

## 9. 동작 확인
```powershell
videomatte --doctor
pytest -q
videomatte input.mp4 --max-frames 30 --box 100,50,600,700 -o preview.mp4
```

## 10. 자주 나는 오류
| 증상 | 원인 / 조치 |
|---|---|
| `Activate.ps1 ... 실행할 수 없습니다` | 3단계의 `Set-ExecutionPolicy` 또는 cmd 사용 |
| `ffmpeg/ffprobe 를 PATH 에서 찾을 수 없습니다` | 새 PowerShell 로 재시작, PATH 확인 |
| `torch.cuda.is_available()` 가 False | CPU wheel 설치됨 / 드라이버 구버전 → 5단계 |
| `CUDA out of memory` | `--chunk-size` ↓, `--sam-max-side 768`, `--refine-max-side 720`, `--sam-model tiny`, 또는 `--device cpu` |
| `Microsoft Visual C++ 14.0 or greater is required` | 6단계의 Build Tools 항목 참고 |
| `pip install git+...` 실패 (`git` 없음) | 2단계 |
| 경로 관련 오류 | 8단계 |

더 많은 항목: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
