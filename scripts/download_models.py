#!/usr/bin/env python
"""모델 가중치 다운로드 (models/ 에 저장, 체크섬 검증).

체크섬 정책:
  * ModelSpec.sha256 이 고정되어 있으면 그 값과 비교한다.
  * 고정값이 없으면(현재 대부분) 최초 다운로드 시 SHA-256 을 models/checksums.json 에 기록하고,
    이후 실행/--verify 에서 그 값과 비교한다 (TOFU: 최초 신뢰). 고정 해시가 아님을 유의.

라이선스가 '미확인'인 모델은 --allow-unverified 없이는 받지 않는다.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from videomatte.models import MODELS, load_checksums, models_dir, save_checksums, sha256_file

DEFAULTS = ["sam2.1_hiera_small"]


def _download_url(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def _download_hf(repo: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=str(dest))
    shutil.rmtree(dest / ".cache", ignore_errors=True)


def _digest(path: Path) -> str:
    """파일이면 파일 해시, 디렉터리면 (상대경로, 파일해시) 목록의 해시."""
    if path.is_file():
        return sha256_file(path)
    import hashlib

    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"모델 이름 (기본: {DEFAULTS}). 'list' 로 목록 출력")
    ap.add_argument("--models-dir", help="저장 위치 (기본: 저장소 models/)")
    ap.add_argument("--allow-unverified", action="store_true", help="가중치 라이선스 미확인 모델도 받는다 (본인 책임으로 확인)")
    ap.add_argument("--verify", action="store_true", help="다운로드 없이 기록된 체크섬과 대조")
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 받는다")
    args = ap.parse_args()

    if args.names == ["list"]:
        for s in MODELS.values():
            flag = "verified" if s.license_verified else "UNVERIFIED"
            print(f"{s.name:24s} {flag:10s} {s.license}")
        return 0

    root = models_dir(args.models_dir)
    root.mkdir(parents=True, exist_ok=True)
    sums = load_checksums(root)
    names = args.names or DEFAULTS
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"알 수 없는 모델: {unknown}. 목록: python scripts/download_models.py list", file=sys.stderr)
        return 2

    rc = 0
    for name in names:
        spec = MODELS[name]
        dest = root / spec.local
        if args.verify:
            if not dest.exists():
                print(f"[MISSING] {name}"); rc = 1; continue
            expected = spec.sha256 or sums.get(name)
            if not expected:
                print(f"[NO-CHECKSUM] {name}: 기록된 체크섬이 없습니다"); rc = 1; continue
            ok = _digest(dest) == expected
            print(f"[{'OK' if ok else 'MISMATCH'}] {name}")
            rc |= 0 if ok else 1
            continue

        if not spec.license_verified and not args.allow_unverified:
            print(f"[SKIP] {name}: 가중치 라이선스 미확인 ({spec.license}). {spec.note}\n"
                  f"       상업 이용 전 직접 확인 후 --allow-unverified 로 다시 실행하세요.")
            rc = 1
            continue
        if dest.exists() and not args.force:
            print(f"[EXISTS] {name} -> {dest}")
        else:
            print(f"[DOWNLOAD] {name}")
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            try:
                (_download_url if spec.kind == "url" else _download_hf)(spec.source, dest)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {name}: {e}", file=sys.stderr)
                rc = 1
                continue
        digest = _digest(dest)
        expected = spec.sha256 or sums.get(name)
        if expected and expected != digest:
            print(f"[CHECKSUM MISMATCH] {name}: 기대 {expected[:12]}…, 실제 {digest[:12]}… (--force 로 재다운로드)", file=sys.stderr)
            rc = 1
            continue
        if not expected:
            sums[name] = digest
            save_checksums(root, sums)
            print(f"  체크섬 최초 기록(TOFU): {digest[:16]}…")
        print(f"[OK] {name}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
