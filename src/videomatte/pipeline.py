"""전체 파이프라인: 검출 -> SAM 2 전파 -> 정밀화/안정화 -> 전경색/합성 -> 인코딩 (청크 단위, resume 가능)."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import composite as comp
from . import matting
from .config import Config
from .detect import Detection, Target, parse_box, parse_points, select_detections
from .errors import DetectionError, VideomatteError
from .io_video import (PRORES4444_ARGS, RawWriter, concat_and_mux, extract_jpegs, h264_args,
                       pick_h264_encoder, probe, read_frames)

log = logging.getLogger(__name__)


def _h(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _file_sig(path: Path) -> list:
    st = path.stat()
    return [str(path.resolve()), st.st_size, st.st_mtime_ns]


class Pipeline:
    """detector / segmenter / refiner 는 주입 가능 (테스트용). 없으면 필요할 때 지연 생성한다."""

    def __init__(self, cfg: Config, detector=None, segmenter=None, refiner=None):
        self.cfg = cfg
        self._detector, self._segmenter, self._refiner = detector, segmenter, refiner
        self._device: str | None = None

    # ------------------------------------------------------------ 지연 생성
    @property
    def device(self) -> str:
        if self._device is None:
            from .device import select_device

            self._device = select_device(self.cfg.device)
        return self._device

    @property
    def detector(self):
        if self._detector is None:
            from .detect import GroundingDinoDetector

            self._detector = GroundingDinoDetector(self.device, self.cfg.models_dir, self.cfg.dino_variant,
                                                   self.cfg.detector_threshold)
        return self._detector

    @property
    def segmenter(self):
        if self._segmenter is None:
            from .segment import Sam2Segmenter

            self._segmenter = Sam2Segmenter(self.device, self.cfg.sam_model, self.cfg.models_dir)
        return self._segmenter

    @property
    def refiner(self):
        if self._refiner is None:
            self._refiner = matting.build_refiner(self.cfg.refiner, self.device if self.cfg.refiner != "none" else "cpu",
                                                  self.cfg.models_dir, self.cfg.vitmatte_variant)
        return self._refiner

    # ------------------------------------------------------------ 메인
    def run(self) -> Path:
        cfg = self.cfg
        if cfg.input is None:
            raise VideomatteError("입력 영상을 지정하세요.")
        info = probe(cfg.input)
        out = cfg.output or cfg.input.with_name(cfg.input.stem + "_matte.mp4")
        if out.resolve() == cfg.input.resolve():
            raise VideomatteError("출력 경로가 입력과 같습니다.")
        total = info.n_frames if not cfg.max_frames else min(info.n_frames, cfg.max_frames)
        if total <= 0:
            raise VideomatteError("프레임 수를 알 수 없거나 0 입니다.")
        W, H = info.width // 2 * 2, info.height // 2 * 2  # yuv420p 는 짝수 해상도 필요
        n_chunks = -(-total // cfg.chunk_size)

        mask_key = _h(_file_sig(cfg.input), cfg.chunk_size, cfg.sam_model, cfg.sam_max_side, cfg.prompt, cfg.box,
                      cfg.points, cfg.select, total, cfg.detector_threshold)
        bg = comp.parse_background(cfg.bg)
        bg_sig = _file_sig(Path(cfg.bg)) if Path(cfg.bg).is_file() else cfg.bg
        encoder = pick_h264_encoder(cfg.encoder)
        render_key = _h(mask_key, cfg.refiner, cfg.vitmatte_variant, cfg.trimap_width, cfg.refine_max_side,
                        cfg.temporal_smooth, cfg.fg_estimate, cfg.despill, cfg.despill_strength, bg_sig, cfg.feather,
                        cfg.color_match, cfg.export_alpha, encoder, cfg.crf)
        work = (cfg.work_dir or out.parent / ".videomatte_work") / f"{cfg.input.stem}-{mask_key}"
        if not cfg.resume and work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        alpha_png_dir = out.with_name(out.stem + "_alpha")
        if cfg.export_alpha in ("png", "both"):
            alpha_png_dir.mkdir(parents=True, exist_ok=True)
        log.info("입력 %dx%d @ %.3ffps, %d프레임, %d청크 (작업 폴더: %s)", info.width, info.height, info.fps, total,
                 n_chunks, work)

        done_frames = sum(self._chunk_len(k, total) for k in range(n_chunks) if self._is_done(work, k, render_key))
        if done_frames:
            log.info("재개: %d/%d 프레임은 이미 처리됨", done_frames, total)
        bar = tqdm(total=total, initial=done_frames, unit="frame", desc="matting", disable=None)
        target: Target | None = None
        seg_paths, alpha_paths = [], []
        try:
            for k in range(n_chunks):
                start, count = k * cfg.chunk_size, self._chunk_len(k, total)
                seg, aseg = work / f"seg_{k:05d}.mp4", work / f"seg_{k:05d}_alpha.mov"
                seg_paths.append(seg)
                alpha_paths.append(aseg)
                if self._is_done(work, k, render_key):
                    continue
                holder = {"t": target}
                masks = self._masks_for_chunk(info, work, k, start, count, holder)
                target = holder["t"]
                self._render_chunk(info, k, start, count, masks, bg, W, H, work, seg, aseg, alpha_png_dir, encoder, bar)
                (work / f"chunk_{k:05d}.done").write_text(render_key)
        finally:
            bar.close()

        concat_and_mux(seg_paths, info.path if info.has_audio else None, out, work)
        if cfg.export_alpha in ("prores", "both"):
            ap = out.with_name(out.stem + "_alpha.mov")
            concat_and_mux(alpha_paths, None, ap, work)
            log.info("알파 ProRes 4444: %s", ap)
        if cfg.export_alpha in ("png", "both"):
            log.info("알파 PNG 시퀀스: %s", alpha_png_dir)
        if not cfg.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        log.info("완료: %s", out)
        return out

    # ------------------------------------------------------------ 보조
    def _chunk_len(self, k: int, total: int) -> int:
        return min(self.cfg.chunk_size, total - k * self.cfg.chunk_size)

    def _is_done(self, work: Path, k: int, render_key: str) -> bool:
        marker = work / f"chunk_{k:05d}.done"
        needs_alpha = self.cfg.export_alpha in ("prores", "both")
        return (marker.exists() and marker.read_text() == render_key and (work / f"seg_{k:05d}.mp4").exists()
                and (not needs_alpha or (work / f"seg_{k:05d}_alpha.mov").exists()))

    def _resolve_target(self, info, work: Path) -> Target:
        cfg = self.cfg
        cache = work / "target.json"
        if cache.exists():
            return Target.from_json(json.loads(cache.read_text()))
        if cfg.box or cfg.points:
            t = Target()
            if cfg.box:
                t.boxes = [parse_box(cfg.box)]
            if cfg.points:
                t.points, t.point_labels = parse_points(cfg.points)
        else:
            frame0 = next(read_frames(info, 0, 1), None)
            if frame0 is None:
                raise VideomatteError("첫 프레임을 읽을 수 없습니다.")
            dets: list[Detection] = self.detector.detect(frame0, cfg.prompt)
            log.info("'%s' 검출 %d개: %s", cfg.prompt, len(dets),
                     [(d.label, round(d.score, 2), [round(v) for v in d.box]) for d in dets])
            chosen = select_detections(dets, cfg.select)
            if not chosen:
                raise DetectionError(
                    f"첫 프레임에서 '{cfg.prompt}' 를 찾지 못했습니다. --detector-threshold 를 낮추거나 "
                    "--box x1,y1,x2,y2 / --points 로 직접 지정하세요."
                )
            t = Target(boxes=[d.box for d in chosen])
        cache.write_text(json.dumps(t.to_json()))
        return t

    def _masks_for_chunk(self, info, work: Path, k: int, start: int, count: int, holder: dict) -> np.ndarray:
        cfg = self.cfg
        mpath = work / f"masks_{k:05d}.npz"
        if mpath.exists():
            return np.load(mpath)["m"]
        seed = None
        if k > 0:
            prev = work / f"masks_{k - 1:05d}.npz"
            if not prev.exists():
                raise VideomatteError(f"이전 청크 마스크 캐시가 없습니다: {prev} (--no-resume 로 다시 실행)")
            seed = np.load(prev)["m"][-1]
        target = None
        if k == 0:
            holder["t"] = target = self._resolve_target(info, work)
        frames_dir = work / "frames_sam"
        n, sw, sh = extract_jpegs(info, start, count, frames_dir, cfg.sam_max_side)
        if n == 0:
            raise VideomatteError(f"청크 {k}: 프레임을 추출하지 못했습니다.")
        if k == 0:
            target = target.scaled(sw / info.width, sh / info.height)
        elif seed.shape != (sh, sw):
            seed = cv2.resize(seed.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks = self.segmenter.propagate(frames_dir, n, target, seed)
        tmp = work / f"masks_{k:05d}.tmp.npz"
        np.savez_compressed(tmp, m=masks)
        tmp.replace(mpath)
        shutil.rmtree(frames_dir, ignore_errors=True)
        return masks

    def _render_chunk(self, info, k, start, count, masks, bg, W, H, work, seg, aseg, png_dir, encoder, bar) -> None:
        cfg = self.cfg
        smoother = matting.TemporalSmoother(cfg.temporal_smooth)
        last = work / f"alpha_last_{k - 1:05d}.npy"
        if k > 0 and last.exists():
            smoother.reset(np.load(last).astype(np.float32))
        want_prores = cfg.export_alpha in ("prores", "both")
        want_png = cfg.export_alpha in ("png", "both")
        bg_iter = bg.frames(W, H, info.fps_frac, start, info.fps)
        writers = [RawWriter(seg, W, H, info.fps_frac, "rgb24", h264_args(encoder, cfg.crf))]
        if want_prores:
            writers.append(RawWriter(aseg, W, H, info.fps_frac, "rgba", PRORES4444_ARGS))
        ok = False
        try:
            n_read = 0
            for i, frame in enumerate(read_frames(info, start, count)):
                frame = frame[:H, :W]
                mask = masks[min(i, len(masks) - 1)]
                alpha = self._alpha(frame, mask)
                alpha = smoother(alpha)
                alpha = comp.feather_alpha(alpha, cfg.feather)
                fg = comp.estimate_foreground(frame, alpha, cfg.fg_estimate)
                fg = comp.despill(fg, cfg.despill, cfg.despill_strength)
                bgf = next(bg_iter)
                if bgf is None:
                    bgf = comp.blur_background(frame)
                fg = comp.color_match(fg, alpha, bgf, cfg.color_match)
                out = comp.compose(fg, alpha, bgf)
                writers[0].write((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
                a8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
                if want_prores:
                    writers[1].write(np.dstack([(np.clip(fg, 0, 1) * 255 + 0.5).astype(np.uint8), a8]))
                if want_png:
                    cv2.imwrite(str(png_dir / f"{start + i:06d}.png"), a8)
                n_read += 1
                bar.update(1)
            if n_read == 0:
                raise VideomatteError(f"청크 {k}: 원본에서 프레임을 읽지 못했습니다.")
            if n_read < count:
                log.warning("청크 %d: 예상 %d프레임 중 %d프레임만 디코딩되었습니다.", k, count, n_read)
                bar.total -= count - n_read
            if smoother.prev is not None:
                np.save(work / f"alpha_last_{k:05d}.npy", smoother.prev.astype(np.float16))
            ok = True
        finally:
            for w_ in writers:
                w_.close() if ok else w_.abort()
        # bg 제너레이터 정리 (ffmpeg 서브프로세스 종료)
        close = getattr(bg_iter, "close", None)
        if close:
            close()

    def _alpha(self, frame: np.ndarray, mask_small: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        H, W = frame.shape[:2]
        side = cfg.refine_max_side
        scale = side / max(H, W) if side and max(H, W) > side else 1.0
        rh, rw = (round(H * scale), round(W * scale)) if scale < 1 else (H, W)
        img = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        m = cv2.resize(mask_small.astype(np.float32), (rw, rh), interpolation=cv2.INTER_LINEAR)
        tw = cfg.trimap_width or matting.auto_trimap_width(rh, rw)
        tri = matting.make_trimap(m, tw, tw)
        alpha = self.refiner.refine(img, m, tri)
        if scale < 1:
            alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        return alpha.astype(np.float32)
