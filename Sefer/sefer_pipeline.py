
from __future__ import annotations

import io
import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
VALID_PATTERN = re.compile(r"^\d{1,4}[A-E]?$")
AMBIGUOUS_MAP = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
}
ALLOWED_SUFFIX_LETTERS = set("ABCDE")


@dataclass
class PipelineConfig:
    detector_weights: str
    input_dir: str
    output_dir: str = "ocr_pipeline_output"
    detect_img_size: int = 1280
    detect_conf: float = 0.10
    detect_iou: float = 0.50
    detect_device: str = "0"
    max_detections: int = 5

    paddle_det_model_dir: Optional[str] = None
    paddle_rec_model_dir: Optional[str] = None
    paddle_cls_model_dir: Optional[str] = None
    paddle_lang: str = "en"

    ocr_target_height: int = 160
    ocr_fast_target_height: int = 128
    ocr_early_accept_conf: float = 0.97
    ocr_low_conf_trigger: float = 0.80
    ocr_use_adaptive_variants: bool = True

    ocr_high_conf_threshold: float = 0.90
    zero_pad_to: int = 4
    keep_original_name: bool = False
    user_prefix: str = ""
    low_conf_mark: str = "!_"

    apply_rename: bool = False
    rename_mode: str = "copy"  # copy or move

    save_crops: bool = False
    save_prepared_crops: bool = False
    save_debug_vis: bool = False

    @property
    def input_path(self) -> Path:
        return Path(self.input_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def crops_dir(self) -> Path:
        return self.output_path / "crops"

    @property
    def prepared_dir(self) -> Path:
        return self.output_path / "prepared_crops"

    @property
    def vis_dir(self) -> Path:
        return self.output_path / "visualizations"

    @property
    def renamed_dir(self) -> Path:
        return self.output_path / "renamed_files"

    @property
    def report_csv(self) -> Path:
        return self.output_path / "ocr_report.csv"

    @property
    def report_xlsx(self) -> Path:
        return self.output_path / "ocr_report.xlsx"

    @property
    def report_json(self) -> Path:
        return self.output_path / "ocr_report.json"

    @property
    def rename_log_csv(self) -> Path:
        return self.output_path / "rename_log.csv"


@dataclass
class DetectionResult:
    found: bool
    det_conf: float
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    crop_path: Optional[str]
    crop_bgr: Optional[np.ndarray]
    debug_path: Optional[str]
    message: str


@dataclass
class OCRResult:
    raw_text: str
    clean_text: str
    ocr_conf: float
    variant_name: str
    valid: bool
    message: str


class SeferPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._detector: Optional[YOLO] = None
        self._ocr: Optional[PaddleOCR] = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def ensure_dirs(self) -> None:
        self.cfg.output_path.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_crops:
            self.cfg.crops_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_prepared_crops:
            self.cfg.prepared_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_debug_vis:
            self.cfg.vis_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.renamed_dir.mkdir(parents=True, exist_ok=True)

    def collect_images(self) -> List[Path]:
        images_dir = self.cfg.input_path
        if not images_dir.exists():
            raise FileNotFoundError(f"Папка с изображениями не найдена: {images_dir.resolve()}")
        files = [p for p in images_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS and p.is_file()]
        return sorted(files)

    def load_models(self) -> None:
        if self._detector is None:
            weights = Path(self.cfg.detector_weights)
            if not weights.exists():
                raise FileNotFoundError(f"Не найдены веса детектора: {weights}")
            self._detector = YOLO(str(weights))
        if self._ocr is None:
            self._ocr = self.build_paddle_ocr()

    def build_paddle_ocr(self) -> PaddleOCR:
        ocr_kwargs = {
            "lang": self.cfg.paddle_lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if self.cfg.paddle_det_model_dir:
            ocr_kwargs["text_detection_model_dir"] = str(self.cfg.paddle_det_model_dir)
        if self.cfg.paddle_rec_model_dir:
            ocr_kwargs["text_recognition_model_dir"] = str(self.cfg.paddle_rec_model_dir)
        if self.cfg.paddle_cls_model_dir and ocr_kwargs["use_textline_orientation"]:
            ocr_kwargs["textline_orientation_model_dir"] = str(self.cfg.paddle_cls_model_dir)
        return PaddleOCR(**ocr_kwargs)

    def to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        return self._clahe.apply(gray)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

    def sharpen(self, gray: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(gray, -1, kernel)

    def resize_keep_aspect(self, img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    def rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos = abs(M[0, 0]); sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def make_ocr_variants(self, crop_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        base_gray = self.to_gray(crop_bgr)
        h, w = base_gray.shape[:2]
        target_h = self.cfg.ocr_fast_target_height if max(h, w) >= 220 else self.cfg.ocr_target_height
        base_gray = self.resize_keep_aspect(base_gray, target_h=target_h)
        gray_clahe = self.apply_clahe(base_gray)
        gray_denoise = self.denoise(base_gray)
        gray_combo = self.sharpen(self.apply_clahe(gray_denoise))
        primary = [
            ("gray_combo", gray_combo),
            ("gray", base_gray),
            ("gray_clahe", gray_clahe),
        ]
        if not self.cfg.ocr_use_adaptive_variants:
            extra = [("gray_denoise", gray_denoise), ("gray_sharp", self.sharpen(base_gray))]
            for angle in (-6, -3, 3, 6):
                extra.append((f"rot_{angle:+d}", self.rotate_image(gray_combo, angle)))
            return primary + extra
        return primary

    def make_fallback_ocr_variants(self, crop_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        base_gray = self.to_gray(crop_bgr)
        base_gray = self.resize_keep_aspect(base_gray, target_h=self.cfg.ocr_target_height)
        gray_denoise = self.denoise(base_gray)
        gray_combo = self.sharpen(self.apply_clahe(gray_denoise))
        return [
            ("gray_denoise", gray_denoise),
            ("gray_sharp", self.sharpen(base_gray)),
            ("rot_-3", self.rotate_image(gray_combo, -3)),
            ("rot_+3", self.rotate_image(gray_combo, 3)),
            ("rot_-6", self.rotate_image(gray_combo, -6)),
            ("rot_+6", self.rotate_image(gray_combo, 6)),
        ]

    def cleanup_text(self, raw_text: str) -> str:
        text = raw_text.upper().strip()
        text = re.sub(r"[^0-9A-Z]", "", text)
        if not text:
            return ""
        chars = list(text)
        for i, ch in enumerate(chars):
            is_last = (i == len(chars) - 1)
            if is_last and ch in ALLOWED_SUFFIX_LETTERS:
                continue
            chars[i] = AMBIGUOUS_MAP.get(ch, ch)
        text = "".join(chars)
        if len(text) > 1:
            core = re.sub(r"[^0-9]", "", text[:-1])
            last = text[-1]
            if last in ALLOWED_SUFFIX_LETTERS or last.isdigit():
                text = core + last
            else:
                text = core
        m = re.match(r"^(\d{1,4})([A-E]?)$", text)
        return f"{m.group(1)}{m.group(2)}" if m else ""

    def is_valid_plate_number(self, text: str) -> bool:
        return bool(VALID_PATTERN.fullmatch(text))

    def split_number_suffix(self, text: str) -> Tuple[str, str]:
        m = re.match(r"^(\d{1,4})([A-E]?)$", text)
        if not m:
            return "", ""
        return m.group(1), m.group(2)

    def canonical_plate_name(self, text: str) -> str:
        digits, suffix = self.split_number_suffix(text)
        if not digits:
            raise ValueError(f"Невалидный номер: {text}")
        return digits.zfill(self.cfg.zero_pad_to) + suffix

    def choose_best_box(self, result, image_shape: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        if result.boxes is None or len(result.boxes) == 0:
            return None
        h, w = image_shape[:2]
        best = None
        best_score = -1.0
        for box in result.boxes:
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            area = max(1, (x2 - x1) * (y2 - y1))
            score = conf + 1e-8 * area
            if score > best_score:
                best_score = score
                best = ((x1, y1, x2, y2), conf)
        return best

    def detect_and_crop_table(self, image_path: Path) -> DetectionResult:
        assert self._detector is not None
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return DetectionResult(False, 0.0, None, None, None, None, "image_read_error")

        pred = self._detector.predict(
            source=img_bgr,
            imgsz=self.cfg.detect_img_size,
            conf=self.cfg.detect_conf,
            iou=self.cfg.detect_iou,
            device=self.cfg.detect_device,
            max_det=self.cfg.max_detections,
            verbose=False,
        )
        if not pred:
            return DetectionResult(False, 0.0, None, None, None, None, "no_prediction_object")
        result = pred[0]
        selected = self.choose_best_box(result, img_bgr.shape)
        if selected is None:
            return DetectionResult(False, 0.0, None, None, None, None, "not_found")

        (x1, y1, x2, y2), det_conf = selected
        crop = img_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return DetectionResult(False, det_conf, (x1, y1, x2, y2), None, None, None, "empty_crop")

        crop_path = None
        if self.cfg.save_crops:
            crop_path = str((self.cfg.crops_dir / f"{image_path.stem}_crop.jpg").resolve())
            cv2.imwrite(crop_path, crop)

        debug_path = None
        if self.cfg.save_debug_vis:
            debug = img_bgr.copy()
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, f"table {det_conf:.3f}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            debug_path = str((self.cfg.vis_dir / f"{image_path.stem}_det_debug.jpg").resolve())
            cv2.imwrite(debug_path, debug)

        return DetectionResult(True, det_conf, (x1, y1, x2, y2), crop_path, crop, debug_path, "ok")

    def run_paddle_on_image(self, img: np.ndarray) -> Tuple[str, float]:
        assert self._ocr is not None
        result = self._ocr.predict(img)
        if result is None:
            return "", 0.0
        try:
            result_list = list(result)
        except TypeError:
            result_list = [result]
        if not result_list:
            return "", 0.0

        texts, scores = [], []
        for item in result_list:
            rec_texts = getattr(item, "rec_texts", None)
            rec_scores = getattr(item, "rec_scores", None)
            if rec_texts is None and isinstance(item, dict):
                rec_texts = item.get("rec_texts") or item.get("texts")
            if rec_scores is None and isinstance(item, dict):
                rec_scores = item.get("rec_scores") or item.get("scores")
            if rec_texts:
                texts.extend([str(t) for t in rec_texts if str(t).strip()])
            if rec_scores:
                for s in rec_scores:
                    try:
                        scores.append(float(s))
                    except Exception:
                        pass
        if not texts:
            return "", 0.0
        return "".join(texts).strip(), float(np.mean(scores)) if scores else 0.0

    def _save_prepared_variant(self, stem: str, variant_name: str, variant_img: np.ndarray) -> None:
        if not self.cfg.save_prepared_crops:
            return
        save_path = self.cfg.prepared_dir / f"{stem}_{variant_name}.jpg"
        if variant_img.ndim == 2:
            cv2.imwrite(str(save_path), variant_img)
        else:
            cv2.imwrite(str(save_path), cv2.cvtColor(variant_img, cv2.COLOR_BGR2RGB))

    def _pick_better_ocr_candidate(self, best: OCRResult, candidate: OCRResult) -> OCRResult:
        candidate_rank = (1 if candidate.valid else 0, candidate.ocr_conf)
        best_rank = (1 if best.valid else 0, best.ocr_conf)
        return candidate if candidate_rank > best_rank else best

    def _evaluate_variants(self, variants: List[Tuple[str, np.ndarray]], stem_for_debug: str) -> OCRResult:
        best = OCRResult("", "", 0.0, "", False, "ocr_failed")
        for variant_name, variant_img in variants:
            variant_bgr = cv2.cvtColor(variant_img, cv2.COLOR_GRAY2BGR) if variant_img.ndim == 2 else variant_img
            raw_text, score = self.run_paddle_on_image(variant_bgr)
            clean_text = self.cleanup_text(raw_text)
            valid = self.is_valid_plate_number(clean_text)
            self._save_prepared_variant(stem_for_debug, variant_name, variant_img)
            candidate = OCRResult(
                raw_text=raw_text,
                clean_text=clean_text,
                ocr_conf=score,
                variant_name=variant_name,
                valid=valid,
                message="ok" if valid else "invalid_pattern",
            )
            best = self._pick_better_ocr_candidate(best, candidate)
            if candidate.valid and candidate.ocr_conf >= self.cfg.ocr_early_accept_conf:
                return candidate
        return best

    def recognize_plate_from_crop(self, crop_bgr: np.ndarray, crop_stem: str) -> OCRResult:
        if crop_bgr is None or crop_bgr.size == 0:
            return OCRResult("", "", 0.0, "", False, "crop_read_error")
        primary_variants = self.make_ocr_variants(crop_bgr)
        best = self._evaluate_variants(primary_variants, crop_stem)
        need_fallback = (not best.valid) or (best.ocr_conf < self.cfg.ocr_low_conf_trigger)
        if need_fallback:
            fallback = self.make_fallback_ocr_variants(crop_bgr)
            fallback_best = self._evaluate_variants(fallback, crop_stem)
            best = self._pick_better_ocr_candidate(best, fallback_best)
        return best

    def decide_status(self, det: DetectionResult, ocr: Optional[OCRResult]) -> str:
        if not det.found:
            return "not_found"
        if ocr is None or not ocr.valid:
            return "ocr_invalid"
        if ocr.ocr_conf >= self.cfg.ocr_high_conf_threshold:
            return "ok"
        return "low_conf"

    def build_target_base_name(self, clean_text: str) -> str:
        base = self.canonical_plate_name(clean_text)
        return f"{self.cfg.user_prefix}{base}" if self.cfg.user_prefix else base

    def attach_duplicate_suffixes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        counters: Dict[str, int] = {}
        final_names = []
        for _, row in df.iterrows():
            proposed = row["proposed_stem"]
            suffix = row["ext"]
            if not proposed:
                final_names.append("")
                continue
            cnt = counters.get(proposed, 0)
            if cnt == 0:
                final_names.append(proposed + suffix)
            else:
                final_names.append(f"{proposed}_{cnt}{suffix}")
            counters[proposed] = cnt + 1
        df["final_filename"] = final_names
        return df

    def build_final_name(self, original_name: str, proposed_filename: str, status: str) -> str:
        if not proposed_filename:
            return f"{self.cfg.low_conf_mark}{original_name}" if status != "ok" else original_name
        if self.cfg.keep_original_name:
            stem = Path(original_name).stem
            ext = Path(original_name).suffix
            final = f"{Path(proposed_filename).stem}_{stem}{ext}"
        else:
            final = proposed_filename
        if status in {"low_conf", "not_found", "ocr_invalid"}:
            final = self.cfg.low_conf_mark + final
        return final

    def process_images_folder(
        self,
        progress_callback: Optional[Callable[[int, int, Path, Optional[str]], None]] = None,
    ) -> pd.DataFrame:
        self.ensure_dirs()
        self.load_models()
        image_paths = self.collect_images()
        if not image_paths:
            raise RuntimeError(f"В папке нет поддерживаемых изображений: {self.cfg.input_path.resolve()}")

        rows = []
        total = len(image_paths)
        for i, image_path in enumerate(image_paths, start=1):
            det = self.detect_and_crop_table(image_path)
            ocr = self.recognize_plate_from_crop(det.crop_bgr, image_path.stem) if det.found and det.crop_bgr is not None else None
            status = self.decide_status(det, ocr)
            clean_text = ocr.clean_text if ocr else ""
            proposed_stem = self.build_target_base_name(clean_text) if clean_text and self.is_valid_plate_number(clean_text) else ""
            rows.append({
                "input_path": str(image_path.resolve()),
                "input_filename": image_path.name,
                "stem": image_path.stem,
                "ext": image_path.suffix,
                "det_found": det.found,
                "det_conf": det.det_conf,
                "bbox_xyxy": det.bbox_xyxy,
                "crop_path": det.crop_path,
                "debug_path": det.debug_path,
                "det_message": det.message,
                "ocr_raw_text": ocr.raw_text if ocr else "",
                "ocr_clean_text": clean_text,
                "ocr_conf": ocr.ocr_conf if ocr else 0.0,
                "ocr_variant": ocr.variant_name if ocr else "",
                "ocr_valid": (ocr.valid if ocr else False),
                "ocr_message": (ocr.message if ocr else "ocr_skipped"),
                "status": status,
                "proposed_stem": proposed_stem,
            })
            if progress_callback:
                progress_callback(i, total, image_path, status)
        df = pd.DataFrame(rows)
        df = self.attach_duplicate_suffixes(df)
        df["final_name"] = df.apply(
            lambda r: self.build_final_name(r["input_filename"], r["final_filename"], r["status"]), axis=1
        )
        return df

    def safe_copy_or_move_file(self, src: Path, dst: Path, mode: str = "copy") -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        if dst.exists():
            raise FileExistsError(f"Файл назначения уже существует: {dst}")
        mode = mode.lower().strip()
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "move":
            shutil.move(str(src), str(dst))
        else:
            raise ValueError('rename_mode должен быть "copy" или "move"')

    def apply_rename_plan(self, df: pd.DataFrame) -> pd.DataFrame:
        renamed = []
        self.cfg.renamed_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            src = Path(row["input_path"])
            dst = self.cfg.renamed_dir / row["final_name"]
            try:
                self.safe_copy_or_move_file(src, dst, mode=self.cfg.rename_mode)
                renamed.append((str(src), str(dst), self.cfg.rename_mode, "ok"))
            except Exception as e:
                renamed.append((str(src), str(dst), self.cfg.rename_mode, f"error: {e}"))
        rename_log = pd.DataFrame(renamed, columns=["src", "dst", "mode", "rename_status"])
        rename_log.to_csv(self.cfg.rename_log_csv, index=False, encoding="utf-8-sig")
        return rename_log

    def build_summary(self, df: pd.DataFrame) -> Dict[str, object]:
        return {
            "images_dir": str(self.cfg.input_path.resolve()),
            "detector_weights": str(Path(self.cfg.detector_weights).resolve()),
            "output_dir": str(self.cfg.output_path.resolve()),
            "n_files": int(len(df)),
            "n_ok": int((df["status"] == "ok").sum()),
            "n_low_conf": int((df["status"] == "low_conf").sum()),
            "n_not_found": int((df["status"] == "not_found").sum()),
            "n_ocr_invalid": int((df["status"] == "ocr_invalid").sum()),
            "ocr_high_conf_threshold": self.cfg.ocr_high_conf_threshold,
            "detect_conf": self.cfg.detect_conf,
            "keep_original_name": self.cfg.keep_original_name,
            "user_prefix": self.cfg.user_prefix,
            "apply_rename": self.cfg.apply_rename,
            "rename_mode": self.cfg.rename_mode,
        }

    def save_reports(self, df: pd.DataFrame, rename_log_df: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
        self.cfg.output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cfg.report_csv, index=False, encoding="utf-8-sig")

        try:
            with pd.ExcelWriter(self.cfg.report_xlsx, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="all_results")
                df[df["status"] == "ok"].to_excel(writer, index=False, sheet_name="ok")
                df[df["status"] == "low_conf"].to_excel(writer, index=False, sheet_name="low_conf")
                df[df["status"].isin(["not_found", "ocr_invalid"])].to_excel(writer, index=False, sheet_name="manual_review")
                if rename_log_df is not None:
                    rename_log_df.to_excel(writer, index=False, sheet_name="rename_log")
            xlsx_saved = True
        except Exception:
            xlsx_saved = False

        summary = self.build_summary(df)
        if rename_log_df is not None:
            summary["rename_log_rows"] = int(len(rename_log_df))
        self.cfg.report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        out = {"csv": self.cfg.report_csv, "json": self.cfg.report_json}
        if xlsx_saved:
            out["xlsx"] = self.cfg.report_xlsx
        if rename_log_df is not None:
            out["rename_log_csv"] = self.cfg.rename_log_csv
        return out


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def dataframe_to_xlsx_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=name[:31] or "sheet")
    return buffer.getvalue()
