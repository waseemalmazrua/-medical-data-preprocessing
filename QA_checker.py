"""
medical_seg_qa_pipeline.py
==========================
Production-ready, research-grade Quality Assurance (QA) pipeline for
3D medical image segmentation datasets.

Supports:
  - Generic  : root/images/ + root/labels/
  - BraTS    : BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/<case>/
  - nnU-Net  : nnUNet_raw/DatasetXXX/imagesTr/ + labelsTr/

Author : Senior Medical Imaging ML Engineer
"""

from __future__ import annotations

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("MedSegQA")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CaseRecord:
    """Represents one segmentation case (image + label pair)."""
    case_id: str
    image_paths: List[Path]          # one per modality
    label_path: Path
    modality_tags: List[str] = field(default_factory=list)


@dataclass
class StructuralResult:
    case_id: str
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    size_match: bool = True
    spacing_match: bool = True
    origin_match: bool = True
    direction_match: bool = True
    header_valid: bool = True
    missing_modalities: List[str] = field(default_factory=list)


@dataclass
class SemanticResult:
    case_id: str
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    unique_labels: List[int] = field(default_factory=list)
    is_empty_mask: bool = False
    missing_required_classes: List[int] = field(default_factory=list)
    unexpected_labels: List[int] = field(default_factory=list)
    has_float_contamination: bool = False


@dataclass
class DistributionResult:
    case_id: str
    voxel_count_per_class: Dict[int, int] = field(default_factory=dict)
    voxel_ratio_per_class: Dict[int, float] = field(default_factory=dict)
    # BraTS sub-regions (None if not BraTS)
    wt_voxels: Optional[int] = None
    tc_voxels: Optional[int] = None
    et_voxels: Optional[int] = None


@dataclass
class DatasetSummary:
    total_cases: int = 0
    failed_structural: int = 0
    empty_masks: int = 0
    missing_class_cases: int = 0
    skipped_corrupted: int = 0
    multi_class_cases: int = 0
    dataset_format: str = "unknown"
    class_presence_stats: Dict[int, int] = field(default_factory=dict)   # label -> n_cases present
    class_imbalance_score: float = 0.0
    split_safety_warnings: List[str] = field(default_factory=list)
    case_structural_results: List[Dict] = field(default_factory=list)
    case_semantic_results: List[Dict] = field(default_factory=list)
    case_distribution_results: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset format detection & loading
# ---------------------------------------------------------------------------

class DatasetDetector:
    """Detects and loads cases from a dataset root directory."""

    BRATS_MODALITIES = ["flair", "t1ce", "t1", "t2"]
    BRATS_SEG_SUFFIX = "_seg.nii.gz"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def detect_format(self) -> str:
        """Return one of 'generic', 'brats', 'nnunet'."""
        # nnU-Net: contains imagesTr or labelsTr
        if (self.root / "imagesTr").exists() or (self.root / "labelsTr").exists():
            return "nnunet"
        # BraTS: look for nested subject dirs with _seg.nii.gz
        candidate = self._find_brats_root()
        if candidate is not None:
            return "brats"
        # Generic fallback
        if (self.root / "images").exists() or (self.root / "labels").exists():
            return "generic"
        # Deep search
        return "generic"

    def _find_brats_root(self) -> Optional[Path]:
        """Return the directory that contains BraTS subject folders, or None."""
        for d in [self.root, *self.root.iterdir() if self.root.is_dir() else []]:
            if not isinstance(d, Path) or not d.is_dir():
                continue
            subdirs = [x for x in d.iterdir() if x.is_dir()]
            for sd in subdirs[:5]:
                segs = list(sd.glob("*_seg.nii.gz"))
                if segs:
                    return d
        return None

    def load_cases(self, fmt: Optional[str] = None) -> List[CaseRecord]:
        fmt = fmt or self.detect_format()
        if fmt == "brats":
            return self._load_brats()
        elif fmt == "nnunet":
            return self._load_nnunet()
        else:
            return self._load_generic()

    def _load_generic(self) -> List[CaseRecord]:
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"
        if not images_dir.exists():
            images_dir = self.root
        if not labels_dir.exists():
            labels_dir = self.root

        image_files = sorted(
            f for f in images_dir.rglob("*.nii*") if "label" not in f.name.lower()
        )
        label_files = sorted(
            f for f in labels_dir.rglob("*.nii*") if "label" in f.name.lower()
                                                      or labels_dir != images_dir
        )

        # Try to pair by stem
        label_map: Dict[str, Path] = {}
        for lf in label_files:
            stem = lf.name.replace(".nii.gz", "").replace(".nii", "")
            label_map[stem] = lf

        cases: List[CaseRecord] = []
        for img in image_files:
            stem = img.name.replace(".nii.gz", "").replace(".nii", "")
            # try exact match, then prefix match
            lbl = label_map.get(stem)
            if lbl is None:
                for k, v in label_map.items():
                    if k.startswith(stem) or stem.startswith(k):
                        lbl = v
                        break
            if lbl is None:
                logger.warning("No label found for image %s — skipping", img.name)
                continue
            cases.append(CaseRecord(case_id=stem, image_paths=[img], label_path=lbl))
        return cases

    def _load_brats(self) -> List[CaseRecord]:
        brats_root = self._find_brats_root() or self.root
        cases: List[CaseRecord] = []
        for subject_dir in sorted(brats_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            seg_files = list(subject_dir.glob("*_seg.nii.gz"))
            if not seg_files:
                continue
            label_path = seg_files[0]
            image_paths: List[Path] = []
            tags: List[str] = []
            for mod in self.BRATS_MODALITIES:
                mod_files = list(subject_dir.glob(f"*_{mod}.nii.gz"))
                if mod_files:
                    image_paths.append(mod_files[0])
                    tags.append(mod)
                else:
                    logger.warning("Missing modality '%s' in %s", mod, subject_dir.name)
            cases.append(
                CaseRecord(
                    case_id=subject_dir.name,
                    image_paths=image_paths,
                    label_path=label_path,
                    modality_tags=tags,
                )
            )
        return cases

    def _load_nnunet(self) -> List[CaseRecord]:
        images_dir = self.root / "imagesTr"
        labels_dir = self.root / "labelsTr"
        if not images_dir.exists():
            images_dir = self.root
        if not labels_dir.exists():
            labels_dir = self.root

        # nnU-Net naming: <case>_XXXX.nii.gz  (modality code at end)
        label_files = sorted(labels_dir.glob("*.nii.gz"))
        # Map case_id -> label
        label_map: Dict[str, Path] = {
            lf.name.replace(".nii.gz", ""): lf for lf in label_files
        }

        # Group image files by base id (strip _XXXX suffix)
        from collections import defaultdict
        img_groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
        for img in sorted(images_dir.glob("*.nii.gz")):
            stem = img.name.replace(".nii.gz", "")
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base, mod_code = parts
            else:
                base, mod_code = stem, "0000"
            img_groups[base].append((mod_code, img))

        cases: List[CaseRecord] = []
        for base_id, mod_list in sorted(img_groups.items()):
            lbl = label_map.get(base_id)
            if lbl is None:
                logger.warning("No label for case %s — skipping", base_id)
                continue
            mod_list.sort(key=lambda x: x[0])
            image_paths = [p for _, p in mod_list]
            tags = [c for c, _ in mod_list]
            cases.append(
                CaseRecord(
                    case_id=base_id,
                    image_paths=image_paths,
                    label_path=lbl,
                    modality_tags=tags,
                )
            )
        return cases


# ---------------------------------------------------------------------------
# Module 1: Structural Integrity Checker
# ---------------------------------------------------------------------------

class StructuralIntegrityChecker:
    """
    Validates geometric consistency between image(s) and label.

    Parameters
    ----------
    spacing_tol : float
        Absolute tolerance for spacing comparison (mm).
    origin_tol : float
        Absolute tolerance for origin comparison (mm).
    direction_tol : float
        Absolute tolerance for direction cosine comparison.
    required_modalities : list of str, optional
        List of required modality tags. Warns if missing.
    """

    def __init__(
        self,
        spacing_tol: float = 1e-3,
        origin_tol: float = 1e-3,
        direction_tol: float = 1e-4,
        required_modalities: Optional[List[str]] = None,
    ) -> None:
        self.spacing_tol = spacing_tol
        self.origin_tol = origin_tol
        self.direction_tol = direction_tol
        self.required_modalities = required_modalities or []

    def check(self, case: CaseRecord) -> StructuralResult:
        result = StructuralResult(case_id=case.case_id)
        try:
            label_img = self._load_sitk(case.label_path)
            if label_img is None:
                result.passed = False
                result.header_valid = False
                result.errors.append("Cannot load label file")
                return result

            ref_size = label_img.GetSize()
            ref_spacing = label_img.GetSpacing()
            ref_origin = label_img.GetOrigin()
            ref_direction = label_img.GetDirection()

            for img_path in case.image_paths:
                img = self._load_sitk(img_path)
                if img is None:
                    result.header_valid = False
                    result.warnings.append(f"Cannot load image {img_path.name}")
                    continue

                # Size
                if img.GetSize() != ref_size:
                    result.size_match = False
                    result.errors.append(
                        f"Size mismatch: image {img_path.name} {img.GetSize()} vs label {ref_size}"
                    )

                # Spacing
                if not np.allclose(img.GetSpacing(), ref_spacing, atol=self.spacing_tol):
                    result.spacing_match = False
                    result.warnings.append(
                        f"Spacing mismatch: {img_path.name} {img.GetSpacing()} vs label {ref_spacing}"
                    )

                # Origin
                if not np.allclose(img.GetOrigin(), ref_origin, atol=self.origin_tol):
                    result.origin_match = False
                    result.warnings.append(
                        f"Origin mismatch: {img_path.name} {img.GetOrigin()} vs label {ref_origin}"
                    )

                # Direction
                if not np.allclose(img.GetDirection(), ref_direction, atol=self.direction_tol):
                    result.direction_match = False
                    result.warnings.append(
                        f"Direction mismatch: {img_path.name}"
                    )

            # Missing modalities check
            if self.required_modalities and case.modality_tags:
                for mod in self.required_modalities:
                    if mod not in case.modality_tags:
                        result.missing_modalities.append(mod)
                        result.warnings.append(f"Missing required modality: {mod}")

            if result.errors:
                result.passed = False

        except Exception as exc:  # noqa: BLE001
            result.passed = False
            result.header_valid = False
            result.errors.append(f"Unexpected error: {exc}")

        return result

    @staticmethod
    def _load_sitk(path: Path) -> Optional[sitk.Image]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sitk.ReadImage(str(path))
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Module 2: Semantic Integrity Checker
# ---------------------------------------------------------------------------

class SemanticIntegrityChecker:
    """
    Validates label semantics: empty masks, required/unexpected classes, dtype.

    Parameters
    ----------
    required_classes : list of int
        Label values that must be present in every case.
    allowed_classes : list of int, optional
        If provided, any label not in this list triggers a warning.
    """

    def __init__(
        self,
        required_classes: Optional[List[int]] = None,
        allowed_classes: Optional[List[int]] = None,
    ) -> None:
        self.required_classes: List[int] = required_classes or []
        self.allowed_classes: Optional[Set[int]] = (
            set(allowed_classes) if allowed_classes is not None else None
        )

    def check(self, case: CaseRecord) -> SemanticResult:
        result = SemanticResult(case_id=case.case_id)
        try:
            nib_img = nib.load(str(case.label_path))
            data = nib_img.get_fdata(dtype=np.float32)

            # Float contamination: check if label values are non-integer
            if not np.all(data == np.floor(data)):
                result.has_float_contamination = True
                result.warnings.append("Label contains non-integer float values")

            data_int = data.astype(np.int32)
            unique = sorted(np.unique(data_int).tolist())
            result.unique_labels = unique

            # Empty mask
            if unique == [0] or len(unique) == 0:
                result.is_empty_mask = True
                result.warnings.append("Mask is empty (background only)")

            # Missing required classes
            for cls in self.required_classes:
                if cls not in unique:
                    result.missing_required_classes.append(cls)
            if result.missing_required_classes:
                result.warnings.append(
                    f"Missing required classes: {result.missing_required_classes}"
                )

            # Unexpected labels
            if self.allowed_classes is not None:
                unexpected = [u for u in unique if u not in self.allowed_classes]
                result.unexpected_labels = unexpected
                if unexpected:
                    result.warnings.append(f"Unexpected label values: {unexpected}")

            if result.missing_required_classes or result.has_float_contamination:
                result.passed = False

        except Exception as exc:  # noqa: BLE001
            result.passed = False
            result.errors.append(f"Semantic check error: {exc}")

        return result


# ---------------------------------------------------------------------------
# Module 3: Distribution Analyzer
# ---------------------------------------------------------------------------

class DistributionAnalyzer:
    """
    Computes voxel-level class distribution statistics per case.

    BraTS sub-region definitions:
      WT = labels {1, 2, 4}
      TC = labels {1, 4}
      ET = label  {4}
    """

    BRATS_LABELS: Set[int] = {1, 2, 4}
    WT_LABELS: Set[int] = {1, 2, 4}
    TC_LABELS: Set[int] = {1, 4}
    ET_LABELS: Set[int] = {4}

    def __init__(self, is_brats: bool = False) -> None:
        self.is_brats = is_brats

    def analyze(self, case: CaseRecord) -> DistributionResult:
        result = DistributionResult(case_id=case.case_id)
        try:
            nib_img = nib.load(str(case.label_path))
            data = nib_img.get_fdata(dtype=np.float32).astype(np.int32)
            total_voxels = data.size

            unique, counts = np.unique(data, return_counts=True)
            for lbl, cnt in zip(unique.tolist(), counts.tolist()):
                result.voxel_count_per_class[lbl] = cnt
                result.voxel_ratio_per_class[lbl] = cnt / total_voxels if total_voxels else 0.0

            if self.is_brats:
                result.wt_voxels = int(np.isin(data, list(self.WT_LABELS)).sum())
                result.tc_voxels = int(np.isin(data, list(self.TC_LABELS)).sum())
                result.et_voxels = int(np.isin(data, list(self.ET_LABELS)).sum())

        except Exception as exc:  # noqa: BLE001
            logger.warning("Distribution analysis failed for %s: %s", case.case_id, exc)

        return result


# ---------------------------------------------------------------------------
# Module 4: Split Safety Analyzer
# ---------------------------------------------------------------------------

class SplitSafetyAnalyzer:
    """
    Recommends split strategies and warns about rare classes.

    Parameters
    ----------
    rare_class_threshold : float
        Fraction of cases below which a class is considered rare (default 0.05).
    et_rare_threshold : float
        BraTS ET-specific rare threshold (default 0.10).
    """

    def __init__(
        self,
        rare_class_threshold: float = 0.05,
        et_rare_threshold: float = 0.10,
    ) -> None:
        self.rare_threshold = rare_class_threshold
        self.et_rare_threshold = et_rare_threshold

    def analyze(
        self,
        summary: DatasetSummary,
        is_brats: bool = False,
    ) -> List[str]:
        warnings_out: List[str] = []
        n = summary.total_cases
        if n == 0:
            return warnings_out

        for cls, count in summary.class_presence_stats.items():
            if cls == 0:
                continue
            ratio = count / n
            if ratio < self.rare_threshold:
                warnings_out.append(
                    f"Class {cls} present in only {count}/{n} cases "
                    f"({ratio:.1%}) — consider stratified splitting."
                )

        if is_brats:
            et_count = summary.class_presence_stats.get(4, 0)
            et_ratio = et_count / n
            if et_ratio < self.et_rare_threshold:
                warnings_out.append(
                    f"BraTS ET (label=4) present in only {et_count}/{n} cases "
                    f"({et_ratio:.1%}) — naive splitting will likely produce "
                    "ET-free validation sets. Use stratified splitting by ET presence."
                )

        warnings_out.append(
            "Recommended strategy: stratified k-fold split based on class presence flags."
        )
        return warnings_out


# ---------------------------------------------------------------------------
# Main QA Pipeline
# ---------------------------------------------------------------------------

class MedicalSegQAPipeline:
    """
    Orchestrates all QA modules for a 3D medical segmentation dataset.

    Parameters
    ----------
    root : str or Path
        Dataset root directory.
    dataset_format : str, optional
        Force format: 'generic', 'brats', 'nnunet'. Auto-detected if None.
    required_classes : list of int, optional
        Classes that must be present in every segmentation mask.
    allowed_classes : list of int, optional
        Whitelist of valid label values.
    required_modalities : list of str, optional
        Required modality tags (e.g. ['flair','t1','t1ce','t2'] for BraTS).
    spacing_tol : float
        Geometric spacing tolerance (mm).
    origin_tol : float
        Geometric origin tolerance (mm).
    direction_tol : float
        Geometric direction tolerance.
    rare_class_threshold : float
        Fraction below which a class is flagged as rare.
    imbalance_threshold : float
        Gini-based imbalance score above which a warning is issued.
    n_workers : int
        Number of parallel workers (1 = sequential).
    verbose : bool
        Include per-case results in summary.
    """

    def __init__(
        self,
        root: str | Path,
        dataset_format: Optional[str] = None,
        required_classes: Optional[List[int]] = None,
        allowed_classes: Optional[List[int]] = None,
        required_modalities: Optional[List[str]] = None,
        spacing_tol: float = 1e-3,
        origin_tol: float = 1e-3,
        direction_tol: float = 1e-4,
        rare_class_threshold: float = 0.05,
        imbalance_threshold: float = 0.7,
        n_workers: int = 1,
        verbose: bool = False,
    ) -> None:
        self.root = Path(root)
        self.n_workers = n_workers
        self.verbose = verbose
        self.imbalance_threshold = imbalance_threshold

        # Detect format
        detector = DatasetDetector(self.root)
        self.fmt = dataset_format or detector.detect_format()
        self.is_brats = self.fmt == "brats"

        logger.info("Dataset format detected: %s", self.fmt)

        # Load cases
        self.cases = detector.load_cases(self.fmt)
        logger.info("Loaded %d cases", len(self.cases))

        # Sub-modules
        _req_mods = required_modalities
        if _req_mods is None and self.is_brats:
            _req_mods = DatasetDetector.BRATS_MODALITIES

        self.structural_checker = StructuralIntegrityChecker(
            spacing_tol=spacing_tol,
            origin_tol=origin_tol,
            direction_tol=direction_tol,
            required_modalities=_req_mods,
        )
        self.semantic_checker = SemanticIntegrityChecker(
            required_classes=required_classes,
            allowed_classes=allowed_classes,
        )
        self.dist_analyzer = DistributionAnalyzer(is_brats=self.is_brats)
        self.split_analyzer = SplitSafetyAnalyzer(
            rare_class_threshold=rare_class_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> DatasetSummary:
        """Execute the full QA pipeline and return a DatasetSummary."""
        summary = DatasetSummary(
            total_cases=len(self.cases),
            dataset_format=self.fmt,
        )

        struct_results: List[StructuralResult] = []
        sem_results: List[SemanticResult] = []
        dist_results: List[DistributionResult] = []

        if self.n_workers > 1:
            struct_results, sem_results, dist_results = self._run_parallel()
        else:
            struct_results, sem_results, dist_results = self._run_sequential()

        # Aggregate
        self._aggregate(summary, struct_results, sem_results, dist_results)

        # Split safety
        summary.split_safety_warnings = self.split_analyzer.analyze(
            summary, is_brats=self.is_brats
        )

        if self.verbose:
            summary.case_structural_results = [asdict(r) for r in struct_results]
            summary.case_semantic_results = [asdict(r) for r in sem_results]
            summary.case_distribution_results = [asdict(r) for r in dist_results]

        return summary

    def export_json(self, summary: DatasetSummary, output_path: str | Path) -> None:
        """Serialize summary to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        payload = _convert(asdict(summary))
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("QA report written to %s", out)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_case(
        self, case: CaseRecord
    ) -> Tuple[StructuralResult, SemanticResult, DistributionResult]:
        s = self.structural_checker.check(case)
        se = self.semantic_checker.check(case)
        d = self.dist_analyzer.analyze(case)
        return s, se, d

    def _run_sequential(
        self,
    ) -> Tuple[
        List[StructuralResult], List[SemanticResult], List[DistributionResult]
    ]:
        structs, sems, dists = [], [], []
        for case in tqdm(self.cases, desc="QA Cases", unit="case"):
            try:
                s, se, d = self._process_case(case)
            except Exception as exc:  # noqa: BLE001
                logger.error("Fatal error processing case %s: %s", case.case_id, exc)
                s = StructuralResult(case_id=case.case_id, passed=False)
                s.errors.append(str(exc))
                se = SemanticResult(case_id=case.case_id, passed=False)
                d = DistributionResult(case_id=case.case_id)
            structs.append(s)
            sems.append(se)
            dists.append(d)
        return structs, sems, dists

    def _run_parallel(
        self,
    ) -> Tuple[
        List[StructuralResult], List[SemanticResult], List[DistributionResult]
    ]:
        structs, sems, dists = [], [], []
        futures_map = {}
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            for case in self.cases:
                future = pool.submit(self._process_case, case)
                futures_map[future] = case.case_id

            for future in tqdm(
                as_completed(futures_map),
                total=len(futures_map),
                desc="QA Cases (parallel)",
                unit="case",
            ):
                cid = futures_map[future]
                try:
                    s, se, d = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Fatal error for case %s: %s", cid, exc)
                    s = StructuralResult(case_id=cid, passed=False)
                    s.errors.append(str(exc))
                    se = SemanticResult(case_id=cid, passed=False)
                    d = DistributionResult(case_id=cid)
                structs.append(s)
                sems.append(se)
                dists.append(d)
        return structs, sems, dists

    def _aggregate(
        self,
        summary: DatasetSummary,
        structs: List[StructuralResult],
        sems: List[SemanticResult],
        dists: List[DistributionResult],
    ) -> None:
        class_presence: Dict[int, int] = {}
        all_ratios: Dict[int, List[float]] = {}

        for s in structs:
            if not s.passed:
                summary.failed_structural += 1
            if not s.header_valid:
                summary.skipped_corrupted += 1

        for se in sems:
            if se.is_empty_mask:
                summary.empty_masks += 1
            if se.missing_required_classes:
                summary.missing_class_cases += 1
            if len(se.unique_labels) > 1:  # more than background
                summary.multi_class_cases += 1
            for lbl in se.unique_labels:
                class_presence[lbl] = class_presence.get(lbl, 0) + 1

        for d in dists:
            for lbl, ratio in d.voxel_ratio_per_class.items():
                all_ratios.setdefault(lbl, []).append(ratio)

        summary.class_presence_stats = class_presence

        # Class imbalance score (Gini coefficient on mean voxel ratios across fg classes)
        fg_means = [
            float(np.mean(v))
            for k, v in all_ratios.items()
            if k != 0 and v
        ]
        if len(fg_means) >= 2:
            summary.class_imbalance_score = self._gini(fg_means)
            if summary.class_imbalance_score > self.imbalance_threshold:
                logger.warning(
                    "High class imbalance detected (Gini=%.3f > %.3f).",
                    summary.class_imbalance_score,
                    self.imbalance_threshold,
                )

    @staticmethod
    def _gini(values: List[float]) -> float:
        """Compute normalized Gini coefficient."""
        arr = np.sort(np.abs(np.array(values, dtype=np.float64)))
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * np.dot(idx, arr) / (n * arr.sum())) - (n + 1) / n)

    # ------------------------------------------------------------------
    # Convenience: pretty-print summary
    # ------------------------------------------------------------------

    def print_summary(self, summary: DatasetSummary) -> None:
        sep = "=" * 60
        print(sep)
        print("  MEDICAL SEGMENTATION QA REPORT")
        print(sep)
        print(f"  Dataset format      : {summary.dataset_format}")
        print(f"  Total cases         : {summary.total_cases}")
        print(f"  Failed structural   : {summary.failed_structural}")
        print(f"  Empty masks         : {summary.empty_masks}")
        print(f"  Missing-class cases : {summary.missing_class_cases}")
        print(f"  Skipped/corrupted   : {summary.skipped_corrupted}")
        print(f"  Multi-class cases   : {summary.multi_class_cases}")
        print(f"  Imbalance score     : {summary.class_imbalance_score:.4f}")
        print()
        print("  Class presence (label: n_cases):")
        for cls, cnt in sorted(summary.class_presence_stats.items()):
            pct = cnt / summary.total_cases * 100 if summary.total_cases else 0
            print(f"    Label {cls:3d}  →  {cnt} cases  ({pct:.1f}%)")
        print()
        if summary.split_safety_warnings:
            print("  Split Safety Warnings:")
            for w in summary.split_safety_warnings:
                print(f"    ⚠  {w}")
        print(sep)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def _example_generic(root: str) -> None:
    """Example: Generic images/labels dataset."""
    pipeline = MedicalSegQAPipeline(
        root=root,
        required_classes=[0, 1],
        allowed_classes=[0, 1],
        spacing_tol=1e-2,
        n_workers=1,
        verbose=True,
    )
    summary = pipeline.run()
    pipeline.print_summary(summary)
    pipeline.export_json(summary, Path(root) / "qa_report_generic.json")


def _example_brats(root: str) -> None:
    """Example: BraTS-style dataset."""
    pipeline = MedicalSegQAPipeline(
        root=root,
        dataset_format="brats",
        required_classes=[1, 2, 4],
        allowed_classes=[0, 1, 2, 4],
        required_modalities=["flair", "t1", "t1ce", "t2"],
        rare_class_threshold=0.05,
        n_workers=4,
        verbose=True,
    )
    summary = pipeline.run()
    pipeline.print_summary(summary)
    pipeline.export_json(summary, Path(root) / "qa_report_brats.json")


def _example_nnunet(root: str) -> None:
    """Example: nnU-Net raw dataset."""
    pipeline = MedicalSegQAPipeline(
        root=root,
        dataset_format="nnunet",
        required_classes=[1],
        spacing_tol=1e-3,
        n_workers=2,
        verbose=False,
    )
    summary = pipeline.run()
    pipeline.print_summary(summary)
    pipeline.export_json(summary, Path(root) / "qa_report_nnunet.json")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Medical Segmentation Dataset QA Pipeline"
    )
    parser.add_argument("root", type=str, help="Dataset root directory")
    parser.add_argument(
        "--format",
        choices=["generic", "brats", "nnunet"],
        default=None,
        help="Force dataset format (auto-detected if omitted)",
    )
    parser.add_argument(
        "--required-classes",
        nargs="*",
        type=int,
        default=None,
        metavar="CLS",
        help="Required label classes (e.g. 1 2 4)",
    )
    parser.add_argument(
        "--allowed-classes",
        nargs="*",
        type=int,
        default=None,
        metavar="CLS",
        help="Whitelist of valid label values",
    )
    parser.add_argument(
        "--spacing-tol",
        type=float,
        default=1e-3,
        help="Spacing comparison tolerance (mm)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel worker count"
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=0.05,
        help="Rare-class fraction threshold",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=0.7,
        help="Gini imbalance score threshold",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-case output")
    parser.add_argument("--export-json", type=str, default=None, metavar="PATH")

    args = parser.parse_args()

    pipe = MedicalSegQAPipeline(
        root=args.root,
        dataset_format=args.format,
        required_classes=args.required_classes,
        allowed_classes=args.allowed_classes,
        spacing_tol=args.spacing_tol,
        rare_class_threshold=args.rare_threshold,
        imbalance_threshold=args.imbalance_threshold,
        n_workers=args.workers,
        verbose=args.verbose,
    )

    report = pipe.run()
    pipe.print_summary(report)

    if args.export_json:
        pipe.export_json(report, args.export_json)
    else:
        pipe.export_json(report, Path(args.root) / "qa_report.json")
