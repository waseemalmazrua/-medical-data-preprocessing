# ============================================================
# Medical Imaging Dataset Quality Control (Research-grade)
# Author: Waseem
# ============================================================

import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm


# ------------------------------------------------------------
# 1️⃣ Structural Integrity Checker (with float tolerance)
# ------------------------------------------------------------
class StructuralIntegrityChecker:
    """
    Checks geometric consistency between image and label.
    - size        (exact)
    - direction   (exact)
    - spacing     (float tolerance)
    - origin      (float tolerance)

    Critical for CT / MRI voxel-wise learning.
    """

    @staticmethod
    def _almost_equal_tuple(a, b, tol=1e-4):
        return all(abs(x - y) < tol for x, y in zip(a, b))

    @staticmethod
    def check(image: sitk.Image, label: sitk.Image):
        checks = {
            "size": (image.GetSize(), label.GetSize()),
            "spacing": (image.GetSpacing(), label.GetSpacing()),
            "origin": (image.GetOrigin(), label.GetOrigin()),
            "direction": (image.GetDirection(), label.GetDirection()),
        }

        errors = {}

        for key, (img_val, lbl_val) in checks.items():

            # size & direction → strict check
            if key in ("size", "direction"):
                if img_val != lbl_val:
                    errors[key] = {
                        "image": img_val,
                        "label": lbl_val
                    }

            # spacing & origin → float tolerance
            else:
                if not StructuralIntegrityChecker._almost_equal_tuple(
                    img_val, lbl_val, tol=1e-4
                ):
                    errors[key] = {
                        "image": img_val,
                        "label": lbl_val
                    }

        return errors  # empty dict => OK


# ------------------------------------------------------------
# 2️⃣ Semantic Integrity Checker
# ------------------------------------------------------------
class SemanticIntegrityChecker:
    """
    Checks semantic validity of segmentation labels.

    Example:
    0 = background
    1 = liver        (required)
    2 = tumor        (optional)
    """

    def __init__(self, required_classes=(1,), optional_classes=(2,)):
        self.required_classes = required_classes
        self.optional_classes = optional_classes

    def check(self, label_array: np.ndarray):
        unique_labels = np.unique(label_array).astype(int)
        issues = []

        # Empty mask (all background)
        if len(unique_labels) == 1 and unique_labels[0] == 0:
            issues.append("Empty segmentation mask (background only)")

        # Required anatomy missing
        for cls in self.required_classes:
            if cls not in unique_labels:
                issues.append(f"Missing required class: {cls}")

        return {
            "unique_labels": unique_labels.tolist(),
            "issues": issues
        }


# ------------------------------------------------------------
# 3️⃣ Distribution Checker
# ------------------------------------------------------------
class DistributionChecker:
    """
    Computes voxel distribution per class.
    Useful for class imbalance analysis.
    """

    @staticmethod
    def check(label_array: np.ndarray):
        total_voxels = label_array.size
        stats = {}

        for cls in np.unique(label_array):
            count = int(np.sum(label_array == cls))
            stats[int(cls)] = {
                "voxels": count,
                "ratio": round(count / total_voxels, 6)
            }

        return stats


# ------------------------------------------------------------
# 4️⃣ Case-Level QA Checker
# ------------------------------------------------------------
class CaseQAChecker:
    """
    Full QA for a single patient case (image + label).
    """

    def __init__(self):
        self.structural_checker = StructuralIntegrityChecker()
        self.semantic_checker = SemanticIntegrityChecker()
        self.distribution_checker = DistributionChecker()

    def run(self, image_path: str, label_path: str):
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        label_array = sitk.GetArrayFromImage(label)

        report = {}

        # Structural integrity
        structural_errors = self.structural_checker.check(image, label)
        report["structural_ok"] = len(structural_errors) == 0
        report["structural_errors"] = structural_errors

        # Semantic integrity
        report["semantic"] = self.semantic_checker.check(label_array)

        # Distribution
        report["distribution"] = self.distribution_checker.check(label_array)

        return report


# ------------------------------------------------------------
# 5️⃣ Dataset-Level QA Checker
# ------------------------------------------------------------
class DatasetQAChecker:
    """
    Runs QA over the entire dataset (patient-level).

    Ensures:
    - No structural corruption
    - No empty labels
    - No missing required anatomy
    - Safe for train/val/test usage
    """

    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.case_checker = CaseQAChecker()

    def run(self, verbose=False):
        image_paths = sorted(glob(f"{self.images_dir}/*.nii*"))
        label_paths = sorted(glob(f"{self.labels_dir}/*.nii*"))

        assert len(image_paths) == len(label_paths), (
            "❌ Image / Label count mismatch"
        )

        summary = {
            "total_cases": len(image_paths),
            "failed_structural": [],
            "semantic_issues": [],
            "empty_masks": [],
        }

        for img_path, lbl_path in tqdm(
            zip(image_paths, label_paths),
            total=len(image_paths),
            desc="Running Dataset QA"
        ):
            report = self.case_checker.run(img_path, lbl_path)

            # Structural errors
            if not report["structural_ok"]:
                summary["failed_structural"].append({
                    "case": img_path,
                    "errors": report["structural_errors"]
                })

            # Semantic issues
            if report["semantic"]["issues"]:
                summary["semantic_issues"].append({
                    "case": img_path,
                    "issues": report["semantic"]["issues"]
                })

            # Empty masks
            if report["semantic"]["unique_labels"] == [0]:
                summary["empty_masks"].append(img_path)

            if verbose:
                print(img_path)
                print(report)

        return summary


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
# from pprint import pprint
#
# checker = DatasetQAChecker(
#     images_dir="/content/drive/MyDrive/dataset_general_for_liver/train/images",
#     labels_dir="/content/drive/MyDrive/dataset_general_for_liver/train/labels"
# )
#
# summary = checker.run()
# pprint(summary)
