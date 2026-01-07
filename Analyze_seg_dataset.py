import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from collections import defaultdict


def analyze_segmentation_dataset(labels_dir):
    """
    Generic analyzer for ANY multi-class 3D segmentation dataset.
    Works with MONAI, nnU-Net, or any NIfTI-based dataset.

    Parameters
    ----------
    labels_dir : str
        Path to directory containing label .nii or .nii.gz files.

    Returns
    -------
    summary : dict
        Dataset statistics.
    """

    label_paths = sorted(
        glob(os.path.join(labels_dir, "*.nii")) +
        glob(os.path.join(labels_dir, "*.nii.gz"))
    )

    if len(label_paths) == 0:
        raise RuntimeError("❌ No label files found")

    total_cases = len(label_paths)

    # Dynamic containers (no class assumptions)
    voxel_counts = defaultdict(int)
    cases_with_class = defaultdict(int)

    background_only_cases = 0
    multi_class_cases = 0

    # -------------------------------------------------
    # Iterate through cases
    # -------------------------------------------------
    for p in tqdm(label_paths, desc="Analyzing dataset"):
        seg = nib.load(p).get_fdata()
        seg = seg.astype(np.int64)

        unique_labels = np.unique(seg)

        # Count voxels
        for lbl in unique_labels:
            voxel_counts[lbl] += np.sum(seg == lbl)

        # Count cases per class (ignore background = 0)
        fg_labels = [l for l in unique_labels if l != 0]

        if len(fg_labels) == 0:
            background_only_cases += 1
        else:
            for lbl in fg_labels:
                cases_with_class[lbl] += 1

        if len(fg_labels) > 1:
            multi_class_cases += 1

    # -------------------------------------------------
    # Build summary
    # -------------------------------------------------
    total_voxels = sum(voxel_counts.values())

    summary = {
        "total_cases": total_cases,
        "background_only_cases": background_only_cases,
        "multi_class_cases": multi_class_cases,
        "classes_found": sorted(voxel_counts.keys()),
        "cases_per_class": dict(cases_with_class),
        "voxel_counts": dict(voxel_counts),
        "voxel_percentages": {
            int(k): 100 * v / total_voxels
            for k, v in voxel_counts.items()
        }
    }

    return summary


def print_summary(summary):
    print("\n📊 DATASET SUMMARY")
    print("=" * 50)

    print(f"Total cases           : {summary['total_cases']}")
    print(f"Background-only cases : {summary['background_only_cases']}")
    print(f"Multi-class cases     : {summary['multi_class_cases']}")

    print("\nClasses found:")
    print(summary["classes_found"])

    print("\nCases per class (foreground only):")
    for k, v in sorted(summary["cases_per_class"].items()):
        print(f"  Class {k:<3}: {v}")

    print("\nVoxel counts:")
    for k, v in sorted(summary["voxel_counts"].items()):
        print(f"  Class {k:<3}: {v:,}")

    print("\nVoxel percentages:")
    for k, v in sorted(summary["voxel_percentages"].items()):
        print(f"  Class {k:<3}: {v:.4f}%")


# # how use 
# # summary = analyze_segmentation_dataset("/dataset/train/labels")
# print_summary(summary)
