
# ============================================================
# Medical Imaging Dataset Utilities (Research-grade)
# ============================================================

import os
import shutil
import random
from glob import glob
import SimpleITK as sitk


# ------------------------------------------------------------
# 1️⃣ MedicalVolumeInspector
# ------------------------------------------------------------
class MedicalVolumeInspector:
    """
    Scientific integrity checker for medical volumes.

    Verifies consistency between image and label in:
    - voxel spacing
    - volume size
    - origin
    - direction cosines

    Works for CT and MRI.
    """

    @staticmethod
    def inspect(image_path, label_path=None, strict=True):
        img = sitk.ReadImage(image_path)

        getters = {
            "size": img.GetSize,
            "spacing": img.GetSpacing,
            "origin": img.GetOrigin,
            "direction": img.GetDirection
        }

        info = {k: v() for k, v in getters.items()}

        if label_path is not None:
            lbl = sitk.ReadImage(label_path)

            for key, getter in getters.items():
                img_val = getter()
                lbl_val = getattr(lbl, getter.__name__)()

                if img_val != lbl_val:
                    msg = (
                        f"Mismatch in {key}:\n"
                        f"  image = {img_val}\n"
                        f"  label = {lbl_val}"
                    )
                    if strict:
                        raise ValueError(msg)
                    else:
                        print("⚠️", msg)

        return info


# ------------------------------------------------------------
# 2️⃣ nnU-Net → General Dataset Converter
# ------------------------------------------------------------
class NnunetToGeneralDataset:
    """
    Converts nnU-Net style dataset into a generic structure:

    dataset/
      train/
        images/
        labels/
      val/
        images/
        labels/
      test/
        images/
        labels/
    """

    def __init__(
        self,
        imagesTr,
        labelsTr,
        imagesTs,
        output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    ):
        self.imagesTr = imagesTr
        self.labelsTr = labelsTr
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        random.seed(seed)

    def run(self):
        images = sorted(glob(os.path.join(self.imagesTr, "*.nii*")))
        labels = sorted(glob(os.path.join(self.labelsTr, "*.nii*")))

        assert len(images) == len(labels), "Image/label count mismatch"

        pairs = list(zip(images, labels))
        random.shuffle(pairs)

        n = len(pairs)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_pairs = pairs[:n_train]
        val_pairs   = pairs[n_train:n_train + n_val]
        test_pairs  = pairs[n_train + n_val:]  # remaining cases

        splits = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }

        for split, data in splits.items():
            img_dir = os.path.join(self.output_dir, split, "images")
            lbl_dir = os.path.join(self.output_dir, split, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            for img, lbl in data:
                shutil.copy(img, img_dir)
                shutil.copy(lbl, lbl_dir)

        # Summary (useful for verification)
        print("\n📊 Dataset split summary:")
        for split, data in splits.items():
            print(f"  {split}: {len(data)} cases")


# ------------------------------------------------------------
# 3️⃣ MedicalDatasetPipeline
# ------------------------------------------------------------
class MedicalDatasetPipeline:
    """
    End-to-end medical dataset pipeline:
    1. Verify medical consistency on a sample case
    2. Convert dataset structure (train / val / test)
    """

    def __init__(self, inspector, converter):
        self.inspector = inspector
        self.converter = converter

    def run(self, sample_image, sample_label):
        print("🔬 Running medical integrity inspection...")
        info = self.inspector.inspect(sample_image, sample_label)

        print("✅ Inspection passed:")
        for k, v in info.items():
            print(f"  {k}: {v}")

        print("\n📂 Converting dataset structure...")
        self.converter.run()
        print("✅ Dataset conversion completed")


# # MedicalVolumeInspector  ──▶  NnunetToGeneralDataset  ──▶  MedicalDatasetPipeline
# (تحقق طبي)                    (تنظيم الداتا)                 (تشغيل كامل)




# # how to use it 


# from dataset import (
#     MedicalVolumeInspector,
#     NnunetToGeneralDataset,
#     MedicalDatasetPipeline
# )

# inspector = MedicalVolumeInspector()

# converter = NnunetToGeneralDataset(
#     imagesTr="nnUNet_raw/Dataset100_SPINE/imagesTr",
#     labelsTr="nnUNet_raw/Dataset100_SPINE/labelsTr",
#     imagesTs="nnUNet_raw/Dataset100_SPINE/imagesTs",
#     output_dir="dataset_general"
# )

# pipeline = MedicalDatasetPipeline(inspector, converter)

# pipeline.run(
#     sample_image="nnUNet_raw/Dataset100_SPINE/imagesTr/SPINE_000_0000.nii.gz",
#     sample_label="nnUNet_raw/Dataset100_SPINE/labelsTr/SPINE_000.nii.gz"
# )
