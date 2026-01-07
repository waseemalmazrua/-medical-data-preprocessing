# Copyright (c) 2023 PYCAD
# This file is part of the PYCAD library and is released under the MIT License:
# https://github.com/amine0110/pycad/blob/main/LICENSE


import os
import shutil
import nibabel as nib
import numpy as np
from glob import glob
import random
import logging
import SimpleITK as sitk


class MultiClassNiftiMerger:
    '''
    If you have multiple nifti files representing different classes for the same patient, then this 
    function is for you, it helps you merge the nifti files into one nifti file.

    ### Params
    - volume_path: Path to the volume NIfTI file.
    - class_paths: List of paths to the class NIfTI files.
    - output_dir: Directory where the merged files will be saved.
    - move_volumes: Flag to control whether to move corresponding volumes.

    ### Example of usage

    ```Python
    # Example usage for directories
    from pycad.datasets import MultiClassNiftiMerger

    volume_dir = 'datasets/hips/hip_right100/volumes'
    class_dirs = ['datasets/hips/hip_right100/segmentations', 'datasets/hips/hip_left100/segmentations']
    output_dir = 'datasets/hips/merged'
    MultiClassNiftiMerger.process_directories(volume_dir, class_dirs, output_dir, move_volumes=True)
    ```
    '''
    
    def __init__(self, volume_path, class_paths, output_dir, move_volumes=False):
        self.volume_path = volume_path
        self.class_paths = class_paths
        self.output_dir = output_dir
        self.move_volumes = move_volumes

        self.segmentations_dir = os.path.join(output_dir, 'segmentations')
        self.volumes_dir = os.path.join(output_dir, 'volumes')

    def check_files(self):
        # Check if files exist
        paths_to_check = [self.volume_path] + self.class_paths
        for path in paths_to_check:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

    def combine_classes(self):
        self.check_files()

        # Create directories for output
        os.makedirs(self.segmentations_dir, exist_ok=True)
        if self.move_volumes:
            os.makedirs(self.volumes_dir, exist_ok=True)

        # Initialize a combined array with zeros
        first_nifti = nib.load(self.class_paths[0])
        combined_classes = np.zeros(first_nifti.shape, dtype=np.int16)

        # Assign new class labels
        for idx, class_path in enumerate(self.class_paths):
            class_nifti = nib.load(class_path)
            class_data = class_nifti.get_fdata()
            combined_classes[class_data > 0] = idx + 1

        # Create a new NIfTI image for the combined classes
        combined_nifti = nib.Nifti1Image(combined_classes, affine=class_nifti.affine)

        # Save the new NIfTI file
        combined_filename = os.path.basename(self.volume_path).replace('volume', 'combined')
        combined_path = os.path.join(self.segmentations_dir, combined_filename)
        nib.save(combined_nifti, combined_path)

        # Optionally move the volume file
        if self.move_volumes:
            shutil.copy(self.volume_path, self.volumes_dir)

        print(f"Combined NIfTI file saved at: {combined_path}")

    @staticmethod
    def process_directories(volume_dir, class_dirs, output_dir, ext='.nii.gz', move_volumes=False):
        volume_files = glob(os.path.join(volume_dir, f'*{ext}'))

        for volume_file in volume_files:
            volume_filename = os.path.basename(volume_file)
            class_paths = [glob(os.path.join(class_dir, f"{volume_filename.split('.')[0]}*{ext}")) for class_dir in class_dirs]
            class_paths = [item for sublist in class_paths for item in sublist] # Flatten list

            if class_paths:
                merger = MultiClassNiftiMerger(
                    volume_file,
                    class_paths,
                    output_dir,
                    move_volumes
                )
                merger.combine_classes()


class DataSplitter:
    '''
    This class is for splitting the images and labels into train/valid/test folders. The format by default is the yolo format, it is as follows:\n
    train\n
    |__ images\n
        |__ image_0\n
        |__ image_1\n
        |__ ...\n
    |__ labels\n
        |__ labels_0\n
        |__ labels_1\n
        |__ ...\n
    \n
    valid\n
    |__ images\n
        |__ image_0\n
        |__ image_1\n
        |__ ...\n
    |__ labels\n
        |__ label_0\n
        |__ label_1\n
        |__ ...\n
    \n
    test\n
    |__ images\n
        |__ image_0\n
        |__ image_1\n
        |__ ...\n
    |__ labels\n
        |__ label_0\n
        |__ label_1\n
        |__ ...\n
    
    ### Params
    - images_dir: the path to the images
    - labels_dir: the path to the labels 
    - output_dir: the path to save the split folders 
    - train_ratio: the train ratio, default=0.7
    - valid_ratio: the validation ratio, default=0.2
    - test_ratio: the test ratio, default=0.1
    - delete_input: whether you want to delete the input files after split, default=False

    ### Example of usage:
    ```
    from pycad.datasets import DataSplitter

    img = 'datasets/dental/xray_panoramic_mandible/images'
    msk = 'datasets/dental/xray_panoramic_mandible/masks'
    output = 'datasets/dental/test'

    splitter = DataSplitter(img, msk, output, 0.7, 0.2, 0.1, delete_input=False)
    splitter.run()
    '''
    def __init__(self, images_dir, labels_dir, output_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, delete_input=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.delete_input = delete_input
        self.setup_directories()

    def setup_directories(self):
        self.dirs = {
            'train': {'images': os.path.join(self.output_dir, 'train', 'images'),
                      'labels': os.path.join(self.output_dir, 'train', 'labels')},
            'valid': {'images': os.path.join(self.output_dir, 'valid', 'images'),
                      'labels': os.path.join(self.output_dir, 'valid', 'labels')},
            'test': {'images': os.path.join(self.output_dir, 'test', 'images'),
                     'labels': os.path.join(self.output_dir, 'test', 'labels')}
        }
        for d in self.dirs.values():
            for path in d.values():
                os.makedirs(path, exist_ok=True)

    def get_filenames(self):
        images = sorted(os.listdir(self.images_dir))
        labels = sorted(os.listdir(self.labels_dir))
        return images, labels

    def split_data(self, images, labels):
        data = list(zip(images, labels))
        random.shuffle(data)
        total = len(data)
        train_end = int(total * self.train_ratio)
        valid_end = train_end + int(total * self.valid_ratio)

        train_data = data[:train_end]
        valid_data = data[train_end:valid_end]
        test_data = data[valid_end:] if self.test_ratio > 0 else []

        return {'train': train_data, 'valid': valid_data, 'test': test_data}

    def copy_files(self, split_data):
        for split, data in split_data.items():
            for img, lbl in data:
                shutil.copy(os.path.join(self.images_dir, img), self.dirs[split]['images'])
                shutil.copy(os.path.join(self.labels_dir, lbl), self.dirs[split]['labels'])
                logging.info(f'Copied {img} and {lbl} to {split} set')

    def run(self):
        images, labels = self.get_filenames()
        split_data = self.split_data(images, labels)
        self.copy_files(split_data)

        if self.delete_input:
            shutil.rmtree(self.images_dir)
            shutil.rmtree(self.labels_dir)
            logging.info('Deleted original input directories')


class MetadataCopier:
    '''
    # Example usage:
    copier = MetadataCopier('datasets/volumes', 'datasets/segmentations', 'datasets/new/volumes', 'datasets/new/segmentations')
    copier.load_and_copy_metadata()
    '''
    def __init__(self, volume_dir, segmentation_dir, output_volumes_dir, output_segmentations_dir):
        self.volume_dir = volume_dir
        self.segmentation_dir = segmentation_dir
        self.output_volumes_dir = output_volumes_dir
        self.output_segmentations_dir = output_segmentations_dir

    def load_and_copy_metadata(self):
        # Ensure the output directories exist
        os.makedirs(self.output_volumes_dir, exist_ok=True)
        os.makedirs(self.output_segmentations_dir, exist_ok=True)
        
        # Get all NIfTI files in the volumes and segmentation directories
        volume_files = [f for f in os.listdir(self.volume_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        segmentation_files = [f for f in os.listdir(self.segmentation_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

        # Assuming filenames are the same for corresponding volume and segmentation
        for volume_file in volume_files:
            if volume_file in segmentation_files:
                volume_path = os.path.join(self.volume_dir, volume_file)
                segmentation_path = os.path.join(self.segmentation_dir, volume_file)

                try:
                    # Load the volume and segmentation
                    volume = sitk.ReadImage(volume_path)
                    segmentation = sitk.ReadImage(segmentation_path)

                    # Copy metadata from segmentation to volume
                    volume.SetOrigin(segmentation.GetOrigin())
                    volume.SetDirection(segmentation.GetDirection())
                    volume.SetSpacing(segmentation.GetSpacing())

                    # Save the modified volume in the output volumes directory
                    modified_volume_path = os.path.join(self.output_volumes_dir, volume_file)
                    sitk.WriteImage(volume, modified_volume_path)
                    print(f'Modified volume saved to: {modified_volume_path}')

                    # Save the segmentation in the output segmentations directory without changing the filename
                    modified_segmentation_path = os.path.join(self.output_segmentations_dir, volume_file)
                    sitk.WriteImage(segmentation, modified_segmentation_path)
                    print(f'Segmentation saved to: {modified_segmentation_path}')
                    
                except RuntimeError as e:
                    print(f"Skipping {volume_file} due to error: {e}")

            else:
                print(f"No matching segmentation found for volume: {volume_file}")


class DataRenamer:
    """
    If we use the pycad splitter to create the train/valid/test folders, then this class is adapted for that, and is waiting for the folders train and valid with the subforlders images and labels.
    """

    def __init__(self, path_to_input, path_to_output, dataset_id, structure):
        self.dataset_id = dataset_id
        self.structure = structure

        self.path_to_train_image = glob(os.path.join(path_to_input, "train/images/*.nii")) + \
                                  glob(os.path.join(path_to_input, "train/images/*.nii.gz"))

        self.path_to_train_labels = glob(os.path.join(path_to_input, "train/labels/*.nii")) + \
                                    glob(os.path.join(path_to_input, "train/labels/*.nii.gz"))

        self.path_to_test_image = glob(os.path.join(path_to_input, "valid/images/*.nii")) + \
                                  glob(os.path.join(path_to_input, "valid/images/*.nii.gz"))

        self.path_to_test_labels = glob(os.path.join(path_to_input, "valid/labels/*.nii")) + \
                                  glob(os.path.join(path_to_input, "valid/labels/*.nii.gz"))

        output_path  = f"{path_to_output}/Dataset{self.dataset_id}_{self.structure}"
        self.path_to_nnunet_imagesTr = os.path.join(output_path, "imagesTr")
        self.path_to_nnunet_labelsTr = os.path.join(output_path, "labelsTr")
        self.path_to_nnunet_imagesTs = os.path.join(output_path, "imagesTs")

        os.makedirs(self.path_to_nnunet_imagesTr, exist_ok=True)
        os.makedirs(self.path_to_nnunet_imagesTs, exist_ok=True)
        os.makedirs(self.path_to_nnunet_labelsTr, exist_ok=True)
    
    def rename_train_data(self):
        for i, (vol, seg) in enumerate(zip(self.path_to_train_image, self.path_to_train_labels)):

            # Rename the training segmentations
            print(f"Segmentation file: {seg}")
            new_seg_filename = f"{self.structure}_{str(i).zfill(3)}.nii.gz"
            new_seg_filepath = os.path.join(self.path_to_nnunet_labelsTr, new_seg_filename) 
            print(f"new segmenation file: {new_seg_filepath}")

            shutil.copy(seg, new_seg_filepath)

            # Rename the training volumes
            print(f"Volume file: {vol}")
            new_volume_filename = f"{self.structure}_{str(i).zfill(3)}_0000.nii.gz"
            new_volume_filepath = os.path.join(self.path_to_nnunet_imagesTr, new_volume_filename)
            print(f"new volume file: {new_volume_filepath}") 

            shutil.copy(vol, new_volume_filepath)
    
    def rename_test_data(self):
        for i, (vol, seg) in enumerate(zip(self.path_to_test_image, self.path_to_test_labels)):

            # Rename the testing volumes
            print(f"Volume file: {vol}")
            new_volume_filename = f"{self.structure}_{str(i).zfill(3)}_0000.nii.gz"
            new_volume_filepath = os.path.join(self.path_to_nnunet_imagesTs, new_volume_filename)
            print(f"new volume file: {new_volume_filepath}") 

            shutil.copy(vol, new_volume_filepath)

            # Rename the testing segmentations
            print(f"segmentation file: {seg}")
            new_seg_filename = f"{self.structure}_{str(i).zfill(3)}.nii.gz"
            new_seg_filepath = os.path.join(self.path_to_nnunet_imagesTs, new_seg_filename)
            print(f"new segmentation file: {new_seg_filepath}") 

            shutil.copy(seg, new_seg_filepath)
    
    def run(self, rename_trainset=True, rename_testset=True):
        if rename_trainset:
            self.rename_train_data()
        
        if rename_testset:
            self.rename_test_data()











# سأشرح **كل كلاس علميًا** من زاوية **الطب والتصوير الطبي (Medical Imaging)**:
# ما المشكلة التي يحلّها، لماذا نحتاجه عمليًا في أبحاث CT / MRI / Ultrasound، وكيف يُستخدم ضمن بايبلاين احترافي (nnU-Net / YOLO / Deep Learning).
# سأحافظ على الدقة العلمية، مع أمثلة كود مبسطة لكل جزء.

# ---

# ## 1️⃣ `MultiClassNiftiMerger`

# ### 🔬 المشكلة الطبية التي يحلّها

# في التصوير الطبي ثلاثي الأبعاد (CT / MRI):

# * غالبًا يكون **كل عضو أو آفة محفوظة في ملف Segmentation مستقل**

#   * ملف للكِلية اليمنى
#   * ملف للكِلية اليسرى
#   * ملف للورم
# * بينما **نماذج التعلم العميق** (nnU-Net، MONAI) تتوقع:

#   * **ملف Segmentation واحد**
#   * يحتوي أرقام Class مختلفة (0,1,2,3…)

# ### 🧠 الفكرة العلمية

# * الـ **Voxel** (بكسل ثلاثي الأبعاد) يحمل قيمة:

#   * `0` = Background
#   * `1` = Class 1 (مثلاً Liver)
#   * `2` = Class 2 (Tumor)
# * هذا الكلاس **يحوّل عدة Masks ثنائية** → **Mask متعددة الأصناف**

# ### ⚙️ كيف يعمل تقنيًا

# 1. يقرأ أول Mask ليعرف الأبعاد `(D, H, W)`
# 2. ينشئ مصفوفة أصفار
# 3. لكل Class:

#    * أي Voxel > 0 → يُعطى رقم Class جديد
# 4. يحفظ NIfTI جديد مع نفس الـ affine (الموقع المكاني)

# ### 🏥 مثال طبي

# CT Abdomen:

# * `liver.nii.gz`
# * `tumor.nii.gz`

# الناتج:

# * `combined.nii.gz`

#   * Liver = 1
#   * Tumor = 2

# ### 🧪 مثال كود مبسط

# ```python
# merger = MultiClassNiftiMerger(
#     volume_path="ct_001.nii.gz",
#     class_paths=["liver_mask.nii.gz", "tumor_mask.nii.gz"],
#     output_dir="merged",
#     move_volumes=True
# )
# merger.combine_classes()
# ```

# ### ✅ لماذا مهم جدًا؟

# * **شرط أساسي لـ nnU-Net**
# * يمنع تدريب خاطئ بسبب تعدد الملفات
# * يحافظ على الإحداثيات الطبية الصحيحة

# ---

# ## 2️⃣ `DataSplitter`

# ### 🔬 المشكلة الطبية

# أي نموذج طبي يحتاج:

# * Training
# * Validation
# * Test
#   **بدون تداخل مرضى** (Data Leakage = كارثة علمية)

# ### 🧠 الفكرة العلمية

# * تقسيم عشوائي مضبوط
# * نفس الصورة + الماسك يبقون معًا
# * نسب مدروسة (70/20/10 أو حسب الدراسة)

# ### ⚙️ ما الذي يفعله؟

# * يخلط البيانات
# * يقسمها حسب النسب
# * ينشئ هيكل:

# ```
# train/
#  ├─ images
#  ├─ labels
# valid/
# test/
# ```

# ### 🏥 مثال طبي

# X-ray أسنان:

# * 500 صورة
# * 500 Mask

# الناتج:

# * 350 تدريب
# * 100 تحقق
# * 50 اختبار

# ### 🧪 مثال كود

# ```python
# splitter = DataSplitter(
#     images_dir="images",
#     labels_dir="masks",
#     output_dir="split",
#     train_ratio=0.8,
#     valid_ratio=0.2,
#     test_ratio=0.0
# )
# splitter.run()
# ```

# ### ⚠️ نقطة علمية مهمة

# إذا جعلت `test_ratio = 0`
# → كل الباقي يذهب للـ validation
# **وهذا سبب ظهور أرقام غير متوقعة عندك سابقًا (مثل 762)**

# ---

# ## 3️⃣ `MetadataCopier`

# ### 🔬 مشكلة طبية خطيرة (لكن شائعة)

# أحيانًا:

# * الـ Mask صحيح شكليًا
# * لكن:

#   * Spacing مختلف
#   * Origin مختلف
#   * Direction مختلف
#     ➡️ النتيجة: **Mask لا يطابق الصورة مكانيًا**

# ### 🧠 الحل العلمي

# في الطب:

# * الـ **Metadata** جزء من الحقيقة الفيزيائية للجسم
# * هذا الكلاس:

#   * ينسخ الإحداثيات من الـ Segmentation إلى الـ Volume

# ### ⚙️ ماذا ينسخ؟

# * Origin (نقطة البداية)
# * Spacing (حجم الفوكسل بالملليمتر)
# * Direction (اتجاه المحاور)

# ### 🏥 مثال طبي

# MRI Brain:

# * الصورة Spacing = 1×1×1 mm
# * الماسك Spacing = 0.5×0.5×1 mm ❌

# بعد النسخ:

# * التطابق يصبح مثاليًا

# ### 🧪 مثال كود

# ```python
# copier = MetadataCopier(
#     "volumes",
#     "segmentations",
#     "fixed/volumes",
#     "fixed/segmentations"
# )
# copier.load_and_copy_metadata()
# ```

# ### ✅ هذا الكلاس **إجباري** قبل:

# * nnU-Net
# * MONAI
# * أي تقييم Dice / IoU

# ---

# ## 4️⃣ `DataRenamer`

# ### 🔬 مشكلة nnU-Net تحديدًا

# nnU-Net **صارم جدًا** في أسماء الملفات:

# | النوع    | الصيغة              |
# | -------- | ------------------- |
# | Volume   | `_0000.nii.gz`      |
# | Mask     | `.nii.gz`           |
# | Training | imagesTr / labelsTr |
# | Testing  | imagesTs            |

# ### 🧠 ماذا يفعل؟

# * يحوّل Split YOLO-style
# * إلى nnU-Net official format

# ### 🏥 مثال طبي

# قبل:

# ```
# train/images/ct12.nii.gz
# train/labels/ct12_mask.nii.gz
# ```

# بعد:

# ```
# imagesTr/brain_001_0000.nii.gz
# labelsTr/brain_001.nii.gz
# ```

# ### 🧪 مثال كود

# ```python
# renamer = DataRenamer(
#     path_to_input="split",
#     path_to_output="nnUNet_raw",
#     dataset_id=101,
#     structure="brain"
# )
# renamer.run()
# ```

# ### ✅ لماذا حاسم؟

# * بدون هذا → nnU-Net يرفض التدريب
# * يضمن:

#   * تعدد القنوات مستقبلًا
#   * تنظيم بحثي صحيح

# ---

# ## 🧠 الخلاصة العلمية (Pipeline احترافي)

# في مشروع طبي حقيقي:

# 1️⃣ Merge Classes
# 2️⃣ Fix Metadata
# 3️⃣ Split Data
# 4️⃣ Rename for nnU-Net
# 5️⃣ Train Model

# كل كلاس هنا **ليس كود عشوائي**
# بل يمثل **قاعدة بحثية مستخدمة في أوراق علمية حقيقية**.

# ---
