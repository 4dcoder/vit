# config/dataset_config.py
# This file will store paths and class mappings, making them easily configurable.

import os

# --- Dataset Paths ---
# Adjust these paths based on where you download/extract the datasets
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

TACO_DOWNLOAD_URL = "https://github.com/pedropro/TACO/blob/master/data/annotations.json"
# TACO images are hosted on Flickr and typically downloaded via a script,
# or you might work with a subset from Kaggle.
# For simplicity, we'll assume a local `TACO_images/` folder for images.
TACO_ANNOTATION_PATH = os.path.join(DATA_ROOT, 'TACO', 'data', 'annotations.json')
TACO_IMAGES_DIR = os.path.join(DATA_ROOT, 'TACO', 'images')

TRASH_ICRA19_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'trash_ICRA19'))
# Assuming Trash-ICRA19 uses YOLO-style .txt annotations or similar in its structure.
# You'll need to locate the actual image and annotation folders.
TRASH_ICRA19_IMAGES_DIR = os.path.join(TRASH_ICRA19_ROOT, 'dataset', 'train_test_val', 'images')
TRASH_ICRA19_ANNOTATIONS_DIR = os.path.join(TRASH_ICRA19_ROOT, 'dataset', 'train_test_val', 'labels')
TRASH_ICRA19_CLASS_NAMES_PATH = os.path.join(TRASH_ICRA19_ROOT, 'dataset', 'train_test_val', 'classes.txt')


# --- Class Mappings ---
# Trash-ICRA19 classes: plastic, bio, rov
TRASH_ICRA19_CLASSES = {
    'plastic': 0,
    'bio': 1,
    'rov': 2
}
NUM_CLASSES_ICRA19 = len(TRASH_ICRA19_CLASSES)


# TACO dataset has 60 categories, we need to map them to ICRA19's 'plastic', 'bio', 'rov'.
# The 'plastic' category will be the most complex mapping.
# 'bio' might map to some 'food waste' or 'organic' categories in TACO.
# 'rov' is unique to ICRA19, so no direct mapping from TACO.
# Unmapped TACO classes can be ignored or mapped to a 'background' if needed.

# This is a preliminary mapping. You will refine this after exploring TACO's classes.
# You'll need to examine TACO's `category_id_map` from its JSON.
TACO_TO_ICRA19_CLASS_MAP = {
    # Example for 'plastic' - you'll add many more TACO plastic types here
    'Plastic bag': 'plastic',
    'Plastic bottle': 'plastic',
    'Plastic cap': 'plastic',
    'Plastic wrapper': 'plastic',
    'Clear plastic bottle': 'plastic',
    'Other plastic': 'plastic',
    # Example for 'bio'
    'Food waste': 'bio',
    'Vegetable': 'bio',
    # You might consider if any other TACO categories relate to 'rov' (unlikely)
    # or map to 'background' / ignore
    'Other': 'background', # Example of mapping to a general "other/background"
    'Can': 'background', # Example of ignoring non-ICRA19 target classes
    # ... many more TACO classes ...
}

# Inverse map for easy lookup during parsing if needed
ICRA19_TO_TACO_CLASS_IDS = {
    icra_name: [] for icra_name in TRASH_ICRA19_CLASSES.keys()
}
# This will be populated dynamically in `utils/dataset_parsers.py` based on TACO's actual class IDs
# and your defined `TACO_TO_ICRA19_CLASS_MAP`.

if __name__ == '__main__':
    print("Dataset Configuration:")
    print(f"TACO Annotations: {TACO_ANNOTATION_PATH}")
    print(f"TACO Images: {TACO_IMAGES_DIR}")
    print(f"Trash-ICRA19 Images: {TRASH_ICRA19_IMAGES_DIR}")
    print(f"Trash-ICRA19 Annotations: {TRASH_ICRA19_ANNOTATIONS_DIR}")
    print(f"Trash-ICRA19 Classes: {TRASH_ICRA19_CLASSES}")
    print(f"TACO to ICRA19 Class Map (partial example): {TACO_TO_ICRA19_CLASS_MAP}")