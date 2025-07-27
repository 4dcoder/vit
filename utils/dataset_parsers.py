# utils/dataset_parsers.py
# This file will contain functions to parse the raw dataset annotations.

import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import glob

# from config.dataset_config import TRASH_ICRA19_CLASSES, TACO_TO_ICRA19_CLASS_MAP

# Define these locally for now to avoid circular import if needed,
# or ensure config is loaded appropriately in main scripts.
TRASH_ICRA19_CLASSES = {
    'plastic': 0,
    'bio': 1,
    'rov': 2
}

# This will be dynamically generated after inspecting TACO's actual categories
TACO_CATEGORY_MAP_ID_TO_NAME = {}
TACO_TO_ICRA19_CLASS_MAP = {
    # This mapping will be filled more comprehensively after inspecting TACO data.
    # For now, a minimal example.
    # TACO categories that map to ICRA19 'plastic'
    'Plastic bag': 'plastic',
    'Plastic bottle': 'plastic',
    'Plastic cap': 'plastic',
    'Plastic wrapper': 'plastic',
    'Clear plastic bottle': 'plastic',
    'Other plastic': 'plastic',
    'Plastic film': 'plastic',
    'Plastic container': 'plastic',
    'Plastic lid': 'plastic',
    'Plastic cutlery': 'plastic',
    'Plastic straw': 'plastic',
    'Plastic cup': 'plastic',

    # TACO categories that map to ICRA19 'bio'
    'Food waste': 'bio',
    'Vegetable': 'bio',
    'Fruit': 'bio',
    'Paper towel': 'bio', # Assuming some organic paper waste goes here

    # Classes to ignore or map to 'background' as they don't fit ICRA19's targets
    'Can': 'background',
    'Glass bottle': 'background',
    'Cigarette': 'background',
    'Cardboard': 'background',
    'Paper': 'background',
    'Metal': 'background',
    'Other': 'background',
    'Battery': 'background',
    'Bottle cap': 'background', # Generic bottle cap, might be plastic or metal, cautious default
    'Crisp packet': 'plastic', # Often plastic
    'Drink can': 'plastic', # Most soft drink cans have plastic lining or are aluminum. Mapping to plastic is a simplification, or can be 'background'
    'Disposable plastic cup': 'plastic',
    'Foam piece': 'plastic', # Styrofoam etc.
    'Glass jar': 'background',
    'Plastic bottle cap': 'plastic', # Explicit plastic cap
    'Plastic container (other)': 'plastic',
    'Pop tab': 'background',
    'Straw': 'plastic', # Generic straw, assume plastic
    'Styrofoam piece': 'plastic',
    'Unlabeled litter': 'background', # Cannot determine type
    'Plastic bottle (other)': 'plastic', # More general plastic bottle
    'Aluminium foil': 'background',
    'Aluminium blister pack': 'background',
    'Carton': 'background',
    'Plastic gloving': 'plastic',
    'Plastic food container': 'plastic',
    'Bottle': 'plastic', # Generic bottle, assume plastic for now if no other info
    'Glass': 'background',
    'Metal can': 'background',
    'Carded blister pack': 'background',
    'Magazine': 'background',
    'Newspaper': 'background',
    'Wrapper': 'plastic', # Generic wrapper, assume plastic
    'Cup': 'plastic', # Generic cup, assume plastic
    'Paper cup': 'background',
    'Drink': 'plastic', # Generic drink, assume plastic container
    'Plastic packaging': 'plastic',
    'Clear plastic container': 'plastic',
    'Other plastic packaging': 'plastic',
    'Plastic bag (other)': 'plastic',
    'Plastic film (other)': 'plastic',
    'Bottle (other)': 'plastic',
    'Can (other)': 'background',
    'Plastic cup (other)': 'plastic',
    'Other plastic bottle': 'plastic',
    'Other drink': 'plastic', # Assume plastic container
    'Foam cup': 'plastic',
    'Glass (other)': 'background',
    'Bottle (glass)': 'background',
    'Can (aluminum)': 'background',
    'Food pouch': 'plastic',
    'Bottle (plastic)': 'plastic', # Explicit plastic bottle
    'Plastic piece': 'plastic',
    'Other metal': 'background',
    'Paper (other)': 'background',
    'Six pack rings': 'plastic',
    'Shoe': 'background',
    'Scrap metal': 'background',
    'Other bio': 'bio',
    'Plastic (other)': 'plastic', # Catch-all for plastics
}


def parse_taco_annotations(json_path, images_dir):
    """
    Parses TACO dataset annotations (COCO format).
    Returns a list of dictionaries, each representing an image with its annotations.
    Annotations are converted to a consistent format (image_path, boxes, labels).
    Boxes are in [x_min, y_min, width, height] format.
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Build category ID to name mapping for TACO
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    global TACO_CATEGORY_MAP_ID_TO_NAME
    TACO_CATEGORY_MAP_ID_TO_NAME = category_id_to_name

    image_annotations = defaultdict(list)
    images_info = {img['id']: img for img in coco_data['images']}

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_name = category_id_to_name[ann['category_id']]

        # Map TACO category to ICRA19 equivalent or 'background'
        icra19_class_name = TACO_TO_ICRA19_CLASS_MAP.get(category_name)

        if icra19_class_name and icra19_class_name != 'background':
            # Get the numerical ID for ICRA19 class
            icra19_class_id = TRASH_ICRA19_CLASSES.get(icra19_class_name)

            if icra19_class_id is not None:
                bbox = ann['bbox'] # [x, y, width, height]
                image_annotations[image_id].append({
                    'bbox': bbox,
                    'category_id': icra19_class_id,
                    'category_name_taco': category_name, # Keep original for debugging
                    'category_name_icra19': icra19_class_name
                })

    parsed_data = []
    for img_id, anns in image_annotations.items():
        img_info = images_info[img_id]
        image_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}. Skipping.")
            continue

        boxes = [a['bbox'] for a in anns]
        labels = [a['category_id'] for a in anns]

        parsed_data.append({
            'image_id': img_id,
            'image_path': image_path,
            'width': img_info['width'],
            'height': img_info['height'],
            'boxes': boxes, # List of [x, y, w, h]
            'labels': labels # List of integer class IDs
        })
    return parsed_data


def parse_trash_icra19_annotations(images_dir, annotations_dir, class_names_path):
    """
    Parses Trash-ICRA19 dataset annotations (assuming YOLO .txt format or similar structure).
    Returns a list of dictionaries, each representing an image with its annotations.
    Annotations are converted to a consistent format (image_path, boxes, labels).
    Boxes are in [x_min, y_min, width, height] (absolute pixel values).
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized)
    """
    icra19_class_names = []
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            icra19_class_names = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: classes.txt not found at {class_names_path}. Using hardcoded class names.")
        icra19_class_names = list(TRASH_ICRA19_CLASSES.keys()) # Fallback

    parsed_data = []
    image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                  glob.glob(os.path.join(images_dir, '*.png'))

    for img_path in image_files:
        file_name_stem = os.path.splitext(os.path.basename(img_path))[0]
        annotation_path_txt = os.path.join(annotations_dir, file_name_stem + '.txt')

        if not os.path.exists(annotation_path_txt):
            # print(f"Warning: Annotation file not found for {img_path}. Skipping.")
            continue # Skip images without annotations

        boxes = []
        labels = []
        try:
            with open(annotation_path_txt, 'r') as f:
                # Read image dimensions to un-normalize YOLO boxes
                # This is a common requirement for YOLO parsing.
                # If image dimensions are not in annotations, you need to load image itself.
                from PIL import Image
                img = Image.open(img_path)
                img_width, img_height = img.size

                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    # Convert normalized YOLO (cx, cy, w, h) to absolute (xmin, ymin, w, h)
                    x_center, y_center, width_norm, height_norm = map(float, parts[1:5])

                    abs_width = width_norm * img_width
                    abs_height = height_norm * img_height
                    x_min = (x_center * img_width) - (abs_width / 2)
                    y_min = (y_center * img_height) - (abs_height / 2)

                    boxes.append([x_min, y_min, abs_width, abs_height])
                    labels.append(class_id) # Already integer ID from YOLO format

                parsed_data.append({
                    'image_path': img_path,
                    'width': img_width,
                    'height': img_height,
                    'boxes': boxes, # List of [x_min, y_min, width, height]
                    'labels': labels # List of integer class IDs
                })
        except Exception as e:
            print(f"Error parsing {annotation_path_txt}: {e}. Skipping.")
            continue

    return parsed_data

if __name__ == '__main__':
    # This block is for testing the parsers directly
    print("--- Testing Dataset Parsers ---")
    from config.dataset_config import TACO_ANNOTATION_PATH, TACO_IMAGES_DIR, \
                                    TRASH_ICRA19_IMAGES_DIR, TRASH_ICRA19_ANNOTATIONS_DIR, \
                                    TRASH_ICRA19_CLASS_NAMES_PATH

    print("\nParsing TACO Dataset (might take a while for large dataset)...")
    if os.path.exists(TACO_ANNOTATION_PATH):
        taco_data = parse_taco_annotations(TACO_ANNOTATION_PATH, TACO_IMAGES_DIR)
        print(f"Parsed {len(taco_data)} images from TACO.")
        if taco_data:
            print("Sample TACO entry (mapped to ICRA19 format):")
            sample = taco_data[0]
            print(f"Image Path: {sample['image_path']}")
            print(f"Boxes (first 2): {sample['boxes'][:2]}")
            print(f"Labels (first 2): {sample['labels'][:2]}")
            print(f"Original TACO categories mapped to ICRA19: {TACO_CATEGORY_MAP_ID_TO_NAME}")
            # print("Mapped TACO categories to ICRA19 classes:", {k: TACO_TO_ICRA19_CLASS_MAP[k] for k in TACO_TO_ICRA19_CLASS_MAP if k in TACO_CATEGORY_MAP_ID_TO_NAME.values()}) # This needs refinement

            # Quick check for mapping consistency
            mapped_classes_count = defaultdict(int)
            for entry in taco_data:
                for label_id in entry['labels']:
                    # Assuming labels are already ICRA19 IDs
                    for class_name, class_id in TRASH_ICRA19_CLASSES.items():
                        if class_id == label_id:
                            mapped_classes_count[class_name] += 1
                            break
            print("Distribution of mapped ICRA19 classes in TACO data:")
            for cls, count in mapped_classes_count.items():
                print(f"- {cls}: {count} instances")

    else:
        print(f"TACO annotation JSON not found at {TACO_ANNOTATION_PATH}. Please download TACO first.")

    print("\nParsing Trash-ICRA19 Dataset...")
    if os.path.exists(TRASH_ICRA19_IMAGES_DIR) and os.path.exists(TRASH_ICRA19_ANNOTATIONS_DIR):
        icra19_data = parse_trash_icra19_annotations(TRASH_ICRA19_IMAGES_DIR, TRASH_ICRA19_ANNOTATIONS_DIR, TRASH_ICRA19_CLASS_NAMES_PATH)
        print(f"Parsed {len(icra19_data)} images from Trash-ICRA19.")
        if icra19_data:
            print("Sample Trash-ICRA19 entry:")
            sample = icra19_data[0]
            print(f"Image Path: {sample['image_path']}")
            print(f"Boxes (first 2): {sample['boxes'][:2]}")
            print(f"Labels (first 2): {sample['labels'][:2]}")
            # Print actual class names for ICRA19 for verification
            print("ICRA19 Classes:", TRASH_ICRA19_CLASSES)
    else:
        print(f"Trash-ICRA19 image/annotation directories not found. Please check paths in {os.path.join(os.path.dirname(__file__), '..', 'config', 'dataset_config.py')}.")