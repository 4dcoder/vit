# datasets/trash_detection_dataset.py
# This will be your PyTorch Dataset class.

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as T
import random

# Import the parsing functions (ensure paths are correct for import)
try:
    from utils.dataset_parsers import parse_taco_annotations, parse_trash_icra19_annotations
    from config.dataset_config import TRASH_ICRA19_CLASSES, TACO_ANNOTATION_PATH, TACO_IMAGES_DIR, \
                                      TRASH_ICRA19_IMAGES_DIR, TRASH_ICRA19_ANNOTATIONS_DIR, \
                                      TRASH_ICRA19_CLASS_NAMES_PATH, NUM_CLASSES_ICRA19
except ImportError:
    # Fallback for direct testing or if run from a different directory
    print("Warning: Could not import from config/utils. Assuming running in a test context.")
    # Define placeholder values for TRASH_ICRA19_CLASSES if not imported
    TRASH_ICRA19_CLASSES = {
        'plastic': 0,
        'bio': 1,
        'rov': 2
    }
    NUM_CLASSES_ICRA19 = len(TRASH_ICRA19_CLASSES)
    # Dummy parse functions for testing outside main project
    def parse_taco_annotations(*args, **kwargs): return []
    def parse_trash_icra19_annotations(*args, **kwargs): return []
    TACO_ANNOTATION_PATH, TACO_IMAGES_DIR = '', ''
    TRASH_ICRA19_IMAGES_DIR, TRASH_ICRA19_ANNOTATIONS_DIR, TRASH_ICRA19_CLASS_NAMES_PATH = '', '', ''


class TrashDetectionDataset(Dataset):
    def __init__(self, dataset_name, split='train', transforms=None,
                 taco_json_path=None, taco_images_dir=None,
                 icra19_images_dir=None, icra19_annotations_dir=None,
                 icra19_class_names_path=None):
        """
        Args:
            dataset_name (str): 'taco' or 'trash_icra19'.
            split (str): 'train', 'val', or 'test'. Used for ICRA19, ignored for TACO (TACO is usually full dataset).
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transforms = transforms
        self.data = []
        self.class_to_idx = TRASH_ICRA19_CLASSES
        self.idx_to_class = {v: k for k, v in TRASH_ICRA19_CLASSES.items()}
        self.num_classes = NUM_CLASSES_ICRA19

        if dataset_name == 'taco':
            if taco_json_path and taco_images_dir:
                print(f"Loading TACO dataset from {taco_json_path}...")
                self.data = parse_taco_annotations(taco_json_path, taco_images_dir)
            else:
                print("TACO paths not provided. Using default from config.")
                self.data = parse_taco_annotations(TACO_ANNOTATION_PATH, TACO_IMAGES_DIR)
            print(f"Loaded {len(self.data)} images from TACO (mapped to ICRA19 classes).")
        elif dataset_name == 'trash_icra19':
            if icra19_images_dir and icra19_annotations_dir and icra19_class_names_path:
                print(f"Loading Trash-ICRA19 dataset (split: {split}) from {icra19_images_dir}...")
                # The Trash-ICRA19 dataset structure implies that the 'train', 'test', 'val'
                # subdirectories are under the main images/labels dirs.
                # Adjusting paths based on your provided structure:
                # `TRASH_ICRA19_IMAGES_DIR` is already `.../dataset/train_test_val/images`
                # So we need to filter images by split name.
                # For simplicity in this base class, we will load ALL images from the specified
                # `icra19_images_dir` and handle splits externally or assume `icra19_images_dir`
                # points to the specific split's image folder (e.g., `.../images/train/`).
                # Based on your prompt, `TRASH_ICRA19_ROOT` contains `dataset/train_test_val/`.
                # We need to refine `parse_trash_icra19_annotations` or here to handle the splits.

                # Let's assume for ICRA19, the `images_dir` and `annotations_dir` point to the
                # specific split folder (e.g., `TRASH_ICRA19_ROOT/dataset/train_test_val/images/train`).
                _images_dir = os.path.join(icra19_images_dir, split)
                _annotations_dir = os.path.join(icra19_annotations_dir, split)

                self.data = parse_trash_icra19_annotations(_images_dir, _annotations_dir, icra19_class_names_path)
            else:
                print("Trash-ICRA19 paths not provided. Using default from config.")
                _images_dir = os.path.join(TRASH_ICRA19_IMAGES_DIR, split)
                _annotations_dir = os.path.join(TRASH_ICRA19_ANNOTATIONS_DIR, split)
                self.data = parse_trash_icra19_annotations(_images_dir, _annotations_dir, TRASH_ICRA19_CLASS_NAMES_PATH)
            print(f"Loaded {len(self.data)} images from Trash-ICRA19 (split: {split}).")
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}. Choose 'taco' or 'trash_icra19'.")

        if not self.data:
            print(f"No data loaded for {dataset_name} ({split} split). Check paths and data existence.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']
        image = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(item['boxes'], dtype=torch.float32) # [N, 4] in [xmin, ymin, w, h]
        labels = torch.tensor(item['labels'], dtype=torch.int64) # [N]

        # Convert [xmin, ymin, w, h] to [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2] # w -> xmax, h -> ymax

        # Create target dictionary as expected by object detection models (e.g., DETR/DINO)
        # Note: DETR/DINO typically expects normalized boxes (0-1) in cxcywh format
        # We will handle this normalization and format conversion in the transforms or within the training loop
        # For now, keep as [xmin, ymin, xmax, ymax] absolute pixels.
        target = {
            "boxes": boxes, # Absolute pixel values, [xmin, ymin, xmax, ymax]
            "labels": labels,
            "image_id": torch.tensor([item['image_id']]) if 'image_id' in item else torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), # Area of boxes
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64) # Assuming no crowd annotations
        }

        if self.transforms:
            image, target = self.transforms(image, target) # Custom transform that handles image and target

        return image, target


# Custom collate_fn for object detection (handles variable number of objects per image)
def collate_fn(batch):
    images, targets = zip(*batch)
    # Ensure all images are tensors and targets are lists of dicts
    # For object detection, we usually just stack images and keep targets as a list of dicts
    images = torch.stack(images, 0)
    return images, list(targets)


if __name__ == '__main__':
    print("--- Testing TrashDetectionDataset ---")
    from config.dataset_config import TACO_ANNOTATION_PATH, TACO_IMAGES_DIR, \
                                      TRASH_ICRA19_IMAGES_DIR, TRASH_ICRA19_ANNOTATIONS_DIR, \
                                      TRASH_ICRA19_CLASS_NAMES_PATH, TRASH_ICRA19_CLASSES

    # Dummy transforms for testing
    class DummyTransforms:
        def __call__(self, img, target):
            # Resize image to a common size, normalize pixels
            img = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)

            # Important: Bounding boxes also need to be scaled if the image is resized.
            # This is a simplified dummy, actual transforms need proper handling of boxes.
            original_width, original_height = target['width'], target['height'] # Assume target provides original size
            if 'width' in target and 'height' in target:
                 # Normalize boxes to 0-1 range based on original image size
                boxes_normalized = target['boxes'] / torch.tensor([original_width, original_height, original_width, original_height], dtype=torch.float32)

                # Convert to cxcywh for models like DETR if needed (will do later with real transforms)
                # For now, just show normalized xmin, ymin, xmax, ymax
                target['boxes'] = boxes_normalized
                del target['width'] # Remove original width/height after normalization
                del target['height']
            else:
                 # Fallback if width/height not in target, assume image info is available
                 print("Warning: Original image dimensions not found in target for transform. Box scaling skipped.")


            return img, target

    # Test TACO Dataset
    print("\n--- Testing TACO Dataset ---")
    if os.path.exists(TACO_ANNOTATION_PATH) and os.path.exists(TACO_IMAGES_DIR):
        # We need to ensure `image_id`, `width`, `height` are correctly passed
        # from `parse_taco_annotations` into the `self.data` items for `__getitem__`.
        # The `parse_taco_annotations` already provides this!

        # Define an actual transform that can handle both image and target
        class DetectionTransforms:
            def __init__(self, size=(224, 224)):
                self.size = size
                self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            def __call__(self, img, target):
                original_width, original_height = img.size

                # Resize image
                img = T.functional.resize(img, self.size)
                img = T.functional.to_tensor(img)
                img = self.normalize(img)

                # Scale bounding boxes according to new image size
                boxes = target['boxes'] # [xmin, ymin, xmax, ymax] absolute
                scale_x = self.size[0] / original_width
                scale_y = self.size[1] / original_height
                
                boxes[:, 0] *= scale_x # xmin
                boxes[:, 1] *= scale_y # ymin
                boxes[:, 2] *= scale_x # xmax
                boxes[:, 3] *= scale_y # ymax

                # Clip boxes to image boundaries (0 to size-1)
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, self.size[0] - 1)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, self.size[1] - 1)

                # Convert to cxcywh normalized format (expected by DETR/DINO)
                # x_center = (xmin + xmax) / 2 / width
                # y_center = (ymin + ymax) / 2 / height
                # w = (xmax - xmin) / width
                # h = (ymax - ymin) / height
                boxes_cxcywh = torch.empty_like(boxes)
                boxes_cxcywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 / self.size[0] # x_center
                boxes_cxcywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / self.size[1] # y_center
                boxes_cxcywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / self.size[0]   # width
                boxes_cxcywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / self.size[1]   # height

                target['boxes'] = boxes_cxcywh
                target['orig_size'] = torch.tensor([original_height, original_width]) # For detr/dino
                target['size'] = torch.tensor([self.size[1], self.size[0]]) # For detr/dino

                return img, target

        taco_dataset = TrashDetectionDataset(
            dataset_name='taco',
            transforms=DetectionTransforms(size=(800, 800)), # Common input size for DETR/DINO
            taco_json_path=TACO_ANNOTATION_PATH,
            taco_images_dir=TACO_IMAGES_DIR
        )
        taco_dataloader = DataLoader(taco_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

        for i, (images, targets) in enumerate(taco_dataloader):
            print(f"\nTACO Batch {i+1}:")
            print(f"Images batch shape: {images.shape}")
            print(f"Number of targets in batch: {len(targets)}")
            for j, target in enumerate(targets):
                print(f"  Target {j}: Num boxes={len(target['boxes'])}, Labels={target['labels'].tolist()}")
                print(f"  Sample Box (normalized cxcywh): {target['boxes'][0].tolist() if len(target['boxes']) > 0 else 'N/A'}")
                print(f"  Original Size: {target['orig_size'].tolist()}, Current Size: {target['size'].tolist()}")
            if i >= 1: break # Print only 2 batches

    else:
        print("Skipping TACO dataset test: paths not found or config missing.")

    # Test Trash-ICRA19 Dataset
    print("\n--- Testing Trash-ICRA19 Dataset (Train Split) ---")
    if os.path.exists(os.path.join(TRASH_ICRA19_IMAGES_DIR, 'train')) and \
       os.path.exists(os.path.join(TRASH_ICRA19_ANNOTATIONS_DIR, 'train')):
        icra19_train_dataset = TrashDetectionDataset(
            dataset_name='trash_icra19',
            split='train',
            transforms=DetectionTransforms(size=(800, 800)),
            icra19_images_dir=TRASH_ICRA19_IMAGES_DIR,
            icra19_annotations_dir=TRASH_ICRA19_ANNOTATIONS_DIR,
            icra19_class_names_path=TRASH_ICRA19_CLASS_NAMES_PATH
        )
        icra19_train_dataloader = DataLoader(icra19_train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

        for i, (images, targets) in enumerate(icra19_train_dataloader):
            print(f"\nICRA19 Train Batch {i+1}:")
            print(f"Images batch shape: {images.shape}")
            print(f"Number of targets in batch: {len(targets)}")
            for j, target in enumerate(targets):
                print(f"  Target {j}: Num boxes={len(target['boxes'])}, Labels={target['labels'].tolist()}")
                print(f"  Sample Box (normalized cxcywh): {target['boxes'][0].tolist() if len(target['boxes']) > 0 else 'N/A'}")
                print(f"  Original Size: {target['orig_size'].tolist()}, Current Size: {target['size'].tolist()}")
            if i >= 1: break # Print only 2 batches
    else:
        print("Skipping Trash-ICRA19 dataset test: train split paths not found or config missing.")