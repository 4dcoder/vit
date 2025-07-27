## Project: Underwater Trash Detection & Visual QA for UAVs

**Objective:** Develop and evaluate efficient Vision Transformer (ViT) based object detection models for underwater trash and integrate with a lightweight Visual Question Answering (VQA) system for deployment on NVIDIA Jetson Orin Nano.

**Evaluation Metrics:**

* **Object Detection:** mAP (mean Average Precision), mAP@0.5, mAP@0.75, Inference Latency (ms), FPS (Frames Per Second), Model Size (MB).
* **VQA:** Accuracy (for classification-based VQA), BLEU/ROUGE (for generative VQA, if applicable), Inference Latency (ms), FPS.

---

### Phase 1: Data Preparation & Exploration (Server)

**Goal:** Prepare and understand the datasets for object detection and lay the groundwork for VQA.

**Action Plan:**

* **1.1. Dataset Acquisition & Organization:**
  * Download `[TACO](https://github.com/pedropro/TACO.git) (Trash Annotations in Context)` into `dataset`.
  * Ensure `[Trash-ICRA19](https://conservancy.umn.edu/bitstreams/0239b06a-512e-49c3-80aa-ba33371e11de/download)` is correctly located in `project_root/dataset/trash_ICRA19/`.
  * **Python Jupyter Notebook:** `01_data_acquisition_and_exploration.ipynb`
    * Contains scripts to download (if not already local), extract, and organize datasets.
    * Initial data statistics (number of images, classes, bounding box distributions).
    * Sample image visualizations with annotations.
* **1.2. Data Preprocessing & Custom PyTorch Dataset:**
  * **Objective:** Convert `TACO` and `Trash-ICRA19` annotations to a unified format (e.g., COCO-style JSON compatible with Hugging Face's `datasets` or custom PyTorch `Dataset`).
  * Implement class mapping: `TACO` classes to `Trash-ICRA19` (`plastic`, `bio`, `rov`). Focus on mapping diverse "plastic" categories from `TACO` to `Trash-ICRA19`'s `plastic`. Other TACO classes might be grouped as "other" or ignored for the primary `Trash-ICRA19` fine-tuning.
  * Split `Trash-ICRA19` into train/validation/test sets if not already provided in a suitable split for fine-tuning.
  * **Python Jupyter Notebook:** `02_data_preprocessing_and_dataset.ipynb`
  * **Utility Class:** `utils/dataset_parsers.py`
    * Functions to parse raw annotations (`TACO`, `Trash-ICRA19`) into a standardized internal format.
  * **Utility Class:** `datasets/trash_detection_dataset.py`
    * PyTorch `Dataset` class for object detection, handling image loading, annotation parsing, and initial transformations.
    * `__getitem__` method should return `(image_tensor, target_dict)` where `target_dict` includes `boxes` (cxcywh, normalized) and `labels`.

**High-Level Implementation:**

* **`project_root/` structure:**
  ```
  project_root/
  ├── notebooks/
  │   ├── 01_data_acquisition_and_exploration.ipynb
  │   └── 02_data_preprocessing_and_dataset.ipynb
  ├── datasets/
  │   ├── trash_detection_dataset.py
  │   └── custom_vqa_dataset.py # Placeholder for Phase 3
  ├── utils/
  │   ├── dataset_parsers.py
  │   ├── data_augmentations.py
  │   └── metrics.py
  ├── config/
  │   └── dataset_config.py # Python file for dataset paths, class mappings etc.
  ├── models/ # Will contain saved checkpoints
  └── trash_ICRA19/ # Existing dataset folder
  ```

---

### Phase 2: ViT Object Detection Training & Evaluation (Server)

**Goal:** Train and fine-tune selected ViT object detection models, compare their performance on Trash-ICRA19 (validation set), and prepare for Orin Nano deployment.

**Action Plan:**

* **2.1. Model Definition & Training Configuration:**
  * Define training parameters (epochs, batch size, learning rate, optimizer, scheduler).
  * Utilize Hugging Face `transformers` for loading pre-trained ViT models with object detection heads (DINO/DETR).
  * **Python Jupyter Notebook:** `03_vit_detection_training.ipynb`
    * Contains the main training loop.
    * Will import models from `models/` directory.
  * **Utility Class:** `models/vit_detection_models.py`
    * Functions to load `DinoForObjectDetection` with `Swin-Small` and `DeiT-Small` (or similar efficient variant).
    * Helper function to adjust the detection head for 3 classes (`plastic`, `bio`, `rov`).
  * **Utility Class:** `utils/data_augmentations.py`
    * Implement the extensive data augmentation techniques (random rotations, color jitter, CutMix, Mixup etc.) using `torchvision.transforms` or `Albumentations`.
  * **Utility Class:** `utils/train_utils.py`
    * `train_epoch`, `validate_epoch` functions.
    * Metric computation (mAP).
    * Logging (TensorBoard/Weights & Biases integration).
* **2.2. Training on Primary Dataset (TACO/Large Trash Dataset):**
  * Train `DinoForObjectDetection` with `Swin-Small` backbone.
  * Train `DinoForObjectDetection` with `DeiT-Small` backbone.
  * Store checkpoints.
* **2.3. Fine-tuning on `Trash-ICRA19`:**
  * Load the best performing models from 2.2.
  * Fine-tune them specifically on the `Trash-ICRA19` dataset (using a dedicated training/validation split from `Trash-ICRA19`).
  * This is the critical domain adaptation step.
* **2.4. Performance Evaluation:**
  * Evaluate both fine-tuned models on the `Trash-ICRA19` test set.
  * Record mAP, mAP@0.5, mAP@0.75.
  * Initial inference speed benchmarking (on CPU/GPU of server, not Orin Nano yet) to get a baseline.
  * **Python Jupyter Notebook:** `04_vit_detection_evaluation.ipynb`
    * Load trained models.
    * Run inference on the test set.
    * Compute and display all relevant metrics.
    * Visualize sample predictions.

**High-Level Implementation:**

* **`project_root/` structure (continued):**
  ```
  project_root/
  ├── notebooks/
  │   ├── ...
  │   ├── 03_vit_detection_training.ipynb
  │   └── 04_vit_detection_evaluation.ipynb
  ├── datasets/
  │   └── ...
  ├── utils/
  │   ├── ...
  │   ├── train_utils.py
  │   └── metrics.py
  ├── models/
  │   ├── vit_detection_models.py
  │   └── checkpoints/ # Saved model weights
  └── config/
      └── training_config.py # Hyperparameters
  ```

---

### Phase 3: Visual Question Answering (VQA) Development (Server)

**Goal:** Create a custom VQA dataset for trash and build/train lightweight VQA models for comparison.

**Action Plan:**

* **3.1. Custom VQA Dataset Creation:**
  * **Objective:** Generate relevant question-answer pairs specific to the `Trash-ICRA19` domain (e.g., "Is this plastic?", "What type of trash is near the ROV?", "How many bio-waste items are there?").
  * Leverage object detection results (bounding boxes and class labels from Phase 2) to programmatically generate questions and answers (e.g., if a "plastic" object is detected, generate "Is this plastic? Yes.").
  * Manual annotation will be required for more complex or open-ended questions.
  * **Python Jupyter Notebook:** `05_vqa_dataset_creation.ipynb`
    * Scripts for automated Q&A generation based on object detection.
    * Interface for manual review/addition of Q&A.
  * **Utility Class:** `datasets/custom_vqa_dataset.py`
    * PyTorch `Dataset` class for VQA, handling image loading, question tokenization, and answer encoding.
* **3.2. VQA Model Selection & Training:**
  * **Model 1 (Simpler Fusion):**
    * Vision Encoder: Use the fine-tuned Swin-Small from Phase 2 (extract features from its backbone).
    * Language Encoder: `DistilBERT` or `TinyBERT` (from Hugging Face `transformers`).
    * Fusion: Concatenate image features and language embeddings, pass through a small MLP classifier to predict answers (from a fixed vocabulary of answers like "yes", "no", "plastic", "bio", "rov", numbers 1-5, etc.).
  * **Model 2 (Lightweight VLM):**
    * Research and select a pre-trained "Tiny" VLM designed for efficiency, if available (e.g., a very small version of ViLT, if its pre-trained weights are accessible and adaptable).
  * **Python Jupyter Notebook:** `06_vqa_training.ipynb`
    * Training loops for both VQA models.
    * VQA-specific loss functions and metrics.
  * **Utility Class:** `models/vqa_models.py`
    * Implement the VQA model architectures.
* **3.3. VQA Performance Evaluation:**
  * Evaluate both trained VQA models on your custom VQA test set.
  * Record accuracy, and potentially other text generation metrics if applicable.
  * **Python Jupyter Notebook:** `07_vqa_evaluation.ipynb`
    * Load VQA models.
    * Run inference.
    * Compute metrics.
    * Demonstrate interactive Q&A examples.

**High-Level Implementation:**

* **`project_root/` structure (continued):**
  ```
  project_root/
  ├── notebooks/
  │   ├── ...
  │   ├── 05_vqa_dataset_creation.ipynb
  │   ├── 06_vqa_training.ipynb
  │   └── 07_vqa_evaluation.ipynb
  ├── datasets/
  │   ├── ...
  │   └── custom_vqa_dataset.py
  │   └── custom_vqa_data/ # Generated Q&A files
  ├── utils/
  │   └── ...
  ├── models/
  │   ├── ...
  │   ├── vqa_models.py
  │   └── checkpoints/ # VQA model weights
  └── config/
      └── vqa_config.py # VQA specific parameters
  ```

---

### Phase 4: Model Optimization & Onboard Deployment (Server & Orin Nano)

**Goal:** Optimize the best performing object detection and VQA models for Orin Nano and deploy them for inference.

**Action Plan:**

* **4.1. Model Export & ONNX Conversion (Server):**
  * Export the fine-tuned PyTorch models (best Swin, best DeiT, and best VQA models) to ONNX format. This is the intermediate format for TensorRT.
  * **Python Jupyter Notebook:** `08_model_export_onnx.ipynb`
    * Scripts to load PyTorch models and export to `.onnx` files.
* **4.2. TensorRT Optimization (Orin Nano):**
  * Transfer ONNX models to the Orin Nano.
  * Use `trtexec` or the Python TensorRT API to build optimized TensorRT engines.
  * Experiment with different precision: FP16 (recommended first), INT8 (requires calibration data).
  * **Python Script (on Orin Nano):** `deploy/optimize_tensorrt.py`
    * Takes ONNX path, outputs TensorRT engine path.
* **4.3. Onboard Inference Integration (Orin Nano):**
  * Develop the Python application for real-time inference on the Orin Nano.
  * Integrate camera feed (e.g., using GStreamer or OpenCV `VideoCapture`).
  * Load TensorRT engines for both object detection and VQA.
  * Implement the full perception-to-VQA pipeline.
  * **Python Script (on Orin Nano):** `deploy/uav_inference_pipeline.py`
    * Main script for the UAV.
    * Utilizes `deploy/tensorrt_utils.py`.
  * **Utility Class:** `deploy/tensorrt_utils.py`
    * Helper functions to load TensorRT engines, perform inference, and post-process results.
* **4.4. Performance Benchmarking on Orin Nano:**
  * Measure actual inference latency and FPS for both object detection and VQA on the Orin Nano.
  * Compare the performance of `Swin` vs. `DeiT` (for detection) and your two VQA models.
  * Measure power consumption during inference.
  * **Python Jupyter Notebook/Script (on Orin Nano):** `09_orin_nano_benchmarking.ipynb`
    * Load TensorRT engines.
    * Run timed inference on sample data/live feed.
    * Log performance metrics.

**High-Level Implementation:**

* **`project_root/` structure (continued):**
  ```
  project_root/
  ├── notebooks/
  │   ├── ...
  │   └── 08_model_export_onnx.ipynb
  │   └── 09_orin_nano_benchmarking.ipynb (can be run on Nano with Jupyter)
  ├── deploy/
  │   ├── optimize_tensorrt.py
  │   ├── tensorrt_utils.py
  │   └── uav_inference_pipeline.py
  ├── models/
  │   ├── ...
  │   └── exported_onnx/ # ONNX models
  │   └── tensorrt_engines/ # TensorRT engines (on Nano)
  ├── config/
  │   └── deployment_config.py # Paths to engines, camera settings etc.
  └── (other project folders)
  ```

---

### Phase 5: Autonomous Navigation Integration & System Testing (Orin Nano)

**Goal:** Integrate the developed perception and VQA capabilities with a basic navigation stack and conduct full system testing.

**Action Plan:**

* **5.1. Navigation Interface:**
  * Connect your inference pipeline to a UAV flight controller via a communication protocol (e.g., MAVLink/ROS2).
  * Develop basic navigation commands based on detection and VQA outputs (e.g., "move towards plastic," "avoid bio-waste"). This can be rule-based initially.
  * **Python Script (on Orin Nano):** Extend `deploy/uav_inference_pipeline.py` or create `deploy/uav_navigation_control.py`.
* **5.2. System Integration & Testing:**
  * Conduct comprehensive tests of the integrated system in a simulated environment (e.g., Gazebo, AirSim) or a controlled real-world environment (e.g., a test tank for underwater scenarios).
  * Test different scenarios: varying trash types, lighting, turbidity, complex questions.
  * Log all data (sensor, inference results, navigation commands, power usage).
  * **Python Script (on Orin Nano):** `deploy/full_system_test.py`
* **5.3. Analysis & Refinement:**
  * Analyze logs to identify bottlenecks, errors, and areas for improvement (e.g., model accuracy, inference speed, power efficiency, navigation stability).
  * Iterate on model optimization or navigation logic.

---

This action plan provides a strong roadmap. The next step is to start writing the actual code, beginning with the data preparation phase. We'll start with the `01_data_acquisition_and_exploration.ipynb` and `02_data_preprocessing_and_dataset.ipynb` notebooks and their associated utility files.


**Run Jupyter Notebooks:**

* Open a terminal in your `project_root/`.
* Start Jupyter: `jupyter notebook`
* Navigate to `notebooks/`.
* **First, run `01_data_acquisition_and_exploration.ipynb`:** This notebook will guide you through the dataset acquisition (which is mostly manual steps) and provide initial statistics and visualizations. **Pay close attention to the `TACO_TO_ICRA19_CLASS_MAP` in `utils/dataset_parsers.py` and refine it based on the actual TACO categories you see. This is a crucial step for effective transfer learning.**
* **Then, run `02_data_preprocessing_and_dataset.ipynb`:** This notebook will demonstrate how your `TrashDetectionDataset` and `DataLoader` work, confirming that images and annotations are loaded and transformed correctly.

>
> To run a dev container select ctrl + shift + p -> *Dev Containers: Reopen in Container* and enjoy developing in a well-known VSCode.
