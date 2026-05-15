# drone-object-detection-
## Antlings Internship – AI/ML Technical Assessment

A computer vision pipeline built on the VisDrone dataset to detect humans and
cars from aerial imagery, count humans, and visualize results
using a YOLOv8s model.

## Dataset

**VisDrone2019-DET** — 
[Kaggle Link](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset?resource=download )

Drone/aerial imagery with 10 object classes across train, val, and test-dev splits.

| Split | Usage |
|-------|-------|
| VisDrone2019-DET-train | Model training |
| VisDrone2019-DET-val | Validation during training |
| VisDrone2019-DET-test-dev | Inference & counting |

**Classes detected and counted:**
- 🔴 **Human** — `pedestrian` (0), `people` (1)
- 🔵 **Vehicle** — `car` (3)

---

## Model: YOLOv8s

- **Input size:** 1280×1280
- **Epochs:** 20 (Reduced from 50 to 20 due considering GPU limits and deadline)
- **Batch size:** 4 (Due to memory limit)
- **Framework:** Ultralytics YOLOv8

---
Notebook made in kaggle notebook after facing GPU limits in Colab

**Key libraries include:**

- ultralytics
- opencv
- matplotlib
- numpy


---

## Pipeline Overview

### Task 01 — Dataset Understanding & Preprocessing
- Explored dataset structure across train/val/test splits
- Visualized class distribution, sample annotations, and bounding box sizes
- Noted challenges: small object sizes and dense crowds

### Task 02 — Model Training
- Fine-tuned YOLOv8s on VisDrone training set
- Used image size 1280 to preserve detail for small objects
- Training outputs saved to Google Drive (`outputs/yolov8s_visdrone/`)

### Task 03 — Detection & Human Counting
- Runs inference on test-dev images at `conf=0.25`
- Draws color-coded bounding boxes 
- Overlays human and vehicle count on each image
- Prints total and average counts across the full test set

### Task 04 — Tracking
- ByteTrack implementation included in notebook
- Builds a video from test frames, runs `model.track()` with persistent IDs
- Plots human count per frame over time

### Task 05 — Evaluation
- Ran `model.val()` on validation set
- Computed mAP@0.5, mAP@0.5:0.95, Precision, Recall
- Plotted per-class AP@0.5 bar chart

---

## Outputs

All outputs are saved to `outputs/` and in Google Drive:

| File | Description |
|------|-------------|
| `class_distribution.png` | Class frequency across dataset |
| `sample_annotations.png` | Ground truth bounding box examples |
| `bbox_distribution.png` | Bounding box size distribution |
| `sample_predictions.png` | Val set prediction samples |
| `detection_results.png` | Test set detection + count overlay |
| `per_class_ap.png` | Per-class AP@0.5 bar chart |
| `tracking_timeline.png` | Human count per frame (bonus) |

---

## Demo Video

[▶ Watch Demo on Google Drive](#link) 

---

## Challenges & Limitations

- **Small objects** — many humans appear as very few pixels at drone altitude;
  1280px input size helps but small detections remain difficult
- **Dense crowds** — heavy overlap between pedestrians causes missed detections
- **Class imbalance** — pedestrian and people classes dominate; rarer classes
  like tricycle and awning-tricycle have lower AP
- **GPU constraints** — batch size and epoch limited due to Colab GPU limits. Shifted to Kaggle notebook.

---

## Results

| Metric | Value |
|--------|-------|
| mAP@0.5 | *( output)* |
| mAP@0.5:0.95 | *(  output)* |
| Precision | *(  output)* |
| Recall | *(   output)* |

---

## Acknowledgements
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- AI tools used for implementation assistance: Gemini in colab, Claude (Anthropic) 
