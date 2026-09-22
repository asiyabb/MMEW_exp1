Here is a comprehensive `README.md` file tailored to your codebase, model architecture, and performance analysis:

```markdown
# Micro-to-Macro Guided Transformer for Expression Recognition

A PyTorch implementation of a **Micro-to-Macro Guided Dual-Path Transformer** designed to classify facial expressions from image sequences. The network combines spatial visual features extracted via a Vision Transformer (ViT-B/16) with motion dynamics extracted from dense Farnebäck Optical Flow, using a micro-expression-guided cross-attention mechanism for micro/macro-expression recognition.

---

## 📌 Features & Architecture Overview

* **Dual-Path Feature Extraction:**
  * **Spatial Branch:** Pre-trained Vision Transformer (`ViT-B/16`) extracting 768-dimensional spatial embeddings per frame.
  * **Motion Branch:** Dynamic CNN encoder converting 2-channel dense optical flow $(u, v)$ sequences into 256-dimensional motion vectors.
* **Learnable Gate Fusion:** Dynamically weights and concatenates spatial and motion representations (`768 + 256 = 1024` dimensions).
* **Spatio-Temporal Transformer:** A 2-layer Transformer Encoder that models long-range temporal dependencies across video frames.
* **Micro-to-Macro Cross-Attention Guidance:** Employs multi-head attention to let fine-grained micro-temporal features guide macro-expression classification.

---

## 📁 Repository Structure

```text
.
├── config.py             # Global hyperparameter configurations, paths, and class mappings
├── dataset.py            # PyTorch Dataset for loading video frames & computing Optical Flow
├── model.py              # Architecture (ViT + Motion CNN + Gate Fusion + Transformer)
├── evaluate.py           # Model evaluation, metric calculation & visualization pipeline
├── evaluate2.py          # Updated evaluation script targeting specific model checkpoints
├── plot_comparison.py    # Per-class F1-score & Recall breakdown plotting script
├── MMEW_Micro_Exp.xlsx   # Dataset annotations metadata
└── README.md             # Project documentation

```

---

## ⚙️ Configuration & Prerequisites

### Dependencies

Install the required packages using `pip`:

```bash
pip install torch torchvision numpy pandas opencv-python pillow matplotlib seaborn scikit-learn

```

### Key Configurations (`config.py`)

Edit `config.py` to match your local path setup:

* **`BASE_DIR`**: Root dataset directory.
* **`CLASSES`**: `['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']` (6 categories).
* **`SEQUENCE_LENGTH`**: `16` frames per sequence.
* **`VIT_DIM`**: `768` | **`MOTION_DIM`**: `256` | **`FUSED_DIM`**: `1024`.
* **`BATCH_SIZE`**: `8` | **`NUM_EPOCHS`**: `40` | **`LEARNING_RATE`**: `1e-4`.

---

## 🚀 Usage Instructions

### 1. Training the Model

To start training the guided Transformer model, ensure your dataset path is set in `config.py` and run:

```bash
python train.py

```

*(Checkpoints will be saved in the configured `checkpoints/` directory).*

### 2. Evaluation & Metric Visualization

To evaluate a trained checkpoint against validation data and generate performance plots:

```bash
python evaluate.py

```

This outputs classification report metrics (Precision, Recall, F1-Score, Support) and saves a dual-panel evaluation plot (`guided_experiment_results.png`) containing:

1. **Normalized Confusion Matrix**
2. **Per-Class Metrics Bar Chart**

---

## 📊 Performance & Class Imbalance Analysis

### Class-Wise Metrics Summary

Due to extreme class imbalance in micro/macro-expression datasets (e.g., MMEW), standard training yields strong performance on majority classes, while sparse classes (*anger*, *fear*, *sadness*) present a high sensitivity challenge:

| Expression Class | Precision | Recall | F1-Score | Status |
| --- | --- | --- | --- | --- |
| **Anger** | `0.00` | `0.00` | `0.00` | High sample sparsity |
| **Disgust** | `0.85` | `0.73` | `0.79` | High performance |
| **Fear** | `0.75` | `1.00` | `0.86` | High performance |
| **Happiness** | `1.00` | `1.00` | `1.00` | Perfect recognition |
| **Sadness** | `1.00` | `0.75` | `0.86` | High precision |
| **Surprise** | `0.88` | `0.94` | `0.91` | High performance |

### Key Observations

* **Overall Best Accuracy:** ~`87.0%` top validation accuracy reached within 40 epochs.
* **Minority Class Bottleneck:** Class imbalance causes zero-shot prediction issues for sparse classes like `anger` when random unstratified splits are used.
* **Remediation Strategies:**
1. Use **`WeightedRandomSampler`** in DataLoader for batch-level class balancing.
2. Implement **Focal Loss** ($\gamma = 1.5$) to down-weight easy majority samples (`happiness`, `surprise`).
3. Apply **Stratified K-Fold CV** to ensure minority class representation across train/val splits.



```

```
