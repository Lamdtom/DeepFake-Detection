# Deepfake Detection using EfficientNetB4

This project implements a deepfake detection pipeline using the **EfficientNetB4** convolutional neural network architecture on the [Comprehensive Deepfake Detection Dataset (CDDF)](https://www.kaggle.com/datasets/cabani/efficientnetb4-deepfake-detection). The dataset consists of **110,694 frames extracted from 480 videos**, containing both authentic and deepfake samples.

## 📌 Project Overview

With the rise of synthetic media and AI-generated content, detecting deepfakes is critical for maintaining trust and integrity in digital media. This project:

* Preprocesses the CDDF dataset (resizing, normalization, labeling).
* Uses EfficientNetB4, a state-of-the-art CNN model, for frame-based binary classification (real vs fake).
* Evaluates performance using metrics such as accuracy, precision, recall, and F1-score.

---

## 🚀 Model Summary

* **Architecture**: EfficientNetB4 (pretrained on ImageNet, fine-tuned for binary classification)
* **Input size**: 380 × 380 RGB frames
* **Optimizer**: Adam
* **Loss function**: Binary Cross-Entropy
* **Evaluation**: Accuracy, Precision, Recall, F1-Score

---

## 📁 Dataset

* Name: **Comprehensive Deepfake Detection Dataset**
* Size: \~110k image frames from 480 videos
* Classes: `Real`, `Fake`
* Source: Public dataset from Kaggle

---

## ⚙️ Setup

### Installation

```bash
git clone https://github.com/Lamdtom/DeepFake-Detection
cd DeepFake-Detection
pip install -r requirements.txt
```

---

## 🧪 Training

```python
python train.py --train_dataset ./dataset/train --val_dataset ./dataset/val
```

Model checkpoints, logs, and metrics are saved to the `outputs/` directory.

---

## 📊 Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 91.3% |
| Precision | 90.5% |
| Recall    | 92.1% |
| F1-Score  | 91.3% |

---

## 🔮 Future Work

To improve detection accuracy and robustness, the following directions are planned:

### 📈 Model Enhancements

* 🔁 **Add more CNN backbones**:

  * ResNet50 / ResNet101
  * InceptionV3 / Xception
  * DenseNet121 / DenseNet201
  * EfficientNetV2 variants
* 🔍 **Train on video-level sequences** using 3D CNNs or RNNs for spatiotemporal learning.
* ⚡ **Ensemble models** for better generalization and confidence calibration.
* 🧠 **Self-supervised pretraining** on unlabeled video data to learn deepfake-specific representations.

### 🛠️ Data Improvements

* ⚖️ **Balance classes** and augment deepfake frames with transformations to reduce overfitting.
* 👁️ **Integrate face landmarks** or eye/mouth movement cues as additional inputs.
* 🎥 **Fuse temporal context** by incorporating adjacent frames or optical flow.

### 🧪 Evaluation Improvements

* ✅ Add support for **cross-dataset testing** (e.g., FaceForensics++, DFDC, Celeb-DF).
* 🧯 Improve **explainability** using Grad-CAM to visualize CNN focus regions.

---
