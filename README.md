# 🫁 COVID-19 Chest X-Ray Classification

A deep learning project for classifying chest X-ray images into three categories: **Covid-19**, **Normal**, and **Viral Pneumonia** — using both a custom CNN and VGG16 transfer learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [How to Run](#how-to-run)
- [Technologies](#technologies)

---

## 🔍 Overview

This project builds and compares two approaches to medical image classification:

1. **Custom CNN** — A convolutional neural network trained from scratch
2. **VGG16 Transfer Learning** — A pre-trained VGG16 model fine-tuned for our 3-class problem

The goal is to assist in the detection of COVID-19 from chest X-ray images.

---

## 📦 Dataset

**Source:** [COVID-19 Image Dataset – Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

| Class | Label |
|---|---|
| Normal | 0 |
| Covid-19 | 1 |
| Viral Pneumonia | 2 |

- Images resized to **224×224 pixels**
- Normalized to **[0, 1]** range
- Split: **80% train / 20% test**

---

## 📁 Project Structure

```
├── covid_classification.ipynb   # Main notebook (Google Colab)
├── README.md                    # Project documentation
```

---

## 🧠 Models

### Model 1 — Custom CNN

```
Conv2D(32) → MaxPooling
Conv2D(64) → MaxPooling
Flatten → Dense(128) → Dropout(0.2)
Dense(256) → Dropout(0.2)
Dense(3)  ← output layer
```

- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy (from_logits=True)
- Epochs: 5

---

### Model 2 — VGG16 Transfer Learning

- Base: VGG16 pre-trained on ImageNet (all layers frozen)
- Custom head: `Dense(3, activation='softmax')`
- Optimizer: Adam
- Loss: sparse_categorical_crossentropy
- Epochs: 10

---

## 📊 Results

| Model | Test Accuracy |
|---|---|
| Custom CNN | ~85.94% |
| VGG16 Transfer Learning | TBD after training |

---

## ▶️ How to Run

1. Open the notebook in **Google Colab**
2. Install dependencies:
   ```python
   pip install kagglehub
   ```
3. The dataset is downloaded automatically via `kagglehub`
4. Run all cells in order

---

## 🛠️ Technologies

| Tool | Purpose |
|---|---|
| Python 3.12 | Programming language |
| TensorFlow / Keras | Deep learning framework |
| NumPy | Array manipulation |
| Pillow (PIL) | Image loading & resizing |
| scikit-learn | Train/test split |
| Matplotlib | Visualization |
| KaggleHub | Dataset download |

---

## 👤 Author

> Project developed as part of a deep learning study on medical image classification.
