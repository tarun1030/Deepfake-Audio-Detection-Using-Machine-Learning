# 🕵️‍♂️ Audio Deepfake Detection ![License](https://img.shields.io/badge/license-MIT-green.svg) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

A project focused on detecting deepfake audio using both traditional machine learning classifiers and deep learning models like CNNs.

---

## 📚 Table of Contents
- [🚀 Project Overview](#-project-overview)
- [🧠 Models Used](#-models-used)

---

## 🚀 Project Overview
This repository contains an **Audio Deepfake Detection** system trained to distinguish between real and fake audio clips. The system uses both classical ML classifiers and deep learning CNNs trained on audio features like MFCC or spectrograms.

---

## 🧠 Models Used

### 🔢 Classical ML Models (from `code1/` and `code2/`)
- ✅ MLPClassifier *(best performing)*
- Decision Tree
- Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost
- QDA

### 🧠 Deep Learning Model (from `code3/` and `code4/`)
- CNN model with:
  - Early stopping
  - Learning rate scheduling
  - Regularization & Dropout

---

