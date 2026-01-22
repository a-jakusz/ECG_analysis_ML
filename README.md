# ECG_analysis_ML

## Project Overview
This project focuses on automatic classification of ECG heartbeats using machine learning and deep learning methods.  
The goal is to compare classical baseline models with a deep learning approach and show how explicitly modeling time dependencies in ECG signals improves classification quality.

The project is based on the **MIT-BIH ECG Heartbeat Dataset** (Kaggle version), where each sample represents a single heartbeat described by 187 signal values.

—

## Part 1 (Artsiom Tsiushnikau) – Data analysis + Baseline models

## Dataset
- Number of samples: ~109,000 heartbeats
- Sampling frequency: 125 Hz
- Input: 1D ECG signal (187 points per heartbeat)
- Output: heartbeat class (5 classes)

Classes:
- **0 (N)** – Normal beat  
- **1 (S)** – Supraventricular ectopic beat  
- **2 (V)** – Ventricular ectopic beat  
- **3 (F)** – Fusion beat  
- **4 (Q)** – Unknown / paced beat  

The dataset is highly imbalanced, normal beats dominate (about 75000 normal heartbeats against 24000 of all other classes). This makes evaluation and class handling an important part of the project.
The dataset is also already normalised. It doesn’t represent the absolute values of ECG in Htz, but the rescaled values from 0 to 1. We normalise it once again in order to obtain the distribution closer to Gauss distribution. This improves models performance.

---

## Baseline Machine Learning Models

### Motivation
Baseline models are not used to achieve the best possible performance.  
They serve as a **reference point** to show how much performance gain is obtained when switching to deep learning models. They also don’t treat the dataset as a time sequence.

### Methods
The following classical models were used:
- Logistic Regression
- Random Forest
- SGD-based linear classifier

Each heartbeat was treated as a fixed-length feature vector (187 values), without explicit modeling of temporal dependencies.

### Evaluation
- Main metric: **macro-F1 score** (due to class imbalance)
- Additional metrics: accuracy and confusion matrix

### Observations
- Baseline models achieved low accuracy (except for Random Forest).
- Overfitting on Random Forest (100% accuracy).
- Performance on rare classes (S and F) was limited.
- These results confirmed the need for models that can better capture signal dynamics.

---

## Part 2 (Diemid Rybchenko) – Deep Learning Model (1D Convolutional Neural Network)

### Motivation
ECG data is a time series, where the order and local patterns of signal values matter.  
Deep learning models, especially 1D convolutional networks, are well suited for this type of data.

### Preprocessing
- **Per-sample z-score normalization** was applied to each heartbeat  
  (normalization done independently per sample, without data leakage).
- Class imbalance was handled using **weighted cross-entropy loss** with capped class weights.

### Model Architecture
- 1D Convolutional Neural Network (1D-CNN)
- 3 convolutional blocks with:
  - Conv1D
  - Batch Normalization
  - ReLU activation
  - Max Pooling
- Fully connected layers with dropout
- Output layer with 5 logits (one per class)

### Training Details
- Loss function: Cross-Entropy Loss with class weights
- Optimizer: Adam
- Learning rate scheduler: ReduceLROnPlateau
- Early stopping based on validation macro-F1
- Training performed on CPU (GPU used if available)

### Results
**Test set performance:**
- Accuracy: **~0.98**
- Macro-F1 score: **~0.91**

The deep learning model significantly outperformed baseline models, especially on minority classes.

### Error Analysis
- Confusion matrix analysis showed:
  - Very good recognition of Normal (N), Ventricular (V), and Q beats
  - Strong improvement for rare classes S and F compared to baseline models
- The use of capped class weights helped avoid over-predicting rare classes.

---

## Key Takeaways
- Baseline models are useful as a reference but are limited for time-series ECG data.
- Explicit modeling of temporal structure with 1D-CNN leads to a large performance improvement.
- Macro-F1 is a more appropriate metric than accuracy for imbalanced medical datasets.
- Proper normalization and class imbalance handling are crucial for good results.

---

## Conclusion
This project demonstrates that deep learning models designed for time-series data can significantly improve ECG heartbeat classification compared to classical machine learning approaches.  
The comparison with baseline models clearly shows the benefit of modeling temporal dependencies in biomedical signals.

—

## References

ECG and dataset

Goldberger et al., 2000
PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals - original MIT-BIH Arrhythmia Database
Moody, Mark, 2001
The impact of the MIT-BIH Arrhythmia Database
IEEE Engineering in Medicine and Biology Magazine

     2) Deep Learning

Kiranyaz et al., 2016
Real-Time Patient-Specific ECG Classification by 1D CNN
Rajpurkar et al., 2017
Cardiologist-Level Arrhythmia Detection with CNN

    3) ML and metrics

Goodfellow, Bengio, Courville, 2016
Deep Learning, MIT Press
Powers, 2011
Evaluation: From Precision, Recall and F-measure to ROC. Journal of Machine Learning Technologies

    4) Practices and instruments

Paszke et al., 2019
PyTorch: An Imperative Style, High-Performance Deep Learning Library NeuIPS
Pedregosa et al., 2011
Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research

## Authors
- Diemid Rybchenko
- Artsiom Tsiushnikau
