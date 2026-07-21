# paper_smart_water_flow_event_classification
This repository provides a dataset and a trained MLP-based model for non-intrusive classification of water consumption events from flow sensor time series, including preprocessing components and evaluation tools to support reproducible research.

# Modelling individual residential water uses using machine learning algorithms 
This repository provides a dataset and a trained MLP-based model for non-intrusive classification of water consumption events from flow sensor time series, including preprocessing components and evaluation tools.The files correspond to the final model trained after expanding the training dataset.

---

## Objective

The objective of this work is to classify water consumption patterns based on smart sensor data, using Machine Learning techniques and signal preprocessing methods.

---

## Model

A Multilayer Perceptron (MLP) neural network was used with the following architecture:

- Input layer with 4 features: (i) the duration of the event, (ii) the median flow rate of the event, (iii) the flow rate decreasing time, (iv) the number of repetitions of the predominant event in the signal 
- Dense layer with 32 neurons (ReLU)  
- Batch Normalization  
- Dropout (0.2)  
- Dense layer with 16 neurons (ReLU)  
- Output layer with Softmax activation and 8 classes: Flush toilet 1, Flush toilet 2, Flush toilet 3, Shower, Filter, Washing machine, Tap 1, Kitchen tap  


A K-Nearest Neighbors (KNN) classifier was configured as follows:

- Input features: (i) event duration, (ii) median flow rate, (iii) flow rate decreasing time, and (iv) number of repetitions of the predominant event in the signal.
- Hyperparameter optimization: the number of neighbors (k) was selected using Grid Search with 5-fold cross-validation, evaluating values from 1 to 10.
- Classification algorithm: K-Nearest Neighbors (KNN).
- Distance metric: Minkowski distance (p=2, equivalent to Euclidean distance).
- Neighbor weighting: Uniform.
- Model evaluation: 10-fold cross-validation on the training set.
---

## Preprocessing

- Data normalization using `StandardScaler`  
- Class encoding using `LabelEncoder`  
- Train/test split: 70% / 30% 

---

## Model Evaluation

The model is evaluated using:

- Accuracy  
- Loss  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix  

---

## Project Files

- `data.txt` → dataset  
- `pre_processing.py` → pre_processing script  
- `ANN.py` → model ANN script  
- `KNN.py` → model KNN script  

---
