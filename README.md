# paper_smart_water_flow_event_classification
This repository provides a dataset and a trained MLP-based model for non-intrusive classification of water consumption events from flow sensor time series, including preprocessing components and evaluation tools to support reproducible research.

# Modelling individual residential water uses using machine learning algorithms 
This repository provides a dataset and two scripts. One  MLP-based and other KNN-basead for non-intrusive classification of water consumption events from flow sensor time series, including preprocessing script 

---

## Objective

The objective of this work is to classify water consumption patterns based on smart sensor data, using Machine Learning techniques and signal preprocessing methods.

---

## Dataset Description
The features.txt file contains four numerical features extracted from each water consumption event, together with the corresponding class label.

The input features are:

- Event duration.
- Median flow rate during the event.
- Flow rate decreasing time.
- Number of repetitions of the predominant event in the signal.
  
The class label identifies the corresponding residential water use category.

---

## Model

### Multilayer Perceptron (MLP) 

The network architecture and training parameters are defined in ANN.py.

### K-Nearest Neighbors (KNN)

The model configuration, hyperparameter selection, and evaluation procedures are defined in KNN.py.

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

- `features.txt` → Extracted numerical features and their corresponding class labels.
- `pre_processing.py` → pre_processing script  
- `ANN.py` → model ANN script  
- `KNN.py` → model KNN script


---
