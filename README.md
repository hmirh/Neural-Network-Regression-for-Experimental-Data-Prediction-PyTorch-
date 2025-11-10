# Neural Network Regression for Experimental Data Prediction (PyTorch)

This repository implements a fully connected deep neural network (DNN) using **PyTorch** to perform regression on experimental or simulated data.  
The workflow includes data preprocessing, normalization, train–validation–test splitting, model design, training visualization, and performance evaluation using MSE and R² metrics.

---

## 🧠 Overview

The model predicts a continuous target variable based on seven input features from `mldata_log_est.csv`.

The network is trained to minimize **Mean Squared Error (MSE)** using **stochastic gradient descent (SGD)**.  
Performance is tracked on both training and validation datasets.

---

## 🧩 Key Steps

1. **Data Preparation**
   - Reads the CSV file and separates features (`X`) and target (`y`).
   - Normalizes data using `StandardScaler`.
   - Splits data into **train**, **validation**, and **test** sets.
   - Wraps data in a custom PyTorch `Dataset` and `DataLoader`.

2. **Model Architecture**
   ```python
   nn.Sequential(
       nn.Linear(7, 6),
       nn.Linear(6, 12),
       nn.Linear(12, 36),
       nn.Linear(36, 6),
       nn.Linear(6, 1)
   )
