
# Data Mining / Data Science Project

A structured collection of **machine learning, data mining, and deep learning exercises** implemented in Python using Jupyter Notebooks, practised as a 3 months spanned project. This repository demonstrates **end-to-end ML pipelines**, from data preprocessing and classical models to **CNNs and explainable AI**.

# Introduction & Goals

- Throughout the projects, those standard workflow are discussed and practised that including:
- Problem understanding.
- Data exploration & preprocessing.
- Feature engineering. 
- Model sectiontion & training.
- Validation & parameter tuning.
- Testing & communication results.
- Deployment & monitoring. 


# Contents

Each notebook focuses on **both implementation and conceptual understanding**.

## Repository Structure
```text
├── Exercise_2_Pandas_Introduction.ipynb
├── Exercise_3_4_DataMiningPipeline_LR.ipynb
├── Exercise_5_PCA_tSNE.ipynb
├── Exercise_6_ClassificationPipeline_RDF.ipynb
├── Exercise_7_clustering.ipynb
├── Exercise_8_HeatingLoadRegression.ipynb
├── Exercise_9_10_NeuralNetWorks_fromLR_to_NN.ipynb
├── Exercise_10_plus_ImageRecognition_CNN.ipynb
└── Exercise_11_Explainability_Analysis.ipynb
```

# Exercise Highlights
## 📗 Exercise 2 – Pandas Introduction
**File:** [`Exercise_2_Pandas_Introduction.ipynb`](Exercise_2_Pandas_Introduction.ipynb)


### Overview
Introduction to **Pandas** for tabular data manipulation and exploration.

### What I have learned
- How to create and manipulate `Series` and `DataFrame`
- The difference between `loc` and `iloc` indexing
- How to filter data using logical conditions
- How to handle missing values and basic data cleaning
- Why Pandas is the foundation of all data science and ML workflows

## 📗 Exercise 3 & 4 – Data Mining Pipeline & Logistic Regression

**File:** [`Exercise_3_4_DataMiningPipeline_LR.ipynb`](Exercise_3_4_DataMiningPipeline_LR.ipynb)


### Overview
Implementation of a complete machine learning pipeline using **Logistic Regression**.

### What I have learned
- How to structure an end-to-end machine learning pipeline
- The importance of proper train/test splitting
- How Logistic Regression performs binary classification
- The difference between class prediction and probability prediction
- Why metrics like **ROC-AUC** are preferred for imbalanced datasets
- How data preparation impacts model performance more than the model itself

## 📗 Exercise 5 – PCA & t-SNE

**File:** [`Exercise_5_PCA_tSNE.ipynb`](Exercise_5_PCA_tSNE.ipynb)

### Overview
Dimensionality reduction techniques applied to high-dimensional biological data.

### What I have learned
- Why standardization is mandatory before applying PCA
- How PCA reduces dimensionality while preserving variance
- How to interpret PCA scatter plots
- Why t-SNE is useful for visualization but not for modeling
- The conceptual difference between linear (PCA) and non-linear (t-SNE) methods

## 📗 Exercise 6 – Classification Pipeline with Random Forest

**File:** [`Exercise_6_ClassificationPipeline_RDF.ipynb`](Exercise_6_ClassificationPipeline_RDF.ipynb)

### Overview
Supervised classification using **Random Forest** and ensemble learning.

### What I have learned
- How ensemble methods improve model robustness
- Why Random Forests are less prone to overfitting
- That tree-based models do not require feature scaling
- How feature importance is computed in tree-based models
- The trade-off between model performance and interpretability

## 📗 Exercise 7 – Clustering

**File:** [`Exercise_7_clustering.ipynb`](Exercise_7_clustering.ipynb)

### Overview
Unsupervised learning using **K-Means clustering**.

### What I have learned
- The difference between supervised and unsupervised learning
- Why scaling is critical for distance-based clustering algorithms
- How the Elbow Method helps estimate the optimal number of clusters
- What inertia measures and its limitations
- That clustering results are heuristic and context-dependent

## 📗 Exercise 8 – Heating Load Regression

**File:** [`Exercise_8_HeatingLoadRegression.ipynb`](Exercise_8_HeatingLoadRegression.ipynb)

### Overview
Regression modeling for predicting heating load based on building features.

### What I have learned
- How linear regression models continuous target variables
- How to analyze correlations between features and target
- What R² score represents in regression tasks
- Why multicollinearity affects interpretability
- How MSE penalizes large prediction errors

## 📗 Exercise 9 & 10 – From Logistic Regression to Neural Networks

**File:** [`Exercise_9_10_NeuralNetWorks_fromLR_to_NN.ipynb`](Exercise_9_10_NeuralNetWorks_fromLR_to_NN.ipynb)

### Overview
Transition from traditional machine learning to **neural networks**.

### What I have learned
- Logistic Regression can be seen as a single-layer neural network
- Why one-hot encoding is required for multi-class classification
- How fully connected neural networks process image data
- The role of activation functions and loss functions
- The conceptual difference between forward propagation and backpropagation

## 📗 Exercise 10+ – Image Recognition with CNN

**File:** [`Exercise_10_plus_ImageRecognition_CNN.ipynb`](Exercise_10_plus_ImageRecognition_CNN.ipynb)

### Overview
Image classification using **Convolutional Neural Networks (CNNs)**.

### What I have learned
- How convolutional layers extract spatial features
- Why CNNs outperform fully connected networks on image data
- How pooling layers reduce dimensionality and overfitting
- The concept of parameter sharing in CNNs
- How CNNs learn hierarchical features from edges to objects

## 📗 Exercise 11 – Explainability Analysis

**File:** [`Exercise_11_Explainability_Analysis.ipynb`](Exercise_11_Explainability_Analysis.ipynb)

### Overview
Explainable AI techniques applied to classification models.

### What I have learned
- Why model accuracy alone is not sufficient
- The importance of explainability in real-world ML systems
- The difference between global and local explanations
- How different feature importance methods yield different insights
- Why trust and transparency are essential for deploying ML models
