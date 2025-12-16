# Machine Learning Projects

## Overview
This repository contains two comprehensive machine learning projects demonstrating end-to-end ML pipeline implementation, including data preprocessing, feature engineering, model training, and evaluation.

---

## Project 1: Healthcare Insurance Expenses Prediction

### Description
A regression-based machine learning project that analyzes and predicts individual medical insurance charges using demographic and health-related features. The project explores the relationships between personal characteristics and healthcare costs through exploratory data analysis and advanced regression techniques.

### Objectives
- Analyze factors influencing medical insurance charges
- Build predictive models to estimate healthcare expenses
- Identify key features affecting insurance premium calculations
- Compare linear and non-linear modeling approaches

### Dataset
- **Source**: [Kaggle Healthcare Insurance Expenses Dataset](https://www.kaggle.com/datasets/arunjangir245/healthcare-insurance-expenses)
- **Size**: 1,338 records with 7 features
- **Features**: Age, Sex, BMI, Number of Children, Smoker Status, Geographic Region
- **Target Variable**: Insurance Charges
- **Data Quality**: No missing values, minimal preprocessing required

### Methodology

#### Exploratory Data Analysis
- Statistical profiling and distribution analysis
- Identification of outliers and data anomalies
- Visualization of feature correlations with charges
- Detection of skewed distributions (log transformation applied)

#### Feature Engineering
- **BMI Categories**: Underweight, Normal, Overweight, Obese
- **Age Groups**: Young (≤30), Middle-aged (30-50), Senior (>50)
- **Interaction Features**: age×bmi, bmi×children captures combined effects
- **Feature Selection**: SelectKBest with F-regression for optimal feature set
- **Scaling**: StandardScaler for normalization

#### Models Implemented
1. **Linear Regression**: Baseline model with feature engineering
2. **Polynomial Features**: Degree 2 and 3 polynomials for non-linearity
3. **Regularization**: L2 regularization (Ridge) for overfitting prevention
4. **Hyperparameter Tuning**: Grid search for optimal regularization strength

### Results
- **Best Model**: Polynomial Features (Degree 2)
- **Test R² Score**: 0.9025 (explains 90.25% of variance)
- **MAE**: $2,585.76
- **Model Accuracy**: 86.2%
- **Key Insights**: Smoking status and age are dominant predictors; non-linear relationships significantly improve performance

### Technologies
- Python 3.x
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- Google Colab / Jupyter Notebook

---

## Project 2: Eye Disease Classification

### Description
A computer vision and machine learning project for automated classification of eye diseases from fundus images. The system identifies four disease categories (Cataract, Diabetic Retinopathy, Glaucoma, and Normal) using classical computer vision features combined with traditional ML classifiers.

### Objectives
- Develop automated eye disease screening system
- Extract discriminative features from fundus images
- Build robust multi-class classification model
- Compare multiple feature extraction and modeling approaches
- Achieve production-ready classification accuracy

### Dataset
- **Source**: [Eye Disease Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Total Images**: 4,217 fundus photographs
- **Image Size**: 512×512 pixels (RGB)
- **Class Distribution**:
  - Cataract: 1,038 images (24.6%)
  - Diabetic Retinopathy: 1,098 images (26.0%)
  - Glaucoma: 1,007 images (23.9%)
  - Normal: 1,074 images (25.5%)
- **Data Balance**: Well-balanced across all classes

### Methodology

#### Feature Extraction Pipeline
The project implements multi-modal feature extraction combining several classical computer vision techniques:

1. **Color Histogram Features**
   - RGB and HSV color space histograms
   - Captures color distribution patterns

2. **GLCM (Gray-Level Co-occurrence Matrix)**
   - Texture analysis from different angles (0°, 45°, 90°, 135°)
   - Extracts statistical properties (contrast, dissimilarity, correlation)

3. **LBP (Local Binary Pattern)**
   - Local texture descriptors for micro-patterns
   - Robust to illumination variations

4. **HOG (Histogram of Oriented Gradients)**
   - Edge and shape information
   - Detects structural patterns in eye images

5. **Structural Features**
   - Medical domain-specific features
   - Vessel patterns, optic disc characteristics

#### Preprocessing
- Image resizing to consistent dimensions (224×224)
- Color normalization and contrast adjustment
- Feature standardization using StandardScaler

#### Dimensionality Reduction
- **PCA (Principal Component Analysis)**
- Variance retention: 98%
- Original features: 2,688 → Reduced features: 844 (69% reduction)
- Maintains medical image detail while improving computational efficiency

#### Models Trained
1. **Logistic Regression** (Best performer)
   - Hyperparameters: C=9e-05, solver='saga', max_iter=2000
   - Multi-class strategy: multinomial
   - Feature scaling: StandardScaler & PCA

2. **Support Vector Machine (SVM)**
   - Kernel optimization for non-linear separation

3. **K-Means Clustering**
   - Unsupervised analysis for disease stratification

### Results
- **Best Model**: Logistic Regression with PCA
- **Overall Test Accuracy**: 83.65%
- **Per-Class Performance**:
  - Cataract: Precision 0.84, Recall 0.76, F1 0.80
  - Diabetic Retinopathy: Precision 0.99, Recall 1.00, F1 1.00
  - Glaucoma: Precision 0.76, Recall 0.61, F1 0.67
  - Normal: Precision 0.75, Recall 0.95, F1 0.84
- **Macro Average**: Precision 0.84, Recall 0.83, F1 0.83
- **Key Insights**: Perfect detection of diabetic retinopathy; glaucoma requires additional discriminative features

### Technologies
- Python 3.x
- OpenCV (cv2) for image processing
- scikit-image (skimage) for feature extraction
- scikit-learn for ML modeling
- numpy, pandas for data manipulation
- matplotlib, seaborn for visualization
- Google Colab environment

### Team Members (Contributors)
- Load & Feature Assembly: Member 1
- Preprocessing: Member 2
- Color & GLCM Features: Member 3
- LBP & HOG Features: Member 4
- Medical Features: Member 5
- Scaling & LR Model: Member 6
- K-Means & Visualization: Member 7

---

## Project Comparison

| Aspect | Healthcare Expenses | Eye Disease Classification |
|--------|---------------------|---------------------------|
| **Type** | Regression | Multi-class Classification |
| **Data Type** | Tabular | Image |
| **Sample Size** | 1,338 | 4,217 |
| **Features** | 7 (engineered to 8-10) | 2,688 (PCA reduced to 844) |
| **Test Accuracy** | 86.2% (R²: 0.9025) | 83.65% |
| **Model Used** | Polynomial Regression | Logistic Regression + PCA |
| **Domain** | Healthcare Economics | Medical Imaging |
| **Key Challenge** | Non-linear relationships | Class Overlapping |

---

