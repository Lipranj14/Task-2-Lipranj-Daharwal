# DecodeLabs Project 2: Data Classification Using AI

This repository contains a Python-based machine learning solution that classifies the classic Iris dataset using a Random Forest Classifier.

## Project Overview

The project loads the Iris dataset (consisting of 150 samples with 4 features each) and builds a classification model to categorize flowers into three target species:
- Setosa
- Versicolor
- Virginica

### Key Steps Performed
1. **Dataset Loading & Analysis**: Reads the Iris dataset features and target classes.
2. **Data Splitting**: Splits the dataset into 80% training and 20% testing sets.
3. **Feature Scaling**: Uses `StandardScaler` to standardize features (mean = 0, variance = 1) for optimal classifier performance.
4. **Model Training**: Trains a `RandomForestClassifier` with 100 estimators.
5. **Evaluation**: Evaluates the model using Accuracy Score, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.

## Getting Started

### Prerequisites

You need Python 3 installed on your system along with the following libraries:
- `pandas`
- `scikit-learn`

You can install the dependencies via pip:
```bash
pip install pandas scikit-learn
```

### File Structure
- `data_classifier.py`: The main Python script that loads, pre-processes, trains, and evaluates the machine learning model.
- `README.md`: Project documentation.

## Running the Project

To execute the data classifier, run:
```bash
python data_classifier.py
```

## Example Output

When executed, the model outputs an overview of the dataset and the training results:

```text
=== DecodeLabs Project 2: Data Classification Using AI ===
Loading the Iris dataset...

Dataset Overview:
Total samples: 150
Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
Target classes: setosa, versicolor, virginica

Data successfully split:
- Training samples: 120
- Testing samples: 30

Training the Random Forest Classifier...
Making predictions on the test set...

=== Results ===
Model Accuracy: 100.00%

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```
