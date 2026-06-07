import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=== DecodeLabs Project 2: Data Classification Using AI ===")
    print("Loading the Iris dataset...")
    
    # 1. Load and understand the dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    print("\nDataset Overview:")
    print(f"Total samples: {X.shape[0]}")
    print(f"Features: {', '.join(iris.feature_names)}")
    print(f"Target classes: {', '.join(iris.target_names)}")
    
    # 2. Split data into training and testing sets
    # Using 80% of data for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nData successfully split:")
    print(f"- Training samples: {X_train.shape[0]}")
    print(f"- Testing samples: {X_test.shape[0]}")
    
    # Feature Scaling (Standardization)
    # This helps many algorithms perform better by scaling features to have a mean of 0 and variance of 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Apply a classification algorithm
    print("\nTraining the Random Forest Classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_scaled, y_train)
    
    # 4. Make predictions and evaluate
    print("Making predictions on the test set...")
    y_pred = classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Results ===")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    # Pass target_names to generate a readable report instead of just class numbers
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    print(report)

if __name__ == '__main__':
    main()
