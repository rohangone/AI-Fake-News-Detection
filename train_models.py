# ======================================
# AI Fake News Detection - Model Training
# ======================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake News Articles :", fake.shape)
print("True News Articles :", true.shape)

# ==========================
# Assign Labels
# ==========================

fake["label"] = 0
true["label"] = 1

# ==========================
# Merge Dataset
# ==========================

data = pd.concat([fake, true], axis=0)

print("\nTotal Dataset Shape :", data.shape)
# ==========================
# Remove Unnecessary Columns
# ==========================

data = data.drop(["title", "subject", "date"], axis=1)

print("\nRemaining Columns:")
print(data.columns)

# ==========================
# Shuffle Dataset
# ==========================

data = data.sample(frac=1, random_state=42)

print("\nDataset Shuffled Successfully!")

# ==========================
# Input and Output
# ==========================

X = data["text"]
y = data["label"]

print("\nInput Shape :", X.shape)
print("Output Shape :", y.shape)
# ==========================
# Shuffle Dataset
# ==========================

data = data.sample(frac=1, random_state=42)

print("\nDataset Shuffled Successfully!")

# ==========================
# Input and Output
# ==========================

X = data["text"]
y = data["label"]

print("\nInput Shape :", X.shape)
print("Output Shape :", y.shape)
# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Data :", X_train.shape)
print("Testing Data :", X_test.shape)
# ==========================
# TF-IDF Vectorization
# ==========================

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nTF-IDF Vectorization Completed!")
print("Training Matrix Shape :", X_train.shape)
print("Testing Matrix Shape :", X_test.shape)
# ==========================
# Logistic Regression
# ==========================

print("\n==============================")
print("Training Logistic Regression...")
print("==============================")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Logistic Regression Completed!")
# ==========================
# KNN
# ==========================

print("\n==============================")
print("Training KNN...")
print("==============================")

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("KNN Completed!")

# ==========================
# Random Forest
# ==========================

print("\n==============================")
print("Training Random Forest...")
print("==============================")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Completed!")

# ==========================
# Neural Network
# ==========================

print("\n==============================")
print("Training Neural Network...")
print("==============================")

mlp = MLPClassifier(
    hidden_layer_sizes=(20,),
    max_iter=20,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train, y_train)

mlp_pred = mlp.predict(X_test)

print("Neural Network Completed!")
# ==========================
# Model Evaluation Function
# ==========================

def evaluate_model(name, y_true, y_pred):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return [name, accuracy, precision, recall, f1]


# Evaluate all models
results = []

results.append(evaluate_model("Logistic Regression", y_test, lr_pred))
results.append(evaluate_model("KNN", y_test, knn_pred))
results.append(evaluate_model("Random Forest", y_test, rf_pred))
results.append(evaluate_model("Neural Network", y_test, mlp_pred))
# ==========================
# Comparison Table
# ==========================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(results_df)

best_model = results_df.loc[results_df["Accuracy"].idxmax()]

print("\nBest Model:")
print(best_model)
# ==========================
# Save Best Model
# ==========================

pickle.dump(lr, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")