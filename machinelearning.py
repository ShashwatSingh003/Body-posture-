import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn import metrics, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier

# === Ensure reproducibility ===
np.random.seed(42)
random.seed(42)

# === Global label encoder ===
label_encoder = LabelEncoder()

def read_data(file, fit_encoder=False):
    data = pd.read_csv(file)
    data = sklearn.utils.shuffle(data, random_state=42)

    X_data = data.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    y_data = data['ActivityName']

    if fit_encoder:
        y_encoded = label_encoder.fit_transform(y_data)
    else:
        y_encoded = label_encoder.transform(y_data)

    return np.array(X_data), np.array(y_encoded)

def train_model(train_x, train_y, model_name='NB', validation=None):
    model = None
    if model_name == 'SVM':
        model = svm.SVC(gamma='scale', probability=True, random_state=42)
    elif model_name == 'XGB':
        model = XGBClassifier(n_estimators=200, max_depth=5, n_jobs=2, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    elif model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=800, alpha=0.0001,
                              solver='sgd', verbose=False, tol=1e-9, random_state=42)
    elif model_name == 'ADA':
        model = AdaBoostClassifier(n_estimators=50, random_state=42)
    elif model_name == 'BAG':
        model = BaggingClassifier(n_jobs=2, n_estimators=50, random_state=42)
    elif model_name == 'RF':
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    elif model_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    else:
        model = GaussianNB()

    model.fit(train_x, train_y)

    if validation is not None:
        val_x, val_y = validation
        y_pred = model.predict(val_x)
        acc = metrics.accuracy_score(val_y, y_pred)
        print(f"\nModel: {model_name}")
        print(f"Validation Accuracy: {acc:.4f}")
        cm = metrics.confusion_matrix(val_y, y_pred)
        print("Confusion Matrix:\n", cm)
        recall = metrics.recall_score(val_y, y_pred, average='macro')
        precision = metrics.precision_score(val_y, y_pred, average='macro')
        f1 = metrics.f1_score(val_y, y_pred, average='macro')
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return model

# === Main execution ===
if __name__ == '__main__':
    # First fit label encoder on training data only
    train_X, train_y = read_data('data/train.csv', fit_encoder=True)
    test_X, test_y = read_data('data/test.csv', fit_encoder=False)

    print("Train Shape : ", train_X.shape, train_y.shape)
    print("Test Shape  : ", test_X.shape, test_y.shape)

    models = ['RF', 'BAG', 'ADA', 'NB', 'SVM', 'XGB', 'KNN', 'MLP']
    trained_models = {}

    for model_name in models:
        trained_models[model_name] = train_model(train_X, train_y, model_name=model_name, validation=(test_X, test_y))

    # Optional: Save best model
    with open("best_model.pkl", "wb") as f:
        pickle.dump(trained_models['RF'], f)  # Change 'RF' if needed
