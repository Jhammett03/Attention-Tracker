import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

#linear regression on averaged data for each video using 5-fold cross validation
def main():
    df = pd.read_csv("dataset.csv")

    feature_cols = [
    "mean_left_ear",
    "mean_right_ear",
    "std_left_ear",
    "std_right_ear",
    "left_x_variance",
    "left_y_variance",
    "right_x_variance",
    "right_y_variance",
    ]

    X = df[feature_cols].values
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)

        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))

    print("cross-validation results (5-fold)")
    print("accuracy mean:", np.mean(accuracies))
    print("accuracy std :", np.std(accuracies))
    print()
    print("macro f1 mean:", np.mean(f1_scores))
    print("macro f1 std :", np.std(f1_scores))


if __name__ == "__main__":
    main()
