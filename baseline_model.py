import argparse
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


OLD_FEATURE_COLS = [
    "mean_left_ear", "mean_right_ear", "std_left_ear", "std_right_ear",
    "left_x_variance", "left_y_variance", "right_x_variance", "right_y_variance",
]

IRIS_FEATURE_COLS = [
    "mean_left_ear", "mean_right_ear", "std_left_ear", "std_right_ear",
    "mean_left_iris_x", "mean_left_iris_y", "mean_right_iris_x", "mean_right_iris_y",
    "var_left_iris_x", "var_left_iris_y", "var_right_iris_x", "var_right_iris_y",
]

GAZE_FEATURE_COLS = [
    "mean_left_ear", "mean_right_ear", "std_left_ear", "std_right_ear",
    "mean_left_gaze_x", "mean_left_gaze_y", "mean_right_gaze_x", "mean_right_gaze_y",
    "var_left_gaze_x", "var_left_gaze_y", "var_right_gaze_x", "var_right_gaze_y",
    "mean_head_x", "mean_head_y", "var_head_x", "var_head_y",
    "mean_bb_tl_x", "mean_bb_tl_y", "mean_bb_tr_x", "mean_bb_tr_y",
    "mean_bb_bl_x", "mean_bb_bl_y", "mean_bb_br_x", "mean_bb_br_y",
    "var_bb_tl_x", "var_bb_tl_y", "var_bb_tr_x", "var_bb_tr_y",
    "var_bb_bl_x", "var_bb_bl_y", "var_bb_br_x", "var_bb_br_y",
]


def choose_feature_cols(df):
    cols = set(df.columns)
    if all(col in cols for col in GAZE_FEATURE_COLS):
        return GAZE_FEATURE_COLS, "gaze"
    elif all(col in cols for col in IRIS_FEATURE_COLS):
        return IRIS_FEATURE_COLS, "iris"
    elif all(col in cols for col in OLD_FEATURE_COLS):
        return OLD_FEATURE_COLS, "classic"
    else:
        raise ValueError(f"could not match dataset format.\navailable columns:\n{list(df.columns)}")


def run_cv(df, feature_cols):
    X = df[feature_cols].values
    y = df["label"].values

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

    return accuracies, f1_scores


def evaluate_csv(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols, dataset_type = choose_feature_cols(df)

    print(f"\nevaluating: {csv_path}")
    print(f"detected dataset type: {dataset_type}")

    accuracies, f1_scores = run_cv(df, feature_cols)

    print("\ncross-validation results (5-fold)")
    print("accuracy mean:", np.mean(accuracies))
    print("accuracy std :", np.std(accuracies))
    print("macro f1 mean:", np.mean(f1_scores))
    print("macro f1 std :", np.std(f1_scores))


def save_model(csv_path="dataset_gaze.csv", output="baseline_model.pkl"):
    df = pd.read_csv(csv_path)
    feature_cols, dataset_type = choose_feature_cols(df)
    X = df[feature_cols].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y_enc)

    with open(output, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler, "le": le, "feature_cols": feature_cols}, f)
    print(f"saved {dataset_type} logistic regression → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="train on all data and save model")
    parser.add_argument("--csv", default="dataset_gaze.csv")
    args = parser.parse_args()

    if args.save:
        save_model(csv_path=args.csv)
    else:
        evaluate_csv(args.csv)


if __name__ == "__main__":
    main()
