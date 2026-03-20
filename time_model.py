import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# linear regression using the time series data from each frame individually
def main():
    data = np.load("sequence_dataset.npz", allow_pickle=True)
    X = data["X"] # shape (N, 30, 6) where N is #videos, 30 timesteps per video, 6 featurs per frame (left_ear, right_ear, left_x, left_y, right_x, right_y)
    y = data["y"]

    N = X.shape[0]

    # flatten time dimension
    X = X.reshape(N, -1)   # (N, 180)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    f1_scores = []

    for train_idx, test_idx in kf.split(X, y_enc):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        acc_scores.append(acc)
        f1_scores.append(f1)

    print("flattened logistic regression (5-fold)")
    print("accuracy mean:", np.mean(acc_scores))
    print("accuracy std :", np.std(acc_scores))
    print()
    print("macro f1 mean:", np.mean(f1_scores))
    print("macro f1 std :", np.std(f1_scores))


if __name__ == "__main__":
    main()
