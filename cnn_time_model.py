import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


class TemporalCNN(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


def train_one_fold(X_train, y_train, X_val, y_val, device):
    model = TemporalCNN(in_channels=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def build_predictor(npz_path="sequence_dataset_gaze.npz"):
    """Train on all data and return (model, scaler, le, device) ready for inference."""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_raw = data["y"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    N, T, C = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, C)).reshape(N, T, C)

    device = torch.device("cpu")
    X_t = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)

    model = train_one_fold(X_t, y_t, X_t, y_t, device)
    model.eval()
    print(f"  trained on {N} samples — classes: {list(le.classes_)}")
    return model, scaler, le, device, C


def save_model(npz_path="sequence_dataset_gaze.npz", output="cnn_model.pt"):
    model, scaler, le, _, in_channels = build_predictor(npz_path)
    torch.save({
        "model_state": model.state_dict(),
        "in_channels": in_channels,
        "scaler": scaler,
        "le": le,
    }, output)
    print(f"saved temporal CNN → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="train on all data and save model")
    parser.add_argument("--npz", default="sequence_dataset_gaze.npz")
    args = parser.parse_args()

    if args.save:
        save_model(npz_path=args.npz)
        return

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y = LabelEncoder().fit_transform(data["y"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    macro_f1s = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        N_train, T, C = X_train.shape
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N_train, T, C)
        X_val = scaler.transform(X_val.reshape(-1, C)).reshape(X_val.shape[0], T, C)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

        model = train_one_fold(X_train_t, y_train_t, X_val_t, y_val_t, device)

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val_t), dim=1).cpu().numpy()

        accuracies.append(accuracy_score(y_val, preds))
        macro_f1s.append(f1_score(y_val, preds, average="macro"))

    print("temporal cnn (5-fold)")
    print("accuracy mean:", np.mean(accuracies))
    print("accuracy std :", np.std(accuracies))
    print("macro f1 mean:", np.mean(macro_f1s))
    print("macro f1 std :", np.std(macro_f1s))


if __name__ == "__main__":
    main()
