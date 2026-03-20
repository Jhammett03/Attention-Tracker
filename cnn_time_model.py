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
        self.pool = nn.AdaptiveAvgPool1d(1) #collapse time steps into 1 value
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x)) #patterns
        x = self.relu(self.conv2(x)) #patterns of patterns
        x = self.pool(x) 
        x = x.squeeze(-1) #condensed to (batch, 32)
        x = self.fc(x)
        return x


def train_one_fold(X_train, y_train, X_val, y_val, device):

    model = TemporalCNN(in_channels=X_train.shape[1]).to(device) # slides window over time dimension
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3) #gradient descent

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(1000):

        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward() #compute new gradient
        optimizer.step() #grad descent

        model.eval() 
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model


def main():

    data = np.load("sequence_dataset.npz", allow_pickle=True)
    X = data["X"]      # (N, 30, 6)
    y = data["y"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    macro_f1s = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # normalize per feature channel using training data only
        scaler = StandardScaler()

        N_train, T, C = X_train.shape
        X_train_reshaped = X_train.reshape(-1, C)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(N_train, T, C)

        N_val = X_val.shape[0]
        X_val_reshaped = X_val.reshape(-1, C)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(N_val, T, C)

        # convert to torch
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1).to(device)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1).to(device)

        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

        model = train_one_fold(X_train_t, y_train_t, X_val_t, y_val_t, device)

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")

        accuracies.append(acc)
        macro_f1s.append(f1)

    print("temporal cnn (5-fold)")
    print("accuracy mean:", np.mean(accuracies))
    print("accuracy std :", np.std(accuracies))
    print()
    print("macro f1 mean:", np.mean(macro_f1s))
    print("macro f1 std :", np.std(macro_f1s))


if __name__ == "__main__":
    main()