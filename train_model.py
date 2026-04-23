"""
Train the PyTorch heart disease classification model and export as pickle files.
Based on Assignment.ipynb - reproduces the exact same model architecture.

Usage:
    python train_model.py

Outputs (in ml_artifacts/):
    - model.pkl        : PyTorch model state dict
    - scaler.pkl       : Fitted StandardScaler
    - feature_columns.pkl : Feature column names after encoding
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


# ─── Model Definition (same as notebook) ───
class HeartDiseaseModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.network(features)


def train():
    print("=" * 60)
    print("  Heart Disease Model Training")
    print("=" * 60)

    # ─── 1. Load Dataset ───
    df = pd.read_csv("heart.csv")
    print(f"\n[1] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"    Target distribution: {dict(df['HeartDisease'].value_counts())}")

    # ─── 2. Prepare Features ───
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    feature_columns = list(X.columns)
    print(f"\n[2] Features after encoding: {len(feature_columns)} columns")
    print(f"    Columns: {feature_columns}")

    # ─── 3. Split Data ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[3] Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # ─── 4. Scale Numeric Features ───
    cols_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    print(f"\n[4] Scaled columns: {cols_to_scale}")

    # ─── 5. Convert to Tensors ───
    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    # ─── 6. Build Model ───
    num_features = X_train_tensor.shape[1]
    model = HeartDiseaseModel(num_features)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[5] Model created: {total_params} trainable parameters")

    # ─── 7. Train ───
    epochs = 200
    print(f"\n[6] Training for {epochs} epochs...")
    print("-" * 40)

    for epoch in range(epochs):
        y_pred = model(X_train_tensor)
        loss = loss_function(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1:3d}/{epochs}], Loss: {loss.item():.4f}")

    # ─── 8. Evaluate ───
    print("-" * 40)
    with torch.no_grad():
        train_preds = model(X_train_tensor).round()
        train_acc = (train_preds == y_train_tensor).float().mean().item()

        test_preds = model(X_test_tensor).round()
        test_acc = (test_preds == y_test_tensor).float().mean().item()

    print(f"\n[7] Results:")
    print(f"    Training Accuracy: {train_acc * 100:.2f}%")
    print(f"    Testing Accuracy:  {test_acc * 100:.2f}%")

    # ─── 9. Save Artifacts ───
    os.makedirs("ml_artifacts", exist_ok=True)

    with open("ml_artifacts/model.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)

    with open("ml_artifacts/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("ml_artifacts/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print(f"\n[8] Artifacts saved to ml_artifacts/")
    print(f"    - model.pkl ({os.path.getsize('ml_artifacts/model.pkl')} bytes)")
    print(f"    - scaler.pkl ({os.path.getsize('ml_artifacts/scaler.pkl')} bytes)")
    print(f"    - feature_columns.pkl ({os.path.getsize('ml_artifacts/feature_columns.pkl')} bytes)")
    print(f"\n{'=' * 60}")
    print("  Training complete! Model ready for deployment.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
