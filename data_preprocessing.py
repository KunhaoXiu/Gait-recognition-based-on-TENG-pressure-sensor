import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _preprocess(data_dir, tag, val_size=0.2, test_size=0.2, random_state=42):
    """
    Load all CSV files in the folder,
    use columns 1~4 as features, last column as label.
    Split into train / val / test (default 60% / 20% / 20%).
    """
    frames = []
    for fname in sorted(f for f in os.listdir(data_dir) if f.endswith(".csv")):
        df = pd.read_csv(os.path.join(data_dir, fname))
        frames.append((df.iloc[:, [0, 1, 2, 3]].values.astype(float),
                       df.iloc[:, -1].values.astype(str)))

    X = np.vstack([f for f, _ in frames])
    y_str = np.concatenate([l for _, l in frames])

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # 先切出 test，再从剩余中切出 val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"[{tag}] classes：{list(le.classes_)}, "
          f"train：{len(y_train)}, val：{len(y_val)}, test：{len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, le


def preprocess_normal(data_dir="datasets/normal", val_size=0.2, test_size=0.2, random_state=42):
    return _preprocess(data_dir, "Normal", val_size, test_size, random_state)


def preprocess_abnormal(data_dir="datasets/abnormal", val_size=0.2, test_size=0.2, random_state=42):
    return _preprocess(data_dir, "Abnormal", val_size, test_size, random_state)
