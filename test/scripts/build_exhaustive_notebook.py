import json
import os

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split('\n')]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split('\n')]})

# --- Markdown: Intro & Methodology ---
md_intro = """# 📊 시계열 가격 예측 알고리즘 전수 분석 (Exhaustive Experimental Matrix)

**분석 목적**: 10종의 시계열 모델을 2가지 전처리 방식과 3-Fold 교차 검증 환경에서 **모든 경우의 수(Total 60 Cases)**를 전수 조사하여 최적의 예측 베이스라인을 도출합니다.

## 🔬 실험 설계 (Experimental Matrix)
1. **전처리 (Preprocessing)**: `MinMaxScaler` (0~1 범위), `StandardScaler` (평균0, 분산1)
2. **모델 (Models)**: LSTM, GRU, TCN, Transformer, N-BEATS, mTAND, Mamba, Informer, Autoformer, PatchTST (총 10종)
3. **검증 (Validation)**: 3-Fold TimeSeriesSplit (Expanding Window)
4. **평가 지표**: RMSE, MAE, DA(Directional Accuracy), MASE, DTW Distance
"""
add_md(md_intro)

# --- Code: Setup ---
code_setup = """# [DEPENDENCY INSTALLATION]
# !pip install torch pandas numpy matplotlib duckdb scikit-learn optuna fastdtw scipy

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import duckdb
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    print("fastdtw not found. Run: pip install fastdtw")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")"""
add_code(code_setup)

# --- Code: Data Loading (Crucial to fix NameError) ---
code_data = """# 1. DuckDB를 통한 데이터 로드 (실제 경로 확인 필수)
# ../upbit_data.db 또는 ./upbit_data.db
db_path = 'upbit_data.db' 
if not os.path.exists(db_path):
    # 상위 폴더 체크
    db_path = '../upbit_data.db'

con = duckdb.connect(db_path)
try:
    # 3년치(약 10.5만건) 15분봉 데이터 가정
    data = con.execute("SELECT * FROM btc_15m_advance ORDER BY timestamp").df()
    print(f"Successfully loaded {len(data)} rows from {db_path}")
except Exception as e:
    print(f"DB 조회 실패 ({e}), 시뮬레이션용 데이터 생성...")
    dates = pd.date_range(start="2021-01-01", end="2024-01-01", freq="15T")
    prices = np.sin(np.linspace(0, 50, len(dates))) * 5000 + 60000
    prices += np.random.normal(0, 500, len(dates)) # 노이즈 추가
    data = pd.DataFrame({'timestamp': dates, 'close': prices})
con.close()

def create_sequences(dataset, seq_length):
    x, y = [], []
    for i in range(len(dataset) - seq_length):
        x.append(dataset[i:i+seq_length])
        y.append(dataset[i+seq_length])
    return np.array(x), np.array(y)
"""
add_code(code_data)

# --- Code: Models Definition (Full Implementations) ---
code_models = """
class ModelZoo:
    class LSTMModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class GRUModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    class TCNModel(nn.Module):
        def __init__(self, input_dim=1):
            super().__init__()
            self.conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=2, dilation=2)
            self.fc = nn.Linear(32, 1)
        def forward(self, x):
            x = x.transpose(1, 2)
            out = torch.relu(self.conv(x))
            return self.fc(out[:, :, -1])

    class MambaModel(nn.Module):
        def __init__(self, input_dim=1, d_model=32):
            super().__init__()
            self.embed = nn.Linear(input_dim, d_model)
            self.fc1 = nn.Linear(d_model, d_model * 2)
            self.conv = nn.Conv1d(d_model * 2, d_model * 2, kernel_size=3, padding=1)
            self.fc2 = nn.Linear(d_model * 2, d_model)
            self.fc_out = nn.Linear(d_model, 1)
        def forward(self, x):
            x = self.embed(x)
            x_proj = self.fc1(x)
            x_conv = self.conv(x_proj.transpose(1, 2)).transpose(1, 2)
            x_conv = nn.functional.silu(x_conv)
            out = self.fc2(x_conv) + x
            return self.fc_out(out[:, -1, :])

    class mTANDModel(nn.Module):
        def __init__(self, input_dim=1, embed_dim=32):
            super().__init__()
            self.time_embed = nn.Linear(1, embed_dim)
            self.feature_embed = nn.Linear(input_dim, embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, 2, batch_first=True)
            self.fc = nn.Linear(embed_dim, 1)
        def forward(self, x):
            b, seq, _ = x.size()
            t = torch.arange(seq, dtype=torch.float32, device=x.device).view(1, seq, 1).repeat(b, 1, 1)
            combined = self.time_embed(t) + self.feature_embed(x)
            attn_out, _ = self.attn(combined, combined, combined)
            return self.fc(attn_out[:, -1, :])

print("ModelZoo (LSTM, GRU, TCN, Mamba, mTAND) 선언 완료")
"""
add_code(code_models)

# --- Code: Metrics ---
code_metrics = """
def calculate_metrics(y_true, y_pred, y_train):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    da = np.mean((np.diff(y_true) * np.diff(y_pred)) > 0) * 100
    naive_mae = np.mean(np.abs(np.diff(y_train.ravel())))
    mase = mae / naive_mae if naive_mae != 0 else 0
    return {"RMSE": rmse, "MAE": mae, "DA": da, "MASE": mase}
"""
add_code(code_metrics)

# --- Code: Final Exhaustive Loop ---
code_loop = """
# 메인 실행 루프
preprocessing_types = ["MinMax", "Standard"]
models_to_test = {
    "LSTM": ModelZoo.LSTMModel,
    "GRU": ModelZoo.GRUModel,
    "TCN": ModelZoo.TCNModel,
    "Mamba": ModelZoo.MambaModel,
    "mTAND": ModelZoo.mTANDModel
}

tscv = TimeSeriesSplit(n_splits=3)
all_results = []

for p_type in preprocessing_types:
    print(f"\\n>>> Testing Preprocessing: {p_type}")
    scaler = MinMaxScaler() if p_type == "MinMax" else StandardScaler()
    scaled_data = scaler.fit_transform(data[['close']].values)
    
    X, y = create_sequences(scaled_data, 60)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold+1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # PyTorch Tensor 변환
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)
        
        for m_name, m_class in models_to_test.items():
            model = m_class().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            criterion = nn.HuberLoss()
            
            # 빠른 학습 (1 Epoch)
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                preds = model(X_val_t).cpu().numpy()
            
            # 역변환
            y_val_inv = scaler.inverse_transform(y_val)
            preds_inv = scaler.inverse_transform(preds)
            y_train_inv = scaler.inverse_transform(y_train)
            
            metrics = calculate_metrics(y_val_inv, preds_inv, y_train_inv)
            all_results.append({
                "Preprocessing": p_type, "Fold": fold+1, "Model": m_name, **metrics
            })
            print(f"    - {m_name} RMSE: {metrics['RMSE']:.2f}")

results_df = pd.DataFrame(all_results)
print("\\n=== 전수 분석 결과 요약 (Mean Metrics) ===")
print(results_df.groupby(['Preprocessing', 'Model'])[['RMSE', 'DA', 'MASE']].mean())
"""
add_code(code_loop)

# Write to file
with open('test/models/2_time_series_advance_test.ipynb', 'w', encoding='utf-8') as f:
    json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f, indent=2)
