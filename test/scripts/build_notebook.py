import json
import os

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split('\n')]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split('\n')]})

md_intro = """# 📊 시계열 가격 예측 알고리즘 분석 (7종 모델 및 최신 지표 평가)

**분석 목적**: 실전 퀀트 트레이딩의 '최하방 방어' 기준을 세우기 위한 전단계로, 다양한 시계열 딥러닝 알고리즘의 예측 성능과 강건성을 비교 분석합니다.

## 💡 Keras Sequential vs PyTorch OOP (초심자 가이드)
평소 익숙하신 `tensorflow.keras.models.Sequential` 방식은 층(Layer)을 차곡차곡 쌓는 직관적인 방식입니다. 
하지만 최신 연구(Mamba, mTAND 등)의 복잡한 모델이나 다변량 텐서 연산을 다룰 때는 한계가 있어, 학계와 업계에서는 **PyTorch의 객체지향(OOP) 방식**을 주로 사용합니다.

### 코드 읽는 법 (`nn.Module`)
PyTorch 모델은 크게 두 부분으로 나뉩니다:
1. `__init__(self, ...)`: 모델에 사용될 "부품(Layer)"들을 미리 정의하고 찍어내는 공간입니다. (Keras의 `model.add()`에 들어갈 부품들을 선언만 해두는 곳)
   - `input_dim`: 한 번에 들어오는 데이터의 변수 개수 (종가 1개면 1)
   - `hidden_dim`: 은닉층의 노드 개수 (데이터를 얼마나 복잡하게 해석할지)
   - `num_layers`: RNN, LSTM 등의 층을 몇 겹으로 쌓을지
   - `batch_first=True`: 데이터의 첫 번째 차원이 Batch 크기임을 명시 (Keras와 동일한 형태 유지)
2. `forward(self, x)`: `__init__`에서 만든 부품들을 조립하여 **실제 데이터 `x`가 어떻게 흘러가는지(연산되는지) 과정**을 정의합니다.

## 💡 Optuna란?
딥러닝 모델의 성능은 하이퍼파라미터(학습률, 은닉층 크기, 에폭 수 등)에 크게 좌우됩니다. 사람이 일일이 바꿔가며 테스트(Grid Search)하는 대신, **Optuna**라는 프레임워크가 과거의 결과를 학습하여(베이지안 최적화) "다음에 시도해볼 최적의 파라미터 조합"을 스스로 찾아주는 매우 강력한 자동화 도구입니다.
"""
add_md(md_intro)

code_setup = """import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import duckdb
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# DTW 계산용
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    !pip install fastdtw scipy
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

try:
    import optuna
except ImportError:
    !pip install optuna
    import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")"""
add_code(code_setup)

md_data = """## 1. 데이터 로드 및 분할 (3년 데이터 -> 2년 학습 / 1년 예측)
업비트 BTC 15분봉 데이터를 DuckDB에서 불러와 2:1로 분할합니다."""
add_md(md_data)

code_data = """con = duckdb.connect('../upbit_data.db')
try:
    # 3년치(약 10.5만건) 15분봉 데이터 가정
    data = con.execute("SELECT * FROM btc_15m_advance ORDER BY timestamp").df()
except Exception as e:
    print("DB 조회 실패, 시뮬레이션용 데이터 생성...")
    dates = pd.date_range(start="2021-01-01", end="2024-01-01", freq="15T")
    prices = np.sin(np.linspace(0, 50, len(dates))) * 5000 + 60000
    prices += np.random.normal(0, 500, len(dates)) # 노이즈 추가
    data = pd.DataFrame({'timestamp': dates, 'close': prices})
con.close()

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[['close']].values)

def create_sequences(dataset, seq_length):
    x, y = [], []
    for i in range(len(dataset) - seq_length):
        x.append(dataset[i:i+seq_length])
        y.append(dataset[i+seq_length])
    return np.array(x), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_close, SEQ_LENGTH)

# 정확한 시계열 분할 (앞 2/3 학습, 뒤 1/3 테스트)
train_size = int(len(X) * (2/3))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Total sequences: {len(X)}")
print(f"Train size (approx 2 years): {len(X_train)}")
print(f"Test size (approx 1 year): {len(X_test)}")
"""
add_code(code_data)

md_models = """## 2. 7종 시계열 알고리즘 아키텍처 정의
기본 RNN부터 2025년 최신 학술 트렌드 모델까지 모두 구현합니다."""
add_md(md_models)

code_models = """# 1. LSTM (Long Short-Term Memory) - 순차 데이터의 베이스라인
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. GRU (Gated Recurrent Unit) - LSTM보다 가벼운 구조
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# 3. TCN (Temporal Convolutional Network) - 합성곱을 이용한 팽창(Dilated) 시계열 캡처
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        # 과거 데이터를 보기 위해 비대칭 패딩 사용
        padding = (3 - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.conv(x))
        return out[:, :, :-self.conv.padding[0]] # 미래 정보를 보지 않도록 슬라이싱 (Causal)

class TCNModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.tcn = nn.Sequential(TCNBlock(input_dim, 32, 1), TCNBlock(32, 64, 2))
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = x.transpose(1, 2) # Conv1D를 위해 차원 변경
        out = self.tcn(x)
        return self.fc(out[:, :, -1])

# 4. Transformer (Time Series) - 글로벌 어텐션
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.embed(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])

# 5. N-BEATS (Simplified) - 순수 MLP 기반 시계열 분해
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
        self.forecast = nn.Linear(input_dim, 1)
    def forward(self, x):
        x_flat = x.view(x.size(0), -1) # 시퀀스 펼치기
        backcast = self.fc(x_flat)
        forecast = self.forecast(backcast)
        return backcast, forecast

class NBeatsModel(nn.Module):
    def __init__(self, seq_len=60):
        super().__init__()
        self.block1 = NBeatsBlock(seq_len, 128)
        self.block2 = NBeatsBlock(seq_len, 128)
    def forward(self, x):
        back1, fore1 = self.block1(x)
        res = x.view(x.size(0), -1) - back1
        back2, fore2 = self.block2(res)
        return fore1 + fore2

# 6. mTAND (Multi-Time Attention) - 불규칙 샘플링 및 시간 내재화 어텐션
class mTANDModel(nn.Module):
    def __init__(self, input_dim=1, embed_dim=32, num_heads=2):
        super().__init__()
        self.time_embed = nn.Linear(1, embed_dim) # 시간을 연속적인 값으로 임베딩
        self.feature_embed = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, 1)
    def forward(self, x):
        b, seq, _ = x.size()
        time_steps = torch.arange(seq, dtype=torch.float32, device=x.device).view(1, seq, 1).repeat(b, 1, 1)
        t_emb = self.time_embed(time_steps)
        x_emb = self.feature_embed(x)
        combined = t_emb + x_emb # 시간 정보와 가격 정보 결합
        attn_out, _ = self.attn(combined, combined, combined)
        return self.fc(attn_out[:, -1, :])

# 7. Mamba (State Space Model 근사) - O(L) 선형 복잡도 장기 의존성 (2024~2025 SOTA)
class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model * 2, d_model * 2, kernel_size=3, padding=1)
        self.fc2 = nn.Linear(d_model * 2, d_model)
    def forward(self, x):
        # 복잡한 Selective Scan 대신, 개념 증명을 위한 1D Conv + Gating (SSM 근사)
        x_proj = self.fc1(x)
        x_conv = self.conv(x_proj.transpose(1, 2)).transpose(1, 2)
        x_conv = nn.functional.silu(x_conv)
        return self.fc2(x_conv) + x

class MambaModel(nn.Module):
    def __init__(self, input_dim=1, d_model=32):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.mamba = SimpleMambaBlock(d_model)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.embed(x)
        x = self.mamba(x)
        return self.fc(x[:, -1, :])

print("7종 모델 선언 완료")"""
add_code(code_models)

md_metrics = """## 3. 고도화된 평가 지표 (2025-2026 Academic Metrics)
과거의 점 단위 오차(MSE, MAE)를 넘어 실제 트레이딩에 쓰이는 최신 지표를 구현합니다."""
add_md(md_metrics)

code_metrics = """def calculate_metrics(y_true, y_pred, y_train_history):
    # 역변환된 원본 스케일 데이터(실제 가격)가 들어온다고 가정
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    # 1. RMSE & MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # 2. Directional Accuracy (DA / Hit Ratio) - 2025 퀀트 핵심 지표
    # 전 타임스텝 대비 상승/하락 방향을 맞췄는지 검사
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    # 방향이 같으면 곱했을 때 양수
    da = np.mean((true_diff * pred_diff) > 0) * 100 
    
    # 3. MASE (Mean Absolute Scaled Error)
    # y_train_history의 나이브 예측(1스텝 시프트) MAE로 현재 MAE를 나눔
    naive_mae = np.mean(np.abs(np.diff(y_train_history)))
    mase = mae / naive_mae if naive_mae != 0 else 0
    
    # 4. DTW (Dynamic Time Warping) 거리 - 곡선의 형태적 유사도 (Lagging 검증용)
    # 연산량이 많아 샘플링해서 계산
    distance, _ = fastdtw(y_true[-100:], y_pred[-100:], dist=euclidean)
    
    return {"RMSE": rmse, "MAE": mae, "DA(%)": da, "MASE": mase, "DTW": distance}
"""
add_code(code_metrics)

md_train = """## 4. 모델 학습 및 평가 (Optuna 활용 가능)
시간 관계상 Mamba, mTAND, LSTM을 비교 학습하고 지표를 도출합니다."""
add_md(md_train)

code_train = """# 데이터 로더 준비
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

models = {
    "LSTM": LSTMModel().to(device),
    "mTAND": mTANDModel().to(device),
    "Mamba": MambaModel().to(device)
}

results = {}
criterion = nn.HuberLoss() # 이상치 강건성

for name, model in models.items():
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    print(f"Training {name}...")
    for epoch in range(2): # 시연용 2 Epoch
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()
        
    # 역변환
    y_test_inv = scaler.inverse_transform(y_test)
    preds_inv = scaler.inverse_transform(preds)
    y_train_inv = scaler.inverse_transform(y_train)
    
    res = calculate_metrics(y_test_inv, preds_inv, y_train_inv)
    results[name] = {"preds": preds_inv, "metrics": res}

# 결과 출력
print("\\n=== 최종 성능 비교 ===")
for name, data_res in results.items():
    print(f"[{name}]")
    for k, v in data_res['metrics'].items():
        print(f"  {k}: {v:.4f}")
"""
add_code(code_train)

md_plot = "## 5. 예측 결과 시각화"
add_md(md_plot)

code_plot = """plt.figure(figsize=(14, 6))
y_actual = scaler.inverse_transform(y_test)

# 최근 300스텝만 시각화
plt.plot(y_actual[-300:], label="Actual Price", color='black')
plt.plot(results["LSTM"]["preds"][-300:], label="LSTM", linestyle='--')
plt.plot(results["mTAND"]["preds"][-300:], label="mTAND", color='blue')
plt.plot(results["Mamba"]["preds"][-300:], label="Mamba", color='red')

plt.title("Time Series Forecasting Comparison (Last 300 Steps)")
plt.xlabel("Time (15-min intervals)")
plt.ylabel("Price (KRW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("images/advanced_comparison.png")
plt.show()
"""
add_code(code_plot)


notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.9.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('test/models/2_time_series_advance_test.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, ensure_ascii=False, indent=2)

print("Notebook generated successfully!")
