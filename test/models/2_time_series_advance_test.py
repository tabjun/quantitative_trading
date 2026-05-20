# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual execution should be performed in the Jupyter Notebook (.ipynb).

# %%
"""
# 📊 시계열 가격 예측 알고리즘 전수 분석
분석 목적: 15종 이상의 시계열 모델을 3년치 15분봉 데이터를 기반으로 전수 조사합니다.

"""

# %%
# [1] 라이브러리 로드 및 환경 설정
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import os
import sys
import time
import pyupbit
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
# [2] DuckDB 접속 및 3년치 데이터 강제 확보
db_path = 'upbit_data.db'

def fetch_and_save_3_years_data():
    print(">>> [데이터 수집 시작] 3년치(약 10.5만건) 15분봉 데이터를 수집합니다...")
    df_list = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    curr = end_date
    while curr > start_date:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute15", to=curr.strftime("%Y-%m-%d %H:%M:%S"), count=200)
        if df is None or df.empty: break
        df_list.append(df)
        curr = df.index[0]
        time.sleep(0.05)
    full_df = pd.concat(df_list).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    full_df.reset_index(inplace=True); full_df.rename(columns={'index': 'timestamp'}, inplace=True)
    with duckdb.connect(db_path) as con:
        con.execute("CREATE OR REPLACE TABLE btc_15m_advance AS SELECT * FROM full_df")
    return full_df

try:
    with duckdb.connect(db_path) as con:
        data = con.execute("SELECT * FROM btc_15m_advance ORDER BY timestamp").df()
except:
    data = fetch_and_save_3_years_data()

print("\n=== 데이터 기초 통계량 (Describe) ===")
desc_df = data.describe()
print(desc_df)

def create_sequences(dataset, seq_length):
    x, y = [], []
    for i in range(len(dataset) - seq_length):
        x.append(dataset[i:i+seq_length]); y.append(dataset[i+seq_length])
    return np.array(x), np.array(y)


# %%
# [3] 15종 모델 정의
class ModelZoo:
    class LinearModel(nn.Module):
        def __init__(self, seq_len=60): super().__init__(); self.fc = nn.Linear(seq_len, 1)
        def forward(self, x): return self.fc(x.squeeze(-1))
    class LSTMModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super().__init__(); self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])
    class GRUModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super().__init__(); self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): out, _ = self.gru(x); return self.fc(out[:, -1, :])
    class TCNModel(nn.Module):
        def __init__(self, input_dim=1):
            super().__init__(); self.conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=2, dilation=2); self.fc = nn.Linear(32, 1)
        def forward(self, x): x = x.transpose(1, 2); return self.fc(torch.relu(self.conv(x))[:, :, -1])
    class TransformerModel(nn.Module):
        def __init__(self, input_dim=1, d_model=32):
            super().__init__(); self.enc = nn.Linear(input_dim, d_model); self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True), 2); self.fc = nn.Linear(d_model, 1)
        def forward(self, x): return self.fc(self.tf(self.enc(x))[:, -1, :])
    class InformerModel(nn.Module):
        def __init__(self, input_dim=1, d_model=32):
            super().__init__(); self.enc = nn.Linear(input_dim, d_model); self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True, dropout=0.1), 2); self.fc = nn.Linear(d_model, 1)
        def forward(self, x): return self.fc(self.tf(self.enc(x))[:, -1, :])
    class AutoformerModel(nn.Module):
        def __init__(self, input_dim=1, d_model=32):
            super().__init__(); self.avg = nn.AvgPool1d(3, 1, 1); self.enc = nn.Linear(input_dim, d_model); self.fc = nn.Linear(d_model, 1)
        def forward(self, x): t = self.avg(x.transpose(1, 2)).transpose(1, 2); r = x - t; return self.fc(self.enc(r + t)[:, -1, :])
    class PatchTSTModel(nn.Module):
        def __init__(self, input_dim=1, d_model=32):
            super().__init__(); self.enc = nn.Linear(5, d_model); self.fc = nn.Linear(d_model, 1)
        def forward(self, x): p = x[:, -5:, :].transpose(1, 2); return self.fc(self.enc(p).squeeze(1))
    class MambaModel(nn.Module):
        def __init__(self, input_dim=1, d_model=32):
            super().__init__(); self.proj = nn.Linear(input_dim, d_model); self.conv = nn.Conv1d(d_model, d_model, 3, 1); self.fc = nn.Linear(d_model, 1)
        def forward(self, x): x = self.proj(x).transpose(1, 2); x = torch.relu(self.conv(x)).transpose(1, 2); return self.fc(x[:, -1, :])
    class mTANDModel(nn.Module):
        def __init__(self, input_dim=1, embed_dim=32):
            super().__init__(); self.te = nn.Linear(1, embed_dim); self.fe = nn.Linear(input_dim, embed_dim); self.attn = nn.MultiheadAttention(embed_dim, 2, batch_first=True); self.fc = nn.Linear(embed_dim, 1)
        def forward(self, x): b, s, _ = x.size(); t = torch.arange(s, dtype=torch.float32, device=x.device).view(1, s, 1).repeat(b, 1, 1); c = self.te(t) + self.fe(x); o, _ = self.attn(c, c, c); return self.fc(o[:, -1, :])
    class ODERNNModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.rnn = nn.GRUCell(input_dim, hidden_dim); self.ode = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)); self.fc = nn.Linear(hidden_dim, 1); self.h_dim = hidden_dim
        def forward(self, x):
            b, s, _ = x.size(); h = torch.zeros(b, self.h_dim).to(x.device)
            for t in range(s): h = self.rnn(x[:, t, :], h); h = h + self.ode(h) * 0.1
            return self.fc(h)
    class NBeatsModel(nn.Module):
        def __init__(self, input_dim=1, seq_len=60):
            super().__init__(); self.fc = nn.Sequential(nn.Linear(seq_len, 128), nn.ReLU(), nn.Linear(128, 1))
        def forward(self, x): return self.fc(x.squeeze(-1))
    class NonStatTFModel(nn.Module):
        def __init__(self, input_dim=1, d_model=32):
            super().__init__(); self.enc = nn.Linear(input_dim, d_model); self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True), 2); self.fc = nn.Linear(d_model, 1)
        def forward(self, x): mu = torch.mean(x, 1, True); x = x - mu; return self.fc(self.tf(self.enc(x))[:, -1, :]) + mu[:, -1, :]
    class DeepARModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])
    class LinearDecompModel(nn.Module):
        def __init__(self, seq_len=60): super().__init__(); self.t_fc = nn.Linear(seq_len, 1); self.r_fc = nn.Linear(seq_len, 1)
        def forward(self, x): x = x.squeeze(-1); return self.t_fc(x) + self.r_fc(x)


# %%
# [4] 전수 조사 실행
models_to_test = {
    'Linear': ModelZoo.LinearModel, 'LSTM': ModelZoo.LSTMModel, 'GRU': ModelZoo.GRUModel,
    'TCN': ModelZoo.TCNModel, 'Transformer': ModelZoo.TransformerModel, 'Informer': ModelZoo.InformerModel,
    'Autoformer': ModelZoo.AutoformerModel, 'PatchTST': ModelZoo.PatchTSTModel, 'Mamba': ModelZoo.MambaModel,
    'mTAND': ModelZoo.mTANDModel, 'ODE-RNN': ModelZoo.ODERNNModel, 'N-BEATS': ModelZoo.NBeatsModel,
    'NonStat-TF': ModelZoo.NonStatTFModel, 'DeepAR': ModelZoo.DeepARModel, 'Linear-Decomp': ModelZoo.LinearDecompModel
}

def calculate_metrics(y_true, y_pred, y_train):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    da = np.mean((np.diff(y_true.ravel(), append=y_true.ravel()[-1:]) * np.diff(y_pred.ravel(), append=y_pred.ravel()[-1:])) > 0) * 100
    naive_mae = np.mean(np.abs(np.diff(y_train.ravel())))
    return {"RMSE": rmse, "MAE": mae, "DA": da, "MASE": mae / naive_mae if naive_mae != 0 else 0}

all_results = []; history_dict = {}; predictions_dict = {}
plot_p_type = "MinMax"

for p_type in ["MinMax", "Standard"]:
    print(f"\n>>> Testing Preprocessing: {p_type}")
    scaler = MinMaxScaler() if p_type == "MinMax" else StandardScaler()
    scaled_data = scaler.fit_transform(data[['close']].values)
    X, y = create_sequences(scaled_data, 60)
    train_idx = int(len(X) * 0.8)
    X_train, X_val = X[:train_idx], X[train_idx:]
    y_train, y_val = y[:train_idx], y[train_idx:]
    
    batch_size = 4096
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                            torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    
    for m_name, m_class in models_to_test.items():
        if m_name in ["Linear", "N-BEATS", "Linear-Decomp"]: model = m_class(seq_len=60).to(device)
        else: model = m_class().to(device)
        opt = optim.AdamW(model.parameters(), lr=0.005); crit = nn.HuberLoss()
        model.train(); hist = []
        best_loss = float('inf'); patience = 5; patience_counter = 0
        for epoch in range(100):
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device); out = model(xb); loss = crit(out, yb)
                opt.zero_grad(); loss.backward(); opt.step(); epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader); hist.append(avg_loss)
            if avg_loss < best_loss - 1e-4: best_loss = avg_loss; patience_counter = 0
            else: patience_counter += 1
            if patience_counter >= patience: break
        model.eval(); preds_list = []
        with torch.no_grad():
            for (xb,) in val_loader: xb = xb.to(device); preds_list.append(model(xb).cpu().numpy())
        preds = np.concatenate(preds_list)
        y_val_inv = scaler.inverse_transform(y_val); preds_inv = scaler.inverse_transform(preds); y_train_inv = scaler.inverse_transform(y_train)
        metrics = calculate_metrics(y_val_inv, preds_inv, y_train_inv)
        all_results.append({"Model": m_name, "P_Type": p_type, **metrics})
        if p_type == plot_p_type:
            history_dict[m_name] = hist; predictions_dict[m_name] = preds_inv.flatten()
            if "Actual" not in predictions_dict: predictions_dict["Actual"] = y_val_inv.flatten()
        print(f"    - {m_name} 완료")

results_df = pd.DataFrame(all_results)
summary = results_df.groupby(['P_Type', 'Model'])[['RMSE', 'DA', 'MASE']].mean().sort_values('RMSE')
print(summary)

# 메타데이터 저장
def json_serial(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

metadata = {"results": all_results, "describe": desc_df.to_dict(), "history": {k: [float(x) for x in v] for k, v in history_dict.items()}, "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")}
os.makedirs('test/results', exist_ok=True)
with open('test/results/metadata.json', 'w') as f: json.dump(metadata, f, default=json_serial)


# %%
# [5] 심층 시각화 (DPI 300 고해상도 인라인 플롯 구성)
import os

# 전역 DPI 300 설정 (인라인 아웃풋 셀 렌더링 화질 극대화)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("\n[1] 모델별 학습 손실(Loss) 곡선 비교")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), dpi=300)
tf_related = ['Transformer', 'Informer', 'Autoformer', 'PatchTST', 'NonStat-TF']
for m_name in tf_related:
    if m_name in history_dict: ax1.plot(history_dict[m_name], label=m_name)
ax1.set_title('Training Loss: Transformer-based Models'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
others = [m for m in models_to_test.keys() if m not in tf_related]
for m_name in others:
    if m_name in history_dict: ax2.plot(history_dict[m_name], label=m_name)
ax2.set_title('Training Loss: RNN, CNN, and Hybrid Models'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(ncol=2)
plt.tight_layout()
plt.show()

print("\n[2] 실제 가격 vs 상위 모델 예측 비교")
plt.figure(figsize=(20, 8), dpi=300)
plt.plot(predictions_dict['Actual'][-300:], label='Actual BTC Price', color='black', linewidth=3)
top_models = summary.loc[plot_p_type].head(5).index.tolist()
for m in top_models: plt.plot(predictions_dict[m][-300:], label=f'{m} (Pred)', linestyle='--', alpha=0.7)
plt.title(f'Price Prediction Comparison - Top 5 Models ({plot_p_type})'); plt.legend(); plt.grid(True, alpha=0.2)
plt.show()

print("\n[3] 최우수 모델 잔차(Residual) 분석")
plt.figure(figsize=(15, 5), dpi=300)
residuals = predictions_dict['Actual'] - predictions_dict[top_models[0]]
sns.histplot(residuals, kde=True, color='teal'); plt.axvline(0, color='red', linestyle='--')
plt.title(f'Residual Distribution - Best Model ({top_models[0]})')
plt.show()

# 이미지 경로는 notebook_to_md.py 추출 규격에 맞게 자동 지정되도록 아카이빙합니다.
os.makedirs('test/results', exist_ok=True)
with open('test/results/img_paths.json', 'w') as f:
    json.dump({
        "loss": "test/images/2_time_series_advance_test_plot_1.png",
        "pred": "test/images/2_time_series_advance_test_plot_2.png",
        "resid": "test/images/2_time_series_advance_test_plot_3.png"
    }, f)

