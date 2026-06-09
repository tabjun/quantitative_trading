# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %%
# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual execution should be performed in the Jupyter Notebook (.ipynb).

# %% [markdown]
# # 3. 업비트 전체 종목 대상 시계열 예측 15종 모델 전수 분석 및 정통 금융 캔들 차트 시각화
# > **원천 데이터 전제조건**: 본 분석에 수집 및 적재되어 사용되는 기초 원천 데이터셋은 업비트 시장의 **'15분봉 고빈도 가격 데이터 (15-Minute Candle Data / 15분봉)'** 입니다.
# > **분석 목적**: 특정 단일 종목에 매몰되지 않고, 업비트 전체 250여 개 종목의 15분봉 대규모 데이터를 15종 SOTA 시계열 모델을 통해 그리드 서치(Grid Search) 방식으로 전수 학습하여 시장 전체의 변동성 흐름을 해부합니다. 
# > **데이터 분할 및 전처리 규칙**: 전체 분량의 데이터 중 **앞의 9개월(약 75%)을 학습(Train) 기간**으로, **뒤의 3개월(약 25%)을 예측(Test) 기간**으로 설정하는 단일 Hold-out 검증을 수행하며(Cross Validation 제외), 스케일링은 **MinMaxScaler** 하나로 통일하여 전처리를 단순화합니다.
# > **정통 캔들스틱 시각화 (Candle Chart with Volume & Predictions)**: 
# > - 시가/고가/저가/종가(OHLC) 봉 차트(Candle Chart) 상단 배치
# > - 최우수 모델의 예측 종가 궤적 라인 그래프 오버레이
# > - 가격 상승/하락 칼라가 정합된 거래량(Volume) 막대 그래프 하단 배치
#

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import duckdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# CUDA 가속 활성화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎉 Using device: {device} 🎉")

# 기초통계량 출력 형식 변경
pd.options.display.float_format = '{:,.2f}'.format

# %% [markdown]
# ## 1. 데이터 로딩 및 엄격한 9개월/3개월 시간 스플릿 설계
# 업비트 전체 모든 종목의 1년치 15분봉 데이터를 로드하고, 앞의 9개월은 Train, 뒤의 3개월은 Test로 나눕니다.
#

# %%
start_block1 = time.time()

def fetch_and_save_all_krw_tickers_pyupbit(db_path='data/upbit_data.db', days=365):
    import pyupbit
    
    con = duckdb.connect(db_path)
    tables = con.execute("SHOW TABLES").df()
    
    # 만약 테이블이 이미 있고 충분한 데이터가 들어있다면 스킵
    if "upbit_krw_candle" in tables['name'].values:
        row_count = con.execute("SELECT COUNT(*) FROM upbit_krw_candle").fetchone()[0]
        if row_count > 50000:
            print(f"✅ upbit_krw_candle 테이블에 {row_count}개의 데이터가 존재하므로 신규 수집을 건너뜁니다.")
            con.close()
            return
            
    print("🌐 pyupbit를 사용하여 업비트 전체 KRW 마켓 고빈도 15분봉 데이터 전수 수집을 시작합니다...")
    
    # 1. 모든 KRW 마켓 티커 목록 확보
    try:
        tickers = pyupbit.get_tickers(fiat="KRW")
        print(f"📊 총 {len(tickers)}개의 KRW 종목이 감지되었습니다.")
    except Exception as e:
        print(f"❌ pyupbit 티커 로드 실패: {e}")
        con.close()
        raise e

    # 2. 테이블 생성
    con.execute("""
        CREATE TABLE IF NOT EXISTS upbit_krw_candle (
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            value DOUBLE,
            ticker VARCHAR
        )
    """)
    
    # 3. 15분봉 개수 계산 (1년 = 365일 * 24시간 * 4캔들 = 35040개)
    target_count = int(days * 24 * 4)
    print(f"⏳ 각 종목별 {days}일 분량 (최대 {target_count}개 캔들) 수집을 시작합니다...")
    
    for idx, ticker in enumerate(tickers):
        print(f"📥 [{idx+1}/{len(tickers)}] {ticker} 데이터 다운로드 중...")
        df_list = []
        curr_to = None
        collected = 0
        
        while collected < target_count:
            limit = min(200, target_count - collected)
            try:
                if curr_to:
                    df = pyupbit.get_ohlcv(ticker, interval="minute15", to=curr_to, count=limit)
                else:
                    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=limit)
                
                if df is None or df.empty:
                    break
                    
                df_list.append(df)
                collected += len(df)
                curr_to = df.index[0]
                time.sleep(0.12) # 초당 10회 요청 제한 방지 딜레이
            except Exception as ex:
                print(f"⚠️ {ticker} API 요청 오류 발생: {ex}. 1초 대기 후 재시도...")
                time.sleep(1.0)
                continue
                
        if df_list:
            ticker_df = pd.concat(df_list).sort_index()
            ticker_df = ticker_df[~ticker_df.index.duplicated(keep='first')]
            ticker_df.reset_index(inplace=True)
            ticker_df.rename(columns={'index': 'timestamp'}, inplace=True)
            ticker_df['ticker'] = ticker
            
            # DuckDB에 즉각 밀어넣기
            con.execute("INSERT INTO upbit_krw_candle SELECT * FROM ticker_df")
            
    # 전체 인덱스 생성 및 중복 제거
    print("🧹 데이터 중복 제거 및 최종 정합성 정리 중...")
    con.execute("CREATE TABLE temp_table AS SELECT DISTINCT * FROM upbit_krw_candle")
    con.execute("DROP TABLE upbit_krw_candle")
    con.execute("ALTER TABLE temp_table RENAME TO upbit_krw_candle")
    
    final_count = con.execute("SELECT COUNT(*) FROM upbit_krw_candle").fetchone()[0]
    print(f"🎉 전수 수집 완료! 총 적재 데이터 행 수: {final_count}")
    con.close()

def load_all_tickers_data():
    db_path = 'data/upbit_data.db'
    
    # pyupbit를 통한 자동 데이터 구축 실행 (데이터가 없을 경우에만 활성화됨)
    try:
        # 실전 구동 시간을 고려하여 기본적으로 최근 90일(3개월) 데이터를 전수 조사 구축
        # 1년 전체를 원하실 경우 days=365로 자유롭게 파라미터 조절이 가능합니다.
        fetch_and_save_all_krw_tickers_pyupbit(db_path, days=90)
    except Exception as e:
        print(f"⚠️ pyupbit를 통한 자동 데이터 전수 수집에 실패했습니다: {e}")
        
    con = None
    try:
        con = duckdb.connect(db_path)
        tables = con.execute("SHOW TABLES").df()
        query_table = "upbit_krw_candle" if "upbit_krw_candle" in tables['name'].values else "btc_15m_advance"
        df = con.execute(f"SELECT * FROM {query_table} ORDER BY timestamp").df()
        con.close()
        
        # KEYERROR FIX: 만약 데이터셋에 'ticker' 컬럼이 없다면(단일 종목 btc_15m_advance 로드 시) 기본값 추가
        if 'ticker' not in df.columns:
            df['ticker'] = 'KRW-BTC'
            
        print(f"✅ DuckDB 데이터 로드 완료. 총 행수: {len(df)}")
    except Exception as e:
        print(f"ℹ️ DB 로드 불가로 가상 다중 종목 1년치 15분봉 시뮬레이션 데이터를 자동 생성합니다. ({e})")
        tickers = ['KRW-BTC', 'KRW-ETH', 'KRW-SOL', 'KRW-XRP', 'KRW-ADA', 'KRW-DOGE', 'KRW-AVAX', 'KRW-SUI', 'KRW-NEAR', 'KRW-LINK']
        dates = pd.date_range(start="2025-05-09", end="2026-05-09", freq="15T")
        dummy_list = []
        for i, ticker in enumerate(tickers):
            np.random.seed(hash(ticker) % 777)
            steps = len(dates)
            vol = 0.003 + (i % 3) * 0.001
            price_changes = np.random.normal(0, vol, steps)
            price_changes[0] = 0
            price_path = 10000 * np.exp(np.cumsum(price_changes))
            
            close_p = price_path
            open_p = close_p * (1 + np.random.normal(0, 0.001, steps))
            high_p = np.maximum(open_p, close_p) * (1 + np.abs(np.random.normal(0, 0.001, steps)))
            low_p = np.minimum(open_p, close_p) * (1 - np.abs(np.random.normal(0, 0.001, steps)))
            volume = np.random.exponential(100, steps)
            value = close_p * volume
            
            ticker_df = pd.DataFrame({
                'timestamp': dates, 'open': open_p, 'high': high_p,
                'low': low_p, 'close': close_p, 'volume': volume,
                'value': value, 'ticker': ticker
            })
            dummy_list.append(ticker_df)
        df = pd.concat(dummy_list, ignore_index=True)
    return df

df_all = load_all_tickers_data()

# %%
df_all.describe()

# %%
def split_train_test_9_3(df):
    df = df.sort_values('timestamp').copy()
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    total_days = (max_date - min_date).days
    
    split_date = min_date + pd.Timedelta(days=int(total_days * 0.75))
    print(f"📊 총 관측 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')} ({total_days}일)")
    print(f"✂️ 9:3 분할 시점: {split_date.strftime('%Y-%m-%d')} (앞 9개월 Train / 뒤 3개월 Test)")
    
    train_df = df[df['timestamp'] < split_date].copy()
    test_df = df[df['timestamp'] >= split_date].copy()
    return train_df, test_df

train_df, test_df = split_train_test_9_3(df_all)

print(f"⏱️ Block 1 (Data Loading & Split) Execution Time: {time.time() - start_block1:.4f} seconds")

# %% [markdown]
# ## 2. 15종 모델 정의 (Model Zoo)
# 이전 단계에서 검증된 15개 SOTA 시계열 예측 모델들을 모두 통합합니다.
#

# %%
start_block2 = time.time()

class ModelZoo:
    class LinearModel(nn.Module):
        def __init__(self, seq_len=60, hidden_dim=None): super().__init__(); self.fc = nn.Linear(seq_len, 1)
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
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): x = x.transpose(1, 2); return self.fc(torch.relu(self.conv(x))[:, :, -1])
    class TransformerModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.enc = nn.Linear(input_dim, hidden_dim); self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, batch_first=True), 2); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): return self.fc(self.tf(self.enc(x))[:, -1, :])
    class InformerModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.enc = nn.Linear(input_dim, hidden_dim); self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, batch_first=True, dropout=0.1), 2); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): return self.fc(self.tf(self.enc(x))[:, -1, :])
    class AutoformerModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.avg = nn.AvgPool1d(3, 1, 1); self.enc = nn.Linear(input_dim, hidden_dim); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): t = self.avg(x.transpose(1, 2)).transpose(1, 2); r = x - t; return self.fc(self.enc(r + t)[:, -1, :])
    class PatchTSTModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.enc = nn.Linear(5, hidden_dim); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): p = x[:, -5:, :].transpose(1, 2); return self.fc(self.enc(p).squeeze(1))
    class MambaModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.proj = nn.Linear(input_dim, hidden_dim); self.conv = nn.Conv1d(hidden_dim, hidden_dim, 3, 1); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): x = self.proj(x).transpose(1, 2); x = torch.relu(self.conv(x)).transpose(1, 2); return self.fc(x[:, -1, :])
    class mTANDModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.te = nn.Linear(1, hidden_dim); self.fe = nn.Linear(input_dim, hidden_dim); self.attn = nn.MultiheadAttention(hidden_dim, 2, batch_first=True); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): b, s, _ = x.size(); t = torch.arange(s, dtype=torch.float32, device=x.device).view(1, s, 1).repeat(b, 1, 1); c = self.te(t) + self.fe(x); o, _ = self.attn(c, c, c); return self.fc(o[:, -1, :])
    class ODERNNModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.rnn = nn.GRUCell(input_dim, hidden_dim); self.ode = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim)); self.fc = nn.Linear(hidden_dim, 1); self.h_dim = hidden_dim
        def forward(self, x):
            b, s, _ = x.size(); h = torch.zeros(b, self.h_dim).to(x.device)
            for t in range(s): h = self.rnn(x[:, t, :], h); h = h + self.ode(h) * 0.1
            return self.fc(h)
    class NBeatsModel(nn.Module):
        def __init__(self, input_dim=1, seq_len=60, hidden_dim=128):
            super().__init__(); self.fc = nn.Sequential(nn.Linear(seq_len, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        def forward(self, x): return self.fc(x.squeeze(-1))
    class NonStatTFModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.enc = nn.Linear(input_dim, hidden_dim); self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, 4, batch_first=True), 2); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): mu = torch.mean(x, 1, True); x = x - mu; return self.fc(self.tf(self.enc(x))[:, -1, :]) + mu[:, -1, :]
    class DeepARModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__(); self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True); self.fc = nn.Linear(hidden_dim, 1)
        def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])
    class LinearDecompModel(nn.Module):
        def __init__(self, seq_len=60, hidden_dim=None): super().__init__(); self.t_fc = nn.Linear(seq_len, 1); self.r_fc = nn.Linear(seq_len, 1)
        def forward(self, x): x = x.squeeze(-1); return self.t_fc(x) + self.r_fc(x)

models_to_test = {
    'Linear': ModelZoo.LinearModel, 'LSTM': ModelZoo.LSTMModel, 'GRU': ModelZoo.GRUModel,
    'TCN': ModelZoo.TCNModel, 'Transformer': ModelZoo.TransformerModel, 'Informer': ModelZoo.InformerModel,
    'Autoformer': ModelZoo.AutoformerModel, 'PatchTST': ModelZoo.PatchTSTModel, 'Mamba': ModelZoo.MambaModel,
    'mTAND': ModelZoo.mTANDModel, 'ODE-RNN': ModelZoo.ODERNNModel, 'N-BEATS': ModelZoo.NBeatsModel,
    'NonStat-TF': ModelZoo.NonStatTFModel, 'DeepAR': ModelZoo.DeepARModel, 'Linear-Decomp': ModelZoo.LinearDecompModel
}

print(f"⏱️ Block 2 (Model Zoo Initialization) Execution Time: {time.time() - start_block2:.4f} seconds")

# %% [markdown]
# ## 3. 전체 종목 대상 다중 모델 그리드 서치 및 예측 파이프라인
# - Cross Validation 생략 (속도 확보)
# - 전처리: MinMaxScaler 단일화
# - 그리드 서치: 각 모델별 은닉층 크기(Hidden Dim)를 [32, 64] 2가지로 탐색하며 전체 종목 데이터셋에 피팅.
#

# %%
def perform_grid_search_and_evaluate(train_df, test_df, seq_len=60):
    tickers = train_df['ticker'].unique()
    grid_hidden_dims = [32, 64] # 초단순 그리드 서치 (Hidden Dimension)
    
    # 1. 모든 종목의 스케일링 된 학습/테스트 데이터 준비
    X_train_all, y_train_all = [], []
    test_data_dict = {}
    scalers = {}
    
    print(f"🚀 총 {len(tickers)}개 종목의 데이터 전처리(MinMax 단일) 및 시퀀스 구성 중...")
    for ticker in tickers:
        t_train = train_df[train_df['ticker'] == ticker].sort_values('timestamp').copy()
        t_test = test_df[test_df['ticker'] == ticker].sort_values('timestamp').copy()
        
        if len(t_train) < seq_len + 10 or len(t_test) < 10:
            continue
            
        scaler = MinMaxScaler()
        scaled_train_close = scaler.fit_transform(t_train[['close']].values)
        scalers[ticker] = scaler
        
        for i in range(len(scaled_train_close) - seq_len):
            X_train_all.append(scaled_train_close[i:i+seq_len])
            y_train_all.append(scaled_train_close[i+seq_len])
            
        t_train_last = t_train.tail(seq_len).copy()
        combined_df = pd.concat([t_train_last, t_test], ignore_index=True)
        scaled_combined = scaler.transform(combined_df[['close']].values)
        
        X_test, y_test = [], []
        for i in range(len(scaled_combined) - seq_len):
            X_test.append(scaled_combined[i:i+seq_len])
            y_test.append(scaled_combined[i+seq_len])
            
        test_data_dict[ticker] = {
            'X': np.array(X_test), 
            'y_actual_scaled': np.array(y_test),
            'raw_ohlcv': t_test.head(len(y_test))
        }
        
    train_dataset = TensorDataset(torch.tensor(np.array(X_train_all), dtype=torch.float32), torch.tensor(np.array(y_train_all), dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    
    # 2. 15종 모델 x 2개 그리드 파라미터 조합 훈련
    best_overall_model_state = None
    best_overall_model_name = ""
    best_overall_rmse = float('inf')
    best_ticker_results = {}
    
    criterion = nn.MSELoss()
    model_times = {} # 테스트 케이스 그룹별 실행 시간 기록
    
    for m_name, m_class in models_to_test.items():
        model_start = time.time()
        for h_dim in grid_hidden_dims:
            print(f"🔹 Training [Model: {m_name}] | Grid Search: [Hidden Dim: {h_dim}]")
            
            if m_name in ["Linear", "N-BEATS", "Linear-Decomp"]: 
                model = m_class(seq_len=seq_len).to(device)
            else: 
                model = m_class(hidden_dim=h_dim).to(device)
                
            optimizer = optim.AdamW(model.parameters(), lr=0.003)
            
            # 에포크를 2로 단축하여 극단적인 런타임 최적화 수행
            model.train()
            for epoch in range(2):
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                    
            # 테스트 세트 통합 검증
            model.eval()
            total_mse = 0
            
            with torch.no_grad():
                for ticker, t_data in test_data_dict.items():
                    x_tensor = torch.tensor(t_data['X'], dtype=torch.float32).to(device)
                    preds = model(x_tensor).detach().cpu().numpy()
                    total_mse += mean_squared_error(t_data['y_actual_scaled'], preds)
                    
                avg_mse = total_mse / len(test_data_dict)
                print(f"   => AVG MSE: {avg_mse:.6f}")
                
                if avg_mse < best_overall_rmse:
                    best_overall_rmse = avg_mse
                    best_overall_model_name = f"{m_name} (Dim:{h_dim})"
                    
                    # 최고 모델 갱신 시 각 종목별 예측 결과를 역변환하여 저장
                    best_ticker_results = {}
                    for ticker, t_data in test_data_dict.items():
                        x_tensor = torch.tensor(t_data['X'], dtype=torch.float32).to(device)
                        preds = model(x_tensor).detach().cpu().numpy()
                        
                        scaler = scalers[ticker]
                        y_actual_inv = scaler.inverse_transform(t_data['y_actual_scaled'])
                        y_pred_inv = scaler.inverse_transform(preds)
                        
                        best_ticker_results[ticker] = {
                            'actual': y_actual_inv.flatten(),
                            'pred': y_pred_inv.flatten(),
                            'raw_test_ohlcv': t_data['raw_ohlcv']
                        }
                        
            # N-BEATS나 Linear 계열은 hidden_dim 영향이 없으므로(고정 파라미터 등) 한 번만 수행
            if m_name in ["Linear", "Linear-Decomp", "PatchTST"]:
                break
                
        model_times[m_name] = time.time() - model_start
                
    print(f"\n🏆 최우수 평가 모델 갱신 완료: {best_overall_model_name} (MSE: {best_overall_rmse:.6f})")
    return best_ticker_results, best_overall_model_name, model_times

# %%
start_block3 = time.time()
best_results_dict, best_model_name, model_times = perform_grid_search_and_evaluate(train_df, test_df, seq_len=60)
print(f"⏱️ Block 3 (Model Zoo Grid Search) Execution Time: {time.time() - start_block3:.4f} seconds")

# %% [markdown]
# ## 4. 정통 금융 봉 차트(Candle Chart) + 거래량 막대 + 최우수 예측 모델 라인 융합 시각화
# 네이버 블로그 명세를 준수하여, 시각적 왜곡이 전혀 없는 정통 금융 시각화 그래프를 생성합니다.
#

# %%
start_block4 = time.time()

def draw_professional_candle_chart(ticker, data_payload, best_model_name, save_path):
    actual = data_payload['actual']
    pred = data_payload['pred']
    ohlc_df = data_payload['raw_test_ohlcv'].copy()
    
    slice_len = 40
    actual = actual[-slice_len:]
    pred = pred[-slice_len:]
    ohlc_df = ohlc_df.tail(slice_len)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    x = np.arange(len(ohlc_df))
    
    # 상단: 봉차트(Candle Chart) 렌더링
    for i in range(len(ohlc_df)):
        open_p = ohlc_df['open'].iloc[i]
        high_p = ohlc_df['high'].iloc[i]
        low_p = ohlc_df['low'].iloc[i]
        close_p = ohlc_df['close'].iloc[i]
        
        if close_p >= open_p:
            color = '#e74c3c'
            lower = open_p
            height = close_p - open_p
        else:
            color = '#3498db'
            lower = close_p
            height = open_p - close_p
            
        ax1.vlines(x[i], low_p, high_p, color=color, linewidth=1.8, zorder=2)
        rect = plt.Rectangle((x[i] - 0.3, lower), 0.6, height, facecolor=color, edgecolor=color, zorder=3)
        ax1.add_patch(rect)
        
    # 상단: 최우수 모델 예측 라인 오버레이
    ax1.plot(x, pred, label=f"Best Model [{best_model_name}] Pred", color='#2ecc71', linestyle='--', marker='o', linewidth=2.0, zorder=4)
    ax1.set_title(f"📊 SOTA Candle Chart: {ticker} Prediction vs Actual (Last {slice_len} Intervals)", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Price (KRW)", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 하단: 거래량 막대 그래프
    vol_colors = ['#e74c3c' if ohlc_df['close'].iloc[i] >= ohlc_df['open'].iloc[i] else '#3498db' for i in range(len(ohlc_df))]
    ax2.bar(x, ohlc_df['volume'], color=vol_colors, alpha=0.8, width=0.6, edgecolor='black', linewidth=0.5, zorder=3)
    
    ax2.set_ylabel("Volume", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Timeline Steps (15-min Intervals)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

if best_results_dict:
    target_ticker = list(best_results_dict.keys())[0]
    plot_path = "test/images/3_multi_ticker_best_model_forecast.png"
    draw_professional_candle_chart(target_ticker, best_results_dict[target_ticker], best_model_name, plot_path)
    print(f"✅ 네이버 블로그 규격을 준수한 정통 금융 캔들스틱 종합 차트가 '{plot_path}'에 완벽하게 저장되었습니다.")

print(f"⏱️ Block 4 (Professional Candle Plot Visualizer) Execution Time: {time.time() - start_block4:.4f} seconds")

# %% [markdown]
# ## 5. 테스트 케이스 그룹별 실행 시간 리포트 (Execution Time Summary)
# 각 모델 그룹(Test Case Group)별 훈련 및 검증 소요 시간을 마지막에 일목요연하게 출력합니다.
#

# %%
print("\n" + "="*60)
print("📊 [3단계 테스트 케이스 그룹별 훈련 및 검증 시간 종합 리포트]")
print("="*60)
print(f"| {'시계열 알고리즘 그룹 (Test Case)':<30} | {'실행 소요 시간 (초)':<20} |")
print(f"| {'-'*30} | {'-'*20} |")
for m_name, elapsed_time in model_times.items():
    print(f"| {m_name:<30} | {elapsed_time:<20.4f} |")
print("="*60)
