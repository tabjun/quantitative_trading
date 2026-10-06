"""27번 S-Mamba 러너(격리 venv `.venvs/smamba_py310_20261006`에서 실행).

저자 공식 코드(`third_party/S-D-Mamba`, 커밋 e7e8bf0)를 그대로 불러 쓴다. 이 환경은 저자 요구사항(`torch==2.0.1`,
`mamba-ssm==1.2.0`)에 맞춰 미리 컴파일된 wheel로 구성했다(`nvcc`·`sudo` 불필요, 시스템 변경 없음).

입력 npz: y(표준화 로그 RV, LOCF), ret(표준화 블록 수익률), ok(타깃 유효 여부), train_end, end, L, seed, quick.
출력 npz: p([train_end,end) 구간 예측, 표준화 단위). 학습 구간 [0,train_end)의 끝 15%는 조기종료 검증에 쓴다. 호출하는 쪽이
neuralforecast·ModernTCN과 같은 두 단계(내부학습→내부검증 구간, 전체 재적합→평가 구간)로 두 번 부른다. 설정은 저자의 ETTh1 스크립트(e_layers 2, d_model 256,
d_state 2, d_ff 256, 학습률 7e-5, 에폭 10, 인내 3, lradj type1)를 따른다. 변수는 2개(log RV, 부호 있는 블록 수익률)이고
iTransformer와 같은 입력이다. 손실은 두 변수의 MSE이며 log RV 변수의 예측만 쓴다.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "S-D-Mamba"))
VAL_FRAC, BATCH, LR, EPOCHS, PATIENCE = 0.15, 32, 7e-5, 10, 3


def fit_predict(y, ret, ok, train_end, end, L, seed, dev, quick):
    import torch
    import torch.nn as nn
    from model.S_Mamba import Model
    cfg = types.SimpleNamespace(seq_len=L, pred_len=1, output_attention=False, use_norm=1, d_model=256, embed="timeF",
                                freq="h", dropout=0.1, class_strategy="projection", e_layers=2, d_state=2, d_ff=256,
                                activation="gelu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(cfg).to(dev)
    Z = np.stack([y, ret], axis=1).astype(np.float32)                      # (nb, 2)
    val_n = int(VAL_FRAC * train_end)
    fit_end = train_end - val_n
    tr = np.array([e for e in range(L, fit_end) if ok[e]])
    va = np.array([e for e in range(fit_end, train_end) if ok[e]])
    mk = lambda idx: (torch.tensor(np.stack([Z[e - L:e] for e in idx])).to(dev), torch.tensor(Z[idx]).to(dev))
    Xtr, Ytr = mk(tr)
    Xva, Yva = mk(va)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()

    def infer(x):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 1024):
                out.append(model(x[i:i + 1024], None, None, None)[:, 0, :].float())
        return torch.cat(out)

    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS if not quick else 1):
        for g in opt.param_groups:                                          # 저자 lradj type1: 에폭마다 0.5배
            g["lr"] = LR * (0.5 ** ep)
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[b], None, None, None)[:, 0, :], Ytr[b])
            loss.backward()
            opt.step()
        vl = float(lossf(infer(Xva), Yva).item())
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    te = np.arange(train_end, end)
    X = torch.tensor(np.stack([Z[e - L:e] for e in te])).to(dev)
    return infer(X)[:, 0].cpu().numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.inp)
    y, ret, ok = z["y"], z["ret"], z["ok"].astype(bool)
    te, en, L, seed, quick = int(z["train_end"]), int(z["end"]), int(z["L"]), int(z["seed"]), bool(z["quick"])
    p = fit_predict(y, ret, ok, te, en, L, seed, dev, quick)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("S-Mamba: 예측에 비유한값")
    np.savez_compressed(a.out, p=p)
    print(f"[S-Mamba] {len(p)}건", flush=True)


if __name__ == "__main__":
    main()
