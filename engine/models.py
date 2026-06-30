"""모델군, 옵티마이저/스케줄러/그래디언트 정책, 시드, window 정규화.

8번에서 그대로 옮겼다. `make_loader`/`make_optimizer`/`make_scheduler`/
`apply_gradient_policy`(+`grad_norm`), `set_seed`, `apply_window_normalization`,
그리고 모델 클래스 전체와 `make_model`을 담는다.

`make_loader` 등의 `ResourceProfile` 타입 힌트는 `from __future__ import annotations`
덕분에 import 시 평가되지 않는다. resources 모듈이 이 모듈을 import하므로,
순환 import를 막기 위해 여기서는 resources를 import하지 않는다.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_window_normalization(window: np.ndarray, mode: str) -> np.ndarray:
    """정규화는 단순 standard뿐 아니라 robust/reversible/frequency 계열도 비교한다."""

    if mode == "none" or mode == "global_standard":
        return window
    if mode in {"window_standard", "revin"}:
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        return (window - mean) / np.where(std < 1e-6, 1.0, std)
    if mode in {"window_robust", "robust_revin"}:
        median = np.nanmedian(window, axis=0, keepdims=True)
        iqr = np.nanquantile(window, 0.75, axis=0, keepdims=True) - np.nanquantile(window, 0.25, axis=0, keepdims=True)
        return (window - median) / np.where(iqr < 1e-6, 1.0, iqr)
    if mode == "window_minmax":
        lo = np.nanmin(window, axis=0, keepdims=True)
        hi = np.nanmax(window, axis=0, keepdims=True)
        return 2.0 * (window - lo) / np.where((hi - lo) < 1e-6, 1.0, hi - lo) - 1.0
    if mode == "asinh_revin":
        median = np.nanmedian(window, axis=0, keepdims=True)
        iqr = np.nanquantile(window, 0.75, axis=0, keepdims=True) - np.nanquantile(window, 0.25, axis=0, keepdims=True)
        return np.arcsinh((window - median) / np.where(iqr < 1e-6, 1.0, iqr))
    raise ValueError(f"지원하지 않는 normalization: {mode}")


class LinearForecaster(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RecurrentForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int, kind: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, levels: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for level in range(levels):
            dilation = 2**level
            layers.extend(
                [
                    nn.Conv1d(in_ch, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.ReLU(),
                ]
            )
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))[..., : x.shape[1]]
        return self.head(z[:, :, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos[:, : x.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1]).squeeze(-1)


class DLinearLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int):
        super().__init__()
        self.seasonal = nn.Linear(seq_len, 1)
        self.trend = nn.Linear(seq_len, 1)
        self.feature_head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = F.avg_pool1d(x.transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
        seasonal = x - trend
        y = self.seasonal(seasonal.transpose(1, 2)).squeeze(-1)
        t = self.trend(trend.transpose(1, 2)).squeeze(-1)
        return self.feature_head(y + t).squeeze(-1)


class NLinearLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.feature_head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, -1:, :]
        z = x - base
        y = self.linear(z.transpose(1, 2)).squeeze(-1)
        return self.feature_head(y).squeeze(-1)


class PatchTSTLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, patch_len: int = 8, heads: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = max(1, seq_len // patch_len)
        self.proj = nn.Linear(n_features * patch_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        usable = self.n_patches * self.patch_len
        z = x[:, -usable:].reshape(b, self.n_patches, f * self.patch_len)
        z = self.encoder(self.proj(z))
        return self.head(z.mean(dim=1)).squeeze(-1)


class AutoformerLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.season_proj = nn.Linear(n_features, hidden)
        self.trend_head = nn.Linear(n_features, 1)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = F.avg_pool1d(x.transpose(1, 2), kernel_size=9, stride=1, padding=4).transpose(1, 2)
        seasonal = x - trend
        z = self.encoder(self.season_proj(seasonal))
        return self.head(z.mean(dim=1)).squeeze(-1) + self.trend_head(trend[:, -1]).squeeze(-1)


class ITransformerLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.var_proj = nn.Linear(seq_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.encoder(self.var_proj(z))
        return self.head(z.mean(dim=1)).squeeze(-1)


class ModernTCNLike(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96):
        super().__init__()
        self.in_proj = nn.Conv1d(n_features, hidden, kernel_size=1)
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x.transpose(1, 2))
        z = self.depthwise(z)
        return self.head(z[:, :, -1]).squeeze(-1)


class MambaLike(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden * 2)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, groups=hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        z = value * torch.sigmoid(gate)
        z = self.conv(z.transpose(1, 2))[..., : x.shape[1]]
        return self.head(z[:, :, -1]).squeeze(-1)


class TimesNetLike(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96):
        super().__init__()
        self.conv3 = nn.Conv1d(n_features, hidden, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(n_features, hidden, kernel_size=7, padding=3)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        a = self.conv3(z).mean(dim=-1)
        b = self.conv7(z).mean(dim=-1)
        return self.head(torch.cat([a, b], dim=-1)).squeeze(-1)


class TimeXerLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.temporal = TransformerForecaster(seq_len, n_features, hidden, heads)
        self.exog_gate = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.temporal(x)
        gate = self.exog_gate(x[:, -1]).squeeze(-1)
        return base * (1.0 + 0.1 * gate)


def make_model(name: str, seq_len: int, n_features: int, hidden: int) -> nn.Module:
    factories: dict[str, Callable[[], nn.Module]] = {
        "Linear": lambda: LinearForecaster(seq_len, n_features, hidden),
        "LSTM": lambda: RecurrentForecaster(n_features, hidden, "lstm"),
        "GRU": lambda: RecurrentForecaster(n_features, hidden, "gru"),
        "TCN": lambda: TCNForecaster(n_features, hidden),
        "Transformer": lambda: TransformerForecaster(seq_len, n_features, hidden),
        "DLinearLike": lambda: DLinearLike(seq_len, n_features),
        "NLinearLike": lambda: NLinearLike(seq_len, n_features),
        "PatchTSTLike": lambda: PatchTSTLike(seq_len, n_features, hidden),
        "AutoformerLike": lambda: AutoformerLike(seq_len, n_features, hidden),
        "ITransformerLike": lambda: ITransformerLike(seq_len, n_features, hidden),
        "ModernTCNLike": lambda: ModernTCNLike(n_features, hidden),
        "MambaLike": lambda: MambaLike(n_features, hidden),
        "TimesNetLike": lambda: TimesNetLike(n_features, hidden),
        "TimeXerLike": lambda: TimeXerLike(seq_len, n_features, hidden),
    }
    if name not in factories:
        raise KeyError(f"알 수 없는 모델명: {name}")
    return factories[name]()


def make_loader(split: dict[str, np.ndarray], batch_size: int, shuffle: bool, profile: "ResourceProfile") -> DataLoader:
    ds = TensorDataset(torch.from_numpy(split["x"]), torch.from_numpy(split["y"]))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        drop_last=False,
    )


def make_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"지원하지 않는 optimizer: {optimizer_name}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
    lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=max(1, epochs),
            steps_per_epoch=max(1, steps_per_epoch),
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    raise ValueError(f"지원하지 않는 scheduler: {scheduler_name}")


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().data.norm(2).item()
        total += value * value
    return float(math.sqrt(total))


def apply_gradient_policy(model: nn.Module, policy: str, epoch: int) -> float:
    norm_before = grad_norm(model)
    if policy == "none":
        return norm_before
    if policy == "clip0.5":
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    elif policy == "clip1":
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    elif policy == "clip5":
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    elif policy == "adaptive":
        threshold = min(5.0, max(0.5, 0.5 + 0.1 * epoch))
        nn.utils.clip_grad_norm_(model.parameters(), threshold)
    else:
        raise ValueError(f"지원하지 않는 gradient policy: {policy}")
    return norm_before
