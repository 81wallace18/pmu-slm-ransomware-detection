"""
Baseline LSTM Forecaster (estilo HARD-Lite).

LSTM que prevê o próximo frame dado uma janela de N-1 frames.
Score de anomalia = erro L1 por feature + WMVE (AUC-weighted) + EMA + 2 níveis.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class LSTMForecaster(nn.Module):
    """LSTM que prevê o próximo vetor de PMU dado uma sequência."""

    def __init__(self, n_features: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_features) — sequência de entrada
        Returns:
            pred: (B, n_features) — previsão do próximo frame
        """
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred


class HARDLiteDetector:
    """
    Detector estilo HARD-Lite: LSTM forecasting + WMVE + EMA + 2 níveis.

    Fluxo fiel ao paper:
    1. Para cada janela, prevê o próximo frame via LSTM
    2. Calcula erro L1 por feature
    3. Compara com limiar por feature (percentil do benigno)
    4. WMVE: voto ponderado por AUC-ROC de cada feature
    5. EMA: suavização temporal do score
    6. Dois níveis de alerta: warning (sensível) e anomaly (específico)
    """

    def __init__(self, model: LSTMForecaster, device: torch.device,
                 ema_alpha: float = 0.3):
        self.model = model
        self.device = device
        self.ema_alpha = ema_alpha

        self.n_features = model.n_features
        self.thresholds_warning: np.ndarray | None = None
        self.thresholds_anomaly: np.ndarray | None = None
        self.feature_weights: np.ndarray | None = None  # AUC-ROC por feature

    def compute_errors(self, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Calcula erro L1 por feature para cada janela.

        Args:
            windows: (n_windows, window_size, n_features)

        Returns:
            errors: (n_windows, n_features)
        """
        self.model.eval()
        errors = []

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[i:i + batch_size]).float().to(self.device)
                input_seq = batch[:, :-1, :]
                target = batch[:, -1, :]
                pred = self.model(input_seq)
                err = torch.abs(pred - target).cpu().numpy()
                errors.append(err)

        return np.concatenate(errors, axis=0)

    def fit(self, benign_windows: np.ndarray, mal_windows: np.ndarray | None = None,
            warning_percentile: float = 90, anomaly_percentile: float = 99):
        """
        Calibra limiares e pesos WMVE.

        Args:
            benign_windows: janelas benignas para calibração
            mal_windows: janelas maliciosas (se disponíveis) para calcular AUC por feature
            warning_percentile: percentil para limiar warning
            anomaly_percentile: percentil para limiar anomaly
        """
        benign_errors = self.compute_errors(benign_windows)

        # Limiares de 2 níveis por feature
        self.thresholds_warning = np.percentile(benign_errors, warning_percentile, axis=0)
        self.thresholds_anomaly = np.percentile(benign_errors, anomaly_percentile, axis=0)

        print(f"Limiares warning (p{warning_percentile}): {self.thresholds_warning}")
        print(f"Limiares anomaly (p{anomaly_percentile}): {self.thresholds_anomaly}")

        # Pesos WMVE: AUC-ROC por feature (se temos dados maliciosos)
        if mal_windows is not None and len(mal_windows) > 0:
            mal_errors = self.compute_errors(mal_windows)
            self.feature_weights = self._compute_auc_weights(benign_errors, mal_errors)
            print(f"Pesos WMVE (AUC): {self.feature_weights}")
        else:
            # Sem malicioso: pesos uniformes (fallback)
            self.feature_weights = np.ones(self.n_features) / self.n_features
            print("Pesos WMVE: uniformes (sem dados maliciosos para AUC)")

    def _compute_auc_weights(self, benign_errors: np.ndarray,
                             mal_errors: np.ndarray) -> np.ndarray:
        """Calcula peso de cada feature baseado no AUC-ROC individual."""
        weights = np.zeros(self.n_features)

        for f in range(self.n_features):
            labels = np.concatenate([
                np.zeros(len(benign_errors)),
                np.ones(len(mal_errors)),
            ])
            scores = np.concatenate([benign_errors[:, f], mal_errors[:, f]])
            try:
                weights[f] = roc_auc_score(labels, scores)
            except ValueError:
                weights[f] = 0.5

        # Normalizar para somar 1
        weights = np.maximum(weights - 0.5, 0)  # só features acima de random
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(self.n_features) / self.n_features

        return weights

    def score_windows_raw(self, windows: np.ndarray,
                          level: str = "warning") -> np.ndarray:
        """
        Score WMVE por janela (sem EMA).

        Args:
            windows: (n_windows, window_size, n_features)
            level: "warning" ou "anomaly"

        Returns:
            scores: (n_windows,) score agregado
        """
        errors = self.compute_errors(windows)
        thresholds = (self.thresholds_warning if level == "warning"
                      else self.thresholds_anomaly)

        votes = (errors > thresholds).astype(np.float32)
        scores = (votes * self.feature_weights).sum(axis=1)
        return scores

    def score_windows(self, windows: np.ndarray,
                      level: str = "warning") -> np.ndarray:
        """Score WMVE + EMA por janela."""
        raw = self.score_windows_raw(windows, level)
        return self._apply_ema(raw)

    def _apply_ema(self, scores: np.ndarray) -> np.ndarray:
        """Aplica suavização EMA."""
        ema = np.zeros_like(scores)
        ema[0] = scores[0]
        for i in range(1, len(scores)):
            ema[i] = self.ema_alpha * scores[i] + (1 - self.ema_alpha) * ema[i - 1]
        return ema

    def detect(self, windows: np.ndarray) -> dict:
        """
        Detecção completa com 2 níveis.

        Returns:
            dict com:
                - scores_warning: scores EMA nível warning
                - scores_anomaly: scores EMA nível anomaly
                - warnings: bool array (janelas em warning)
                - anomalies: bool array (janelas em anomaly)
                - warning_threshold: limiar WMVE para warning
                - anomaly_threshold: limiar WMVE para anomaly
        """
        scores_w = self.score_windows(windows, level="warning")
        scores_a = self.score_windows(windows, level="anomaly")

        # Limiar no score WMVE agregado: >0.5 = maioria das features votou
        w_thresh = 0.3   # warning: mais sensível
        a_thresh = 0.5   # anomaly: maioria

        return {
            "scores_warning": scores_w,
            "scores_anomaly": scores_a,
            "warnings": scores_w > w_thresh,
            "anomalies": scores_a > a_thresh,
            "warning_threshold": w_thresh,
            "anomaly_threshold": a_thresh,
        }


def create_lstm_windows(continuous_data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Cria janelas para o LSTM a partir de dados contínuos.

    Args:
        continuous_data: (n_frames, n_features)
        window_size: tamanho da janela

    Returns:
        windows: (n_windows, window_size, n_features)
    """
    n = len(continuous_data)
    windows = []
    for i in range(n - window_size + 1):
        windows.append(continuous_data[i:i + window_size])
    return np.array(windows, dtype=np.float32)
