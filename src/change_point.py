"""
Change-point detection para scores de anomalia.

Implementa CUSUM (Cumulative Sum Control Chart) para detectar
mudanças de regime nos scores, complementando EMA + limiares fixos.
"""

import numpy as np


class CUSUM:
    """
    Detector de change-point via CUSUM bilateral.

    Detecta mudanças positivas (aumento de anomalia) e negativas
    na série de scores. Útil para:
    - Identificar início de ransomware mesmo com evasão por delay
    - Filtrar flutuações normais de workload
    - Detectar "mudança de regime" vs ruído pontual
    """

    def __init__(self, drift: float = 0.5, threshold: float = 5.0):
        """
        Args:
            drift: desvio mínimo do esperado para acumular (slack parameter).
                   Valores menores = mais sensível, mais FPs.
            threshold: limiar acumulado para disparar alarme.
                   Valores menores = detecção mais rápida, mais FPs.
        """
        self.drift = drift
        self.threshold = threshold

    def detect(self, scores: np.ndarray,
               reference: float | None = None) -> dict:
        """
        Executa CUSUM bilateral na série de scores.

        Args:
            scores: (N,) série temporal de scores
            reference: valor de referência (média esperada em benigno).
                       Se None, usa a média dos primeiros 20% dos scores.

        Returns:
            dict com:
                - g_pos: (N,) acumulador positivo
                - g_neg: (N,) acumulador negativo
                - alarms_pos: (N,) bool, alarmes de aumento
                - alarms_neg: (N,) bool, alarmes de queda
                - change_points: lista de índices onde houve alarme positivo
                - first_alarm: índice do primeiro alarme positivo (-1 se nenhum)
        """
        n = len(scores)

        if reference is None:
            n_ref = max(1, int(n * 0.2))
            reference = scores[:n_ref].mean()

        g_pos = np.zeros(n)
        g_neg = np.zeros(n)
        alarms_pos = np.zeros(n, dtype=bool)
        alarms_neg = np.zeros(n, dtype=bool)

        for i in range(1, n):
            # Acumulador positivo: detecta aumento
            g_pos[i] = max(0, g_pos[i-1] + (scores[i] - reference) - self.drift)
            # Acumulador negativo: detecta queda
            g_neg[i] = max(0, g_neg[i-1] - (scores[i] - reference) - self.drift)

            alarms_pos[i] = g_pos[i] > self.threshold
            alarms_neg[i] = g_neg[i] > self.threshold

            # Reset após alarme (one-shot reset)
            if alarms_pos[i]:
                g_pos[i] = 0
            if alarms_neg[i]:
                g_neg[i] = 0

        change_points = np.where(alarms_pos)[0].tolist()
        first_alarm = change_points[0] if change_points else -1

        return {
            "g_pos": g_pos,
            "g_neg": g_neg,
            "alarms_pos": alarms_pos,
            "alarms_neg": alarms_neg,
            "change_points": change_points,
            "first_alarm": first_alarm,
        }

    def detect_with_reference(self, scores: np.ndarray,
                              benign_scores: np.ndarray) -> dict:
        """
        CUSUM usando média e std do benigno como referência.
        Normaliza os scores antes de aplicar CUSUM.
        """
        ref_mean = benign_scores.mean()
        ref_std = benign_scores.std() + 1e-8

        # Normalizar scores pelo benigno
        normalized = (scores - ref_mean) / ref_std

        return self.detect(normalized, reference=0.0)


def fuse_multiresolution_scores(scores_dict: dict[int, np.ndarray],
                                method: str = "max") -> np.ndarray:
    """
    Funde scores de múltiplas resoluções temporais.

    Args:
        scores_dict: {resolução_ms: scores (N_i,)}
        method: "max" | "weighted_avg" | "vote"

    Returns:
        fused: scores fundidos (tamanho do menor array)
    """
    resolutions = sorted(scores_dict.keys())
    min_len = min(len(scores_dict[r]) for r in resolutions)

    # Alinhar tamanhos (truncar ao menor)
    aligned = np.stack([scores_dict[r][:min_len] for r in resolutions])

    if method == "max":
        return aligned.max(axis=0)
    elif method == "weighted_avg":
        # Peso inversamente proporcional à resolução (fina = mais peso)
        weights = np.array([1.0 / r for r in resolutions])
        weights = weights / weights.sum()
        return (aligned * weights[:, None]).sum(axis=0)
    elif method == "vote":
        # Cada resolução "vota" se score > mediana daquela resolução
        medians = np.median(aligned, axis=1, keepdims=True)
        votes = (aligned > medians).astype(float)
        return votes.mean(axis=0)
    else:
        raise ValueError(f"Método de fusão desconhecido: {method}")
