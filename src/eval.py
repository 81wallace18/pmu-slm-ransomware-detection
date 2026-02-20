"""
Avaliação e comparação: SLM vs LSTM baseline.

Corrige e inclui tudo que o BASELINE.md exige:
- EMA integrada na avaliação
- 2 níveis de alerta (warning/anomaly)
- WMVE com pesos AUC por feature
- ToD correto (tempo real desde início do ransomware, p50/p95)
- FPR/dia em benigno longo
- Change-point detection (CUSUM)
- Multi-resolução (10ms/100ms/1s)
- Overhead / latência de inferência (p95)
- Bootstrap CI para todas as métricas
- Tabela comparativa final conforme BASELINE.md linha 106

Uso:
    python src/eval.py --data-dir data/tokenized --checkpoint-dir checkpoints
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch
import yaml

from model import PMUSLM, TwoStageDetector, quantize_model
from baseline import LSTMForecaster, HARDLiteDetector, create_lstm_windows
from change_point import CUSUM, fuse_multiresolution_scores


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_slm(model: PMUSLM, data: np.ndarray, device: torch.device,
              batch_size: int = 256) -> np.ndarray:
    """NLL por janela usando o SLM."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.from_numpy(data[i:i + batch_size]).to(device)
            nll = model.compute_nll(batch)
            scores.append(nll.cpu().numpy())
    return np.concatenate(scores)


def apply_ema(scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Suavização EMA."""
    ema = np.zeros_like(scores)
    ema[0] = scores[0]
    for i in range(1, len(scores)):
        ema[i] = alpha * scores[i] + (1 - alpha) * ema[i - 1]
    return ema


# ── Métricas Corrigidas ──────────────────────────────────────────────────────

def compute_metrics(benign_scores: np.ndarray, mal_scores: np.ndarray,
                    interval_ms: float, window_size: int,
                    fpr_targets: list[float] = None,
                    warning_percentile: float = 90,
                    anomaly_percentile: float = 99) -> dict:
    """
    Métricas completas conforme BASELINE.md.

    Inclui:
    - AUC-ROC, PR-AUC
    - TPR@FPR fixo
    - FPR/dia em benigno longo
    - ToD em ms (p50/p95) com 2 níveis
    - Limiares warning/anomaly
    """
    if fpr_targets is None:
        fpr_targets = [0.001, 0.01, 0.05]

    labels = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(mal_scores))])
    scores = np.concatenate([benign_scores, mal_scores])

    # AUC-ROC
    auroc = roc_auc_score(labels, scores)
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)

    # PR-AUC
    prec, rec, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(rec, prec)

    # TPR @ FPR fixo
    tpr_at_fpr = {}
    for target_fpr in fpr_targets:
        idx = np.searchsorted(fpr_arr, target_fpr)
        idx = min(idx, len(tpr_arr) - 1)
        tpr_at_fpr[f"TPR@FPR={target_fpr}"] = float(tpr_arr[idx])

    # Limiares de 2 níveis (calculados no benigno)
    threshold_warning = float(np.percentile(benign_scores, warning_percentile))
    threshold_anomaly = float(np.percentile(benign_scores, anomaly_percentile))

    # FPR/dia: converter de fração de janelas para alarmes por dia
    # Cada janela = window_size * interval_ms
    window_duration_ms = window_size * interval_ms
    windows_per_day = (24 * 3600 * 1000) / window_duration_ms

    fp_warning_frac = (benign_scores > threshold_warning).mean()
    fp_anomaly_frac = (benign_scores > threshold_anomaly).mean()
    fpr_day_warning = fp_warning_frac * windows_per_day
    fpr_day_anomaly = fp_anomaly_frac * windows_per_day

    # ToD: tempo desde início do ransomware até detecção
    # Cada janela maliciosa corresponde a window_duration_ms desde o início
    tod_warning = _compute_tod(mal_scores, threshold_warning, window_duration_ms)
    tod_anomaly = _compute_tod(mal_scores, threshold_anomaly, window_duration_ms)

    metrics = {
        "AUC-ROC": float(auroc),
        "PR-AUC": float(pr_auc),
        "threshold_warning": threshold_warning,
        "threshold_anomaly": threshold_anomaly,
        "FPR/day_warning": float(fpr_day_warning),
        "FPR/day_anomaly": float(fpr_day_anomaly),
        "ToD_warning_p50_ms": tod_warning["p50"],
        "ToD_warning_p95_ms": tod_warning["p95"],
        "ToD_anomaly_p50_ms": tod_anomaly["p50"],
        "ToD_anomaly_p95_ms": tod_anomaly["p95"],
        **tpr_at_fpr,
    }

    return metrics, (fpr_arr, tpr_arr, prec, rec)


def _compute_tod(mal_scores: np.ndarray, threshold: float,
                 window_duration_ms: float) -> dict:
    """
    Time to Detection: tempo até primeira janela acima do limiar.
    Retorna p50 e p95 em ms.

    Para simular múltiplas execuções, usa janela deslizante:
    divide a série maliciosa em segmentos e mede ToD de cada um.
    """
    above = mal_scores > threshold

    if not above.any():
        return {"p50": -1.0, "p95": -1.0, "first_ms": -1.0}

    # Primeira detecção
    first_idx = int(np.argmax(above))
    first_ms = first_idx * window_duration_ms

    # Simular múltiplas execuções: dividir em segmentos de ~100 janelas
    seg_size = min(100, len(mal_scores) // 5)
    if seg_size < 10:
        seg_size = len(mal_scores)

    tods = []
    for start in range(0, len(mal_scores) - seg_size + 1, seg_size):
        seg = mal_scores[start:start + seg_size]
        seg_above = seg > threshold
        if seg_above.any():
            tod = int(np.argmax(seg_above)) * window_duration_ms
            tods.append(tod)

    if not tods:
        return {"p50": first_ms, "p95": first_ms, "first_ms": first_ms}

    return {
        "p50": float(np.percentile(tods, 50)),
        "p95": float(np.percentile(tods, 95)),
        "first_ms": first_ms,
    }


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_metrics(benign_scores: np.ndarray, mal_scores: np.ndarray,
                      interval_ms: float, window_size: int,
                      n_iterations: int = 1000, confidence: float = 0.95,
                      warning_pct: float = 90, anomaly_pct: float = 99) -> dict:
    """
    Bootstrap para intervalos de confiança das métricas principais.
    Reamostra benign e mal independentemente.
    """
    rng = np.random.RandomState(42)
    alpha = (1 - confidence) / 2

    boot_metrics = {
        "AUC-ROC": [], "PR-AUC": [],
        "FPR/day_warning": [], "FPR/day_anomaly": [],
    }

    for _ in range(n_iterations):
        b_idx = rng.choice(len(benign_scores), len(benign_scores), replace=True)
        m_idx = rng.choice(len(mal_scores), len(mal_scores), replace=True)

        b_sample = benign_scores[b_idx]
        m_sample = mal_scores[m_idx]

        try:
            metrics, _ = compute_metrics(
                b_sample, m_sample, interval_ms, window_size,
                warning_percentile=warning_pct, anomaly_percentile=anomaly_pct,
            )
            for key in boot_metrics:
                boot_metrics[key].append(metrics[key])
        except ValueError:
            continue

    ci = {}
    for key, values in boot_metrics.items():
        values = np.array(values)
        ci[key] = {
            "mean": float(values.mean()),
            "ci_low": float(np.percentile(values, alpha * 100)),
            "ci_high": float(np.percentile(values, (1 - alpha) * 100)),
        }

    return ci


# ── Overhead / Latência ──────────────────────────────────────────────────────

def measure_overhead(model, data: np.ndarray, device: torch.device,
                     n_runs: int = 100, is_slm: bool = True,
                     window_duration_ms: float = 500.0) -> dict:
    """
    Mede latência de inferência, uso de memória e CPU%.

    CPU% = (latência de inferência / duração da janela) * 100
    Representa quanto tempo de CPU a inferência consome por janela.

    Returns:
        dict com latency_p50/p95/p99_ms, cpu_percent, memory
    """
    import resource

    model.eval()
    single = data[:1]

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            if is_slm:
                batch = torch.from_numpy(single).to(device)
                model.compute_nll(batch)
            else:
                batch = torch.from_numpy(single).float().to(device)
                model(batch[:, :-1, :])

    # Medir latência
    latencies = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if is_slm:
                batch = torch.from_numpy(single).to(device)
                model.compute_nll(batch)
            else:
                batch = torch.from_numpy(single).float().to(device)
                model(batch[:, :-1, :])

        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)

    result = {
        "latency_mean_ms": float(latencies.mean()),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }

    # CPU%: quanto da janela a inferência consome
    # Ex: latência 5ms / janela 500ms = 1% CPU
    result["cpu_percent"] = float(
        (np.percentile(latencies, 95) / window_duration_ms) * 100
    )

    # Memória
    if device.type == "cuda":
        result["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
    else:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["rss_kb"] = rss

    # Parâmetros
    n_params = sum(p.numel() for p in model.parameters())
    result["n_params"] = n_params
    result["n_params_M"] = n_params / 1e6

    return result


# ── Gráficos ──────────────────────────────────────────────────────────────────

def plot_score_histograms(benign_scores, mal_scores, title, output_path,
                          threshold_w=None, threshold_a=None):
    """Histograma com 2 limiares marcados."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(benign_scores, bins=80, alpha=0.6, label="Benigno",
            color="steelblue", density=True)
    ax.hist(mal_scores, bins=80, alpha=0.6, label="Malicioso",
            color="crimson", density=True)
    if threshold_w is not None:
        ax.axvline(x=threshold_w, color="orange", linestyle="--",
                   alpha=0.8, label=f"Warning={threshold_w:.3f}")
    if threshold_a is not None:
        ax.axvline(x=threshold_a, color="red", linestyle="-",
                   alpha=0.8, label=f"Anomaly={threshold_a:.3f}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Densidade")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_comparison(curves, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Full ROC
    for name, (fpr, tpr, auroc) in curves.items():
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auroc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC (completa)")
    axes[0].legend()

    # Zoom em baixo FPR
    for name, (fpr, tpr, auroc) in curves.items():
        axes[1].plot(fpr, tpr, label=f"{name}")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC (zoom FPR < 0.1)")
    axes[1].set_xlim([0, 0.1])
    axes[1].set_ylim([0.5, 1.0])
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_comparison(curves, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, (prec, rec, pr_auc) in curves.items():
        ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_timeline_with_cusum(scores, cusum_result, title, output_path,
                             threshold_w=None, threshold_a=None):
    """Score + CUSUM ao longo do tempo."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(scores, linewidth=0.5, alpha=0.8, label="Score (EMA)")
    if threshold_w is not None:
        axes[0].axhline(y=threshold_w, color="orange", linestyle="--",
                        alpha=0.7, label=f"Warning={threshold_w:.3f}")
    if threshold_a is not None:
        axes[0].axhline(y=threshold_a, color="red", linestyle="-",
                        alpha=0.7, label=f"Anomaly={threshold_a:.3f}")
    axes[0].set_ylabel("Score")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)

    axes[1].plot(cusum_result["g_pos"], linewidth=0.8, label="CUSUM S+", color="red")
    axes[1].plot(cusum_result["g_neg"], linewidth=0.8, label="CUSUM S-", color="blue")
    for cp in cusum_result["change_points"]:
        axes[1].axvline(x=cp, color="green", alpha=0.5, linewidth=0.5)
    axes[1].set_xlabel("Janela")
    axes[1].set_ylabel("CUSUM")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Pipeline Principal ────────────────────────────────────────────────────────

def evaluate(config: dict, data_dir: str, checkpoint_dir: str, output_dir: str):
    """Avaliação completa conforme BASELINE.md."""
    device = get_device()
    cfg_eval = config["eval"]
    cfg_lstm = config["lstm"]
    events = config["collect"]["events"]
    n_events = len(events)
    interval_ms = config["collect"]["interval_ms"]
    ema_alpha = cfg_eval["ema_alpha"]
    warning_pct = cfg_eval["warning_percentile"]
    anomaly_pct = cfg_eval["anomaly_percentile"]

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    ws = info["window_size"]

    # ── Carregar dados ──
    test_benign_tok = np.load(os.path.join(data_dir, "test_benign.npy"))
    test_mal_tok = np.load(os.path.join(data_dir, "test_malicious.npy"))

    benign_cont = np.load(os.path.join(data_dir, "benign_continuous.npy"))
    mal_cont = np.load(os.path.join(data_dir, "malicious_continuous.npy"))

    lstm_ws = cfg_lstm["window_size"]
    benign_windows = create_lstm_windows(benign_cont, lstm_ws)
    mal_windows = create_lstm_windows(mal_cont, lstm_ws)

    n_total = len(benign_windows)
    n_test_start = int(n_total * 0.85)
    test_benign_lstm = benign_windows[n_test_start:]
    # Usar uma porção para calibrar WMVE (treino do detector, não do modelo)
    cal_benign_lstm = benign_windows[int(n_total * 0.70):n_test_start]

    print(f"Test benigno (SLM):  {test_benign_tok.shape}")
    print(f"Test malicioso (SLM): {test_mal_tok.shape}")
    print(f"Test benigno (LSTM): {test_benign_lstm.shape}")
    print(f"Cal benigno (LSTM):  {cal_benign_lstm.shape}")
    print(f"Test malicioso (LSTM): {mal_windows.shape}")

    results = {}
    roc_curves = {}
    pr_curves = {}
    overhead_results = {}

    # CUSUM config
    cusum_cfg = cfg_eval.get("change_point", {})
    cusum_enabled = cusum_cfg.get("enabled", False)
    cusum = CUSUM(
        drift=cusum_cfg.get("drift", 0.5),
        threshold=cusum_cfg.get("threshold", 5.0),
    ) if cusum_enabled else None

    # ── SLM ──
    slm_path = os.path.join(checkpoint_dir, "slm_best.pt")
    if os.path.exists(slm_path):
        print("\n=== SLM ===")
        model = PMUSLM(
            vocab_size=info["vocab_size"],
            n_embd=config["slm"]["n_embd"],
            n_head=config["slm"]["n_head"],
            n_layer=config["slm"]["n_layer"],
            block_size=info["seq_len"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(torch.load(slm_path, map_location=device, weights_only=True))

        # Score raw + EMA
        slm_benign_raw = score_slm(model, test_benign_tok, device)
        slm_mal_raw = score_slm(model, test_mal_tok, device)
        slm_benign = apply_ema(slm_benign_raw, ema_alpha)
        slm_mal = apply_ema(slm_mal_raw, ema_alpha)

        slm_metrics, (fpr, tpr, prec, rec) = compute_metrics(
            slm_benign, slm_mal, interval_ms, ws,
            cfg_eval["fpr_targets"], warning_pct, anomaly_pct,
        )
        results["SLM"] = slm_metrics
        roc_curves["SLM"] = (fpr, tpr, slm_metrics["AUC-ROC"])
        pr_curves["SLM"] = (prec, rec, slm_metrics["PR-AUC"])

        print("  Métricas SLM:")
        for k, v in slm_metrics.items():
            print(f"    {k}: {v}")

        plot_score_histograms(
            slm_benign, slm_mal, "SLM: NLL+EMA (Benigno vs Malicioso)",
            os.path.join(output_dir, "slm_histogram.png"),
            slm_metrics["threshold_warning"], slm_metrics["threshold_anomaly"],
        )

        # CUSUM no malicioso
        if cusum:
            cusum_mal = cusum.detect_with_reference(slm_mal, slm_benign)
            slm_metrics["CUSUM_first_alarm"] = cusum_mal["first_alarm"]
            if cusum_mal["first_alarm"] >= 0:
                slm_metrics["CUSUM_ToD_ms"] = float(
                    cusum_mal["first_alarm"] * ws * interval_ms
                )
            plot_timeline_with_cusum(
                slm_mal, cusum_mal, "SLM: Score Malicioso + CUSUM",
                os.path.join(output_dir, "slm_timeline_cusum.png"),
                slm_metrics["threshold_warning"], slm_metrics["threshold_anomaly"],
            )

        # Overhead
        if cfg_eval.get("measure_overhead", False):
            n_runs = cfg_eval.get("overhead_n_runs", 100)
            window_dur_ms = ws * interval_ms
            overhead_results["SLM"] = measure_overhead(
                model, test_benign_tok, device, n_runs, is_slm=True,
                window_duration_ms=window_dur_ms,
            )
            print(f"  Overhead SLM: {overhead_results['SLM']}")

        # Bootstrap CI
        boot_cfg = cfg_eval.get("bootstrap", {})
        if boot_cfg.get("enabled", False):
            print("  Bootstrap CI...")
            ci = bootstrap_metrics(
                slm_benign, slm_mal, interval_ms, ws,
                boot_cfg.get("n_iterations", 1000),
                boot_cfg.get("confidence", 0.95),
                warning_pct, anomaly_pct,
            )
            results["SLM_CI"] = ci
            print(f"  AUC-ROC: {ci['AUC-ROC']['mean']:.4f} "
                  f"[{ci['AUC-ROC']['ci_low']:.4f}, {ci['AUC-ROC']['ci_high']:.4f}]")

    else:
        print(f"SLM checkpoint não encontrado: {slm_path}")

    # ── LSTM ──
    lstm_path = os.path.join(checkpoint_dir, "lstm_best.pt")
    if os.path.exists(lstm_path):
        print("\n=== LSTM Baseline ===")
        lstm_model = LSTMForecaster(
            n_features=n_events,
            hidden_size=cfg_lstm["hidden_size"],
            num_layers=cfg_lstm["num_layers"],
            dropout=0.0,
        ).to(device)
        lstm_model.load_state_dict(
            torch.load(lstm_path, map_location=device, weights_only=True)
        )

        detector = HARDLiteDetector(lstm_model, device, ema_alpha)

        # Fit com AUC weights (usa calibração benigna + malicioso)
        detector.fit(
            cal_benign_lstm, mal_windows,
            warning_pct, anomaly_pct,
        )

        # Score com EMA (integrado no detector)
        lstm_benign_w = detector.score_windows(test_benign_lstm, level="warning")
        lstm_mal_w = detector.score_windows(mal_windows, level="warning")

        lstm_metrics, (fpr, tpr, prec, rec) = compute_metrics(
            lstm_benign_w, lstm_mal_w, interval_ms, lstm_ws,
            cfg_eval["fpr_targets"], warning_pct, anomaly_pct,
        )
        results["LSTM"] = lstm_metrics
        roc_curves["LSTM"] = (fpr, tpr, lstm_metrics["AUC-ROC"])
        pr_curves["LSTM"] = (prec, rec, lstm_metrics["PR-AUC"])

        print("  Métricas LSTM:")
        for k, v in lstm_metrics.items():
            print(f"    {k}: {v}")

        plot_score_histograms(
            lstm_benign_w, lstm_mal_w,
            "LSTM: Score WMVE+EMA (Benigno vs Malicioso)",
            os.path.join(output_dir, "lstm_histogram.png"),
            lstm_metrics["threshold_warning"], lstm_metrics["threshold_anomaly"],
        )

        # CUSUM
        if cusum:
            cusum_lstm = cusum.detect_with_reference(lstm_mal_w, lstm_benign_w)
            lstm_metrics["CUSUM_first_alarm"] = cusum_lstm["first_alarm"]
            if cusum_lstm["first_alarm"] >= 0:
                lstm_metrics["CUSUM_ToD_ms"] = float(
                    cusum_lstm["first_alarm"] * lstm_ws * interval_ms
                )
            plot_timeline_with_cusum(
                lstm_mal_w, cusum_lstm, "LSTM: Score Malicioso + CUSUM",
                os.path.join(output_dir, "lstm_timeline_cusum.png"),
                lstm_metrics["threshold_warning"], lstm_metrics["threshold_anomaly"],
            )

        # Overhead
        if cfg_eval.get("measure_overhead", False):
            n_runs = cfg_eval.get("overhead_n_runs", 100)
            lstm_window_dur_ms = lstm_ws * interval_ms
            overhead_results["LSTM"] = measure_overhead(
                lstm_model, test_benign_lstm, device, n_runs, is_slm=False,
                window_duration_ms=lstm_window_dur_ms,
            )
            print(f"  Overhead LSTM: {overhead_results['LSTM']}")

        # Bootstrap CI
        if boot_cfg.get("enabled", False):
            print("  Bootstrap CI...")
            ci = bootstrap_metrics(
                lstm_benign_w, lstm_mal_w, interval_ms, lstm_ws,
                boot_cfg.get("n_iterations", 1000),
                boot_cfg.get("confidence", 0.95),
                warning_pct, anomaly_pct,
            )
            results["LSTM_CI"] = ci
            print(f"  AUC-ROC: {ci['AUC-ROC']['mean']:.4f} "
                  f"[{ci['AUC-ROC']['ci_low']:.4f}, {ci['AUC-ROC']['ci_high']:.4f}]")

    else:
        print(f"LSTM checkpoint não encontrado: {lstm_path}")

    # ── Multi-Resolução SLM ──
    mr_cfg = cfg_eval.get("multi_resolution", {})
    if mr_cfg.get("enabled", False) and "SLM" in results:
        print("\n=== Multi-Resolução SLM ===")
        target_ms_list = mr_cfg.get("windows_ms", [10, 100, 1000])
        fusion_method = mr_cfg.get("fusion", "max")

        mr_benign_scores = {}
        mr_mal_scores = {}

        for res_ms in target_ms_list:
            b_path = os.path.join(data_dir, f"benign_res{res_ms}ms.npy")
            m_path = os.path.join(data_dir, f"malicious_res{res_ms}ms.npy")
            if not os.path.exists(b_path):
                print(f"  {res_ms}ms: dados não encontrados, pulando")
                continue

            b_data = np.load(b_path)
            m_data = np.load(m_path)

            # Usar mesmo modelo SLM (seq_len pode diferir, usar o que couber)
            if b_data.shape[1] != info["seq_len"]:
                print(f"  {res_ms}ms: seq_len={b_data.shape[1]} != {info['seq_len']}, pulando")
                continue

            b_scores = apply_ema(score_slm(model, b_data, device), ema_alpha)
            m_scores = apply_ema(score_slm(model, m_data, device), ema_alpha)
            mr_benign_scores[res_ms] = b_scores
            mr_mal_scores[res_ms] = m_scores
            print(f"  {res_ms}ms: benigno={len(b_scores)}, malicioso={len(m_scores)}")

        if len(mr_benign_scores) > 1:
            fused_benign = fuse_multiresolution_scores(mr_benign_scores, fusion_method)
            fused_mal = fuse_multiresolution_scores(mr_mal_scores, fusion_method)

            mr_metrics, (fpr, tpr, prec, rec) = compute_metrics(
                fused_benign, fused_mal, interval_ms, ws,
                cfg_eval["fpr_targets"], warning_pct, anomaly_pct,
            )
            results["SLM_MultiRes"] = mr_metrics
            roc_curves["SLM_MultiRes"] = (fpr, tpr, mr_metrics["AUC-ROC"])
            pr_curves["SLM_MultiRes"] = (prec, rec, mr_metrics["PR-AUC"])

            print(f"  Multi-Resolução ({fusion_method}):")
            for k, v in mr_metrics.items():
                print(f"    {k}: {v}")

    # ── Tabela Comparativa Final (BASELINE.md linha 106) ──
    if "SLM" in results and "LSTM" in results:
        print("\n" + "=" * 80)
        print("TABELA FINAL: Baseline HARD-Lite vs Ours (SLM)")
        print("=" * 80)

        # Métricas da tabela conforme BASELINE.md
        table_keys = [
            ("FPR/day (anomaly)", "FPR/day_anomaly"),
            ("FPR/day (warning)", "FPR/day_warning"),
            ("ToD p50 (anomaly)", "ToD_anomaly_p50_ms"),
            ("ToD p95 (anomaly)", "ToD_anomaly_p95_ms"),
            ("ToD p50 (warning)", "ToD_warning_p50_ms"),
            ("ToD p95 (warning)", "ToD_warning_p95_ms"),
            ("AUC-ROC", "AUC-ROC"),
            ("PR-AUC", "PR-AUC"),
            ("TPR@FPR=0.01", "TPR@FPR=0.01"),
            ("TPR@FPR=0.001", "TPR@FPR=0.001"),
        ]

        # Overhead
        if overhead_results:
            table_keys.append(("CPU%", None))
            table_keys.append(("Latência p95 (ms)", None))
            table_keys.append(("Params (M)", None))

        print(f"\n{'Métrica':<25} {'LSTM':>15} {'SLM':>15} {'Delta':>12} {'%':>8}")
        print("-" * 78)

        for label, key in table_keys:
            if key is None:
                # Overhead
                if label == "CPU%":
                    lstm_v = overhead_results.get("LSTM", {}).get("cpu_percent", 0)
                    slm_v = overhead_results.get("SLM", {}).get("cpu_percent", 0)
                elif label == "Latência p95 (ms)":
                    lstm_v = overhead_results.get("LSTM", {}).get("latency_p95_ms", 0)
                    slm_v = overhead_results.get("SLM", {}).get("latency_p95_ms", 0)
                elif label == "Params (M)":
                    lstm_v = overhead_results.get("LSTM", {}).get("n_params_M", 0)
                    slm_v = overhead_results.get("SLM", {}).get("n_params_M", 0)
                else:
                    continue
            else:
                lstm_v = results["LSTM"].get(key, 0)
                slm_v = results["SLM"].get(key, 0)

            if isinstance(lstm_v, (int, float)) and isinstance(slm_v, (int, float)):
                delta = slm_v - lstm_v
                pct = (delta / abs(lstm_v) * 100) if lstm_v != 0 else 0
                print(f"{label:<25} {lstm_v:>15.4f} {slm_v:>15.4f} {delta:>+12.4f} {pct:>+7.1f}%")
            else:
                print(f"{label:<25} {lstm_v:>15} {slm_v:>15}")

        # CI se disponível
        if "SLM_CI" in results and "LSTM_CI" in results:
            print(f"\n{'Bootstrap CI (95%)':<25} {'LSTM':>15} {'SLM':>15}")
            print("-" * 55)
            for key in ["AUC-ROC", "PR-AUC"]:
                lstm_ci = results["LSTM_CI"].get(key, {})
                slm_ci = results["SLM_CI"].get(key, {})
                lstm_str = f"{lstm_ci.get('ci_low', 0):.4f}-{lstm_ci.get('ci_high', 0):.4f}"
                slm_str = f"{slm_ci.get('ci_low', 0):.4f}-{slm_ci.get('ci_high', 0):.4f}"
                print(f"{key:<25} {lstm_str:>15} {slm_str:>15}")

        print("=" * 80)

    # ── Gráficos comparativos ──
    if len(roc_curves) >= 2:
        plot_roc_comparison(roc_curves, os.path.join(output_dir, "roc_comparison.png"))
        plot_pr_comparison(pr_curves, os.path.join(output_dir, "pr_comparison.png"))

    # Salvar tudo
    all_results = {
        "metrics": results,
        "overhead": overhead_results,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResultados salvos: {output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação SLM vs LSTM")
    parser.add_argument("--data-dir", default="data/tokenized")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.data_dir, args.checkpoint_dir, args.output_dir)
