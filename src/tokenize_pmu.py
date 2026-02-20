"""
Tokenização de séries temporais PMU.

Transforma valores contínuos de contadores em tokens discretos (bins)
usando quantis calculados a partir de dados benignos.
Inclui tokens de ritmo (burst/steady/drop) para capturar dinâmica temporal.

Uso:
    python src/tokenize_pmu.py \
        --benign data/benign_*.csv \
        --malicious data/malicious_*.csv \
        --output-dir data/tokenized
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_csvs(patterns: list[str]) -> pd.DataFrame:
    """Carrega múltiplos CSVs e concatena."""
    dfs = []
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        for f in files:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"Nenhum arquivo encontrado para: {patterns}")
    return pd.concat(dfs, ignore_index=True)


# Tokens de ritmo: 3 estados por evento
RHYTHM_BURST = 0    # derivada positiva forte
RHYTHM_STEADY = 1   # estável
RHYTHM_DROP = 2     # derivada negativa forte


class PMUTokenizer:
    """
    Tokeniza contadores PMU em bins discretos + tokens de ritmo.

    Vocabulário:
      - Bins: evento_idx * n_bins + bin_idx  (0 .. n_events*n_bins - 1)
      - Ritmo: n_events*n_bins + evento_idx*3 + rhythm_state
      - Especiais: pad, sos, eos no final
    """

    def __init__(self, events: list[str], n_bins: int = 32, method: str = "quantile",
                 rhythm_tokens: bool = True, rhythm_window: int = 5):
        self.events = events
        self.n_bins = n_bins
        self.method = method
        self.rhythm_tokens = rhythm_tokens
        self.rhythm_window = rhythm_window
        self.bin_edges: dict[str, np.ndarray] = {}
        self.rhythm_thresholds: dict[str, tuple[float, float]] = {}
        self.fitted = False

        n_events = len(events)
        n_bin_tokens = n_events * n_bins
        n_rhythm_tokens = n_events * 3 if rhythm_tokens else 0

        self.rhythm_offset = n_bin_tokens
        self.pad_token = n_bin_tokens + n_rhythm_tokens
        self.sos_token = n_bin_tokens + n_rhythm_tokens + 1
        self.eos_token = n_bin_tokens + n_rhythm_tokens + 2
        self.vocab_size = n_bin_tokens + n_rhythm_tokens + 3

        # Tokens por frame: n_events (bins) + n_events (ritmo) se habilitado
        self.tokens_per_frame = n_events + (n_events if rhythm_tokens else 0)

    def fit(self, benign_df: pd.DataFrame) -> "PMUTokenizer":
        """Calcula edges dos bins e limiares de ritmo a partir de dados benignos."""
        for event in self.events:
            values = benign_df[event].dropna().values

            # Bin edges
            if self.method == "quantile":
                percentiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(values, percentiles)
                edges = np.unique(edges)
                if len(edges) < 3:
                    edges = np.linspace(values.min(), values.max() + 1e-10, self.n_bins + 1)
            elif self.method == "uniform":
                edges = np.linspace(values.min(), values.max() + 1e-10, self.n_bins + 1)
            else:
                raise ValueError(f"Método desconhecido: {self.method}")

            self.bin_edges[event] = edges

            # Limiares de ritmo: baseados na derivada local
            if self.rhythm_tokens and len(values) > self.rhythm_window:
                diffs = np.diff(values)
                abs_diffs = np.abs(diffs)
                p75 = np.percentile(abs_diffs, 75)
                # burst: derivada > p75, drop: derivada < -p75, steady: entre
                self.rhythm_thresholds[event] = (p75, -p75)

        self.fitted = True
        return self

    def _bin_token(self, event_idx: int, value: float) -> int:
        """Converte valor contínuo em token de bin."""
        event = self.events[event_idx]
        edges = self.bin_edges[event]
        bin_idx = np.searchsorted(edges, value, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, min(len(edges) - 2, self.n_bins - 1))
        return event_idx * self.n_bins + int(bin_idx)

    def _rhythm_token(self, event_idx: int, derivative: float) -> int:
        """Converte derivada local em token de ritmo."""
        event = self.events[event_idx]
        high, low = self.rhythm_thresholds[event]
        if derivative > high:
            state = RHYTHM_BURST
        elif derivative < low:
            state = RHYTHM_DROP
        else:
            state = RHYTHM_STEADY
        return self.rhythm_offset + event_idx * 3 + state

    def transform_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma DataFrame em array de tokens (n_frames, tokens_per_frame).

        Cada frame produz:
          [bin_ev0, bin_ev1, ..., bin_evN, rhythm_ev0, ..., rhythm_evN]
        """
        if not self.fitted:
            raise RuntimeError("Tokenizer não foi treinado (chame fit() primeiro)")

        n_frames = len(df)
        n_events = len(self.events)
        tokens = np.zeros((n_frames, self.tokens_per_frame), dtype=np.int64)

        values = df[self.events].values  # (n_frames, n_events)

        for i in range(n_frames):
            # Tokens de bin
            for j in range(n_events):
                tokens[i, j] = self._bin_token(j, values[i, j])

            # Tokens de ritmo
            if self.rhythm_tokens:
                if i < self.rhythm_window:
                    # Primeiros frames: steady (não temos histórico suficiente)
                    for j in range(n_events):
                        tokens[i, n_events + j] = self.rhythm_offset + j * 3 + RHYTHM_STEADY
                else:
                    for j in range(n_events):
                        # Derivada local: média das diferenças na janela
                        window_vals = values[i - self.rhythm_window:i + 1, j]
                        derivative = (window_vals[-1] - window_vals[0]) / self.rhythm_window
                        tokens[i, n_events + j] = self._rhythm_token(j, derivative)

        return tokens

    def save(self, path: str):
        """Salva tokenizer treinado."""
        state = {
            "events": self.events,
            "n_bins": self.n_bins,
            "method": self.method,
            "rhythm_tokens": self.rhythm_tokens,
            "rhythm_window": self.rhythm_window,
            "vocab_size": self.vocab_size,
            "tokens_per_frame": self.tokens_per_frame,
            "bin_edges": {k: v.tolist() for k, v in self.bin_edges.items()},
            "rhythm_thresholds": {k: list(v) for k, v in self.rhythm_thresholds.items()},
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PMUTokenizer":
        """Carrega tokenizer salvo."""
        with open(path) as f:
            state = json.load(f)
        tok = cls(
            state["events"], state["n_bins"], state["method"],
            state.get("rhythm_tokens", False), state.get("rhythm_window", 5),
        )
        tok.bin_edges = {k: np.array(v) for k, v in state["bin_edges"].items()}
        tok.rhythm_thresholds = {k: tuple(v) for k, v in state.get("rhythm_thresholds", {}).items()}
        tok.fitted = True
        return tok


def create_windows(tokens: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Cria janelas deslizantes de tokens.

    Args:
        tokens: (n_frames, tokens_per_frame)
        window_size: número de frames por janela
        stride: passo entre janelas

    Returns:
        (n_windows, window_size * tokens_per_frame) — sequência achatada
    """
    n_frames = tokens.shape[0]
    windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        window = tokens[start:start + window_size]
        windows.append(window.flatten())
    if not windows:
        raise ValueError(f"Dados insuficientes: {n_frames} frames < window_size {window_size}")
    return np.array(windows, dtype=np.int64)


def create_multiresolution_windows(tokens: np.ndarray, base_interval_ms: int,
                                   target_ms_list: list[int],
                                   window_size: int, stride: int = 1) -> dict[int, np.ndarray]:
    """
    Cria janelas em múltiplas resoluções temporais.

    Para resoluções maiores que base_interval_ms, agrega frames (moda dos bins).

    Args:
        tokens: (n_frames, tokens_per_frame) em resolução base
        base_interval_ms: intervalo base (ex: 10ms)
        target_ms_list: lista de resoluções alvo (ex: [10, 100, 1000])
        window_size: frames por janela
        stride: passo entre janelas

    Returns:
        dict de resolução_ms -> (n_windows, seq_len) janelas
    """
    result = {}

    for target_ms in target_ms_list:
        agg_factor = max(1, target_ms // base_interval_ms)

        if agg_factor == 1:
            # Resolução base: sem agregação
            result[target_ms] = create_windows(tokens, window_size, stride)
        else:
            # Agregar frames: usar último frame do bloco (mais recente)
            n_agg_frames = len(tokens) // agg_factor
            if n_agg_frames < window_size:
                print(f"  Aviso: resolução {target_ms}ms insuficiente "
                      f"({n_agg_frames} < {window_size} frames). Pulando.")
                continue
            agg_tokens = tokens[agg_factor - 1::agg_factor][:n_agg_frames]
            result[target_ms] = create_windows(agg_tokens, window_size, stride)

    return result


def prepare_datasets(config: dict, benign_files: list[str], malicious_files: list[str],
                     output_dir: str) -> dict:
    """Pipeline completo: load -> fit tokenizer -> transform -> window -> save."""
    tok_cfg = config["tokenize"]
    events = config["collect"]["events"]
    interval_ms = config["collect"]["interval_ms"]

    os.makedirs(output_dir, exist_ok=True)

    # Carregar dados
    print("Carregando dados benignos...")
    benign_df = load_csvs(benign_files)
    print(f"  {len(benign_df)} frames benignos")

    print("Carregando dados maliciosos...")
    mal_df = load_csvs(malicious_files)
    print(f"  {len(mal_df)} frames maliciosos")

    # Fit tokenizer no benigno
    print("Treinando tokenizer nos dados benignos...")
    rhythm = tok_cfg.get("rhythm_tokens", True)
    rhythm_win = tok_cfg.get("rhythm_window", 5)
    tokenizer = PMUTokenizer(events, tok_cfg["n_bins"], tok_cfg["method"],
                             rhythm, rhythm_win)
    tokenizer.fit(benign_df)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Tokens per frame: {tokenizer.tokens_per_frame}")
    print(f"  Rhythm tokens: {rhythm}")

    # Tokenizar
    print("Tokenizando...")
    benign_tokens = tokenizer.transform_series(benign_df)
    mal_tokens = tokenizer.transform_series(mal_df)

    # Criar janelas (resolução base)
    ws = tok_cfg["window_size"]
    stride = tok_cfg["stride"]
    print(f"Criando janelas (window_size={ws}, stride={stride})...")
    benign_windows = create_windows(benign_tokens, ws, stride)
    mal_windows = create_windows(mal_tokens, ws, stride)
    print(f"  Janelas benignas:   {benign_windows.shape}")
    print(f"  Janelas maliciosas: {mal_windows.shape}")

    # Multi-resolução
    multi_res_cfg = config.get("eval", {}).get("multi_resolution", {})
    if multi_res_cfg.get("enabled", False):
        target_ms = multi_res_cfg.get("windows_ms", [10, 100, 1000])
        print(f"Criando janelas multi-resolução: {target_ms}ms...")

        benign_mr = create_multiresolution_windows(
            benign_tokens, interval_ms, target_ms, ws, stride)
        mal_mr = create_multiresolution_windows(
            mal_tokens, interval_ms, target_ms, ws, stride)

        for res_ms in benign_mr:
            np.save(os.path.join(output_dir, f"benign_res{res_ms}ms.npy"), benign_mr[res_ms])
            np.save(os.path.join(output_dir, f"malicious_res{res_ms}ms.npy"), mal_mr[res_ms])
            print(f"  {res_ms}ms: benigno={benign_mr[res_ms].shape}, "
                  f"malicioso={mal_mr[res_ms].shape}")

    # Split temporal do benigno: treino / val / teste
    n = len(benign_windows)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": benign_windows[:n_train],
        "val": benign_windows[n_train:n_train + n_val],
        "test_benign": benign_windows[n_train + n_val:],
        "test_malicious": mal_windows,
    }

    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, data)
        print(f"  {name}: {data.shape} -> {path}")

    # Salvar dados contínuos (para baseline LSTM)
    benign_cont = benign_df[events].values.astype(np.float32)
    mal_cont = mal_df[events].values.astype(np.float32)

    # Normalização z-score (fit nos frames de treino do benigno)
    approx_train_frames = n_train * ws
    means = benign_cont[:approx_train_frames].mean(axis=0)
    stds = benign_cont[:approx_train_frames].std(axis=0) + 1e-8
    benign_cont_norm = (benign_cont - means) / stds
    mal_cont_norm = (mal_cont - means) / stds

    np.save(os.path.join(output_dir, "benign_continuous.npy"), benign_cont_norm)
    np.save(os.path.join(output_dir, "malicious_continuous.npy"), mal_cont_norm)
    np.savez(os.path.join(output_dir, "norm_params.npz"), means=means, stds=stds)
    print("  Dados contínuos normalizados salvos (para baseline LSTM)")

    seq_len = ws * tokenizer.tokens_per_frame
    info = {
        "vocab_size": tokenizer.vocab_size,
        "n_events": len(events),
        "tokens_per_frame": tokenizer.tokens_per_frame,
        "rhythm_tokens": rhythm,
        "window_size": ws,
        "seq_len": seq_len,
        "interval_ms": interval_ms,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test_benign": len(splits["test_benign"]),
        "n_test_malicious": len(splits["test_malicious"]),
    }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenização de PMU")
    parser.add_argument("--benign", nargs="+", required=True, help="Padrão glob dos CSVs benignos")
    parser.add_argument("--malicious", nargs="+", required=True, help="Padrão glob dos CSVs maliciosos")
    parser.add_argument("--output-dir", default="data/tokenized")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    info = prepare_datasets(config, args.benign, args.malicious, args.output_dir)
    print("\nResumo:")
    for k, v in info.items():
        print(f"  {k}: {v}")
