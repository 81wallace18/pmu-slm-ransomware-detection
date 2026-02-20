"""
Coleta de PMU via perf stat.

Uso:
    python src/collect.py --label benign --workload idle --duration 300
    python src/collect.py --label malicious --workload ransomware_x --duration 60

Gera CSV em data/<label>_<workload>_<timestamp>.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_perf_output(raw: str, events: list[str]) -> dict[str, float] | None:
    """Parseia uma linha de saída do perf stat -x ';'."""
    values = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(";")
        if len(parts) < 3:
            continue
        val_str = parts[0].strip()
        event_name = parts[2].strip()
        # perf pode retornar <not counted> ou <not supported>
        if "<" in val_str:
            return None
        try:
            val = float(val_str)
        except ValueError:
            continue
        # Mapear nome do evento para nome curto
        for ev in events:
            if ev in event_name:
                values[ev] = val
                break
    if len(values) != len(events):
        return None
    return values


def collect(config: dict, label: str, workload: str, duration: int | None = None,
            output_dir: str = "data") -> str:
    """Executa coleta PMU e grava CSV."""
    cfg = config["collect"]
    events = cfg["events"]
    interval_ms = cfg["interval_ms"]
    dur = duration or cfg["duration_sec"]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{workload}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    event_str = ",".join(events)
    interval_sec = interval_ms / 1000.0

    print(f"Coletando PMU: events={events}")
    print(f"  interval={interval_ms}ms, duration={dur}s")
    print(f"  label={label}, workload={workload}")
    print(f"  output={filepath}")

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp_ms", "label", "workload"] + events)
        writer.writeheader()

        start = time.time()
        frame_idx = 0

        while (time.time() - start) < dur:
            t0 = time.time()

            # perf stat com intervalo curto, -x separador
            cmd = [
                "perf", "stat",
                "-e", event_str,
                "-x", ";",
                "-a",  # system-wide
                "--", "sleep", str(interval_sec),
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=interval_sec + 2
                )
            except subprocess.TimeoutExpired:
                continue
            except FileNotFoundError:
                print("ERRO: 'perf' não encontrado. Instale linux-tools-common.")
                sys.exit(1)

            # perf stat escreve em stderr
            parsed = parse_perf_output(result.stderr, events)
            if parsed is None:
                continue

            elapsed_ms = int((time.time() - start) * 1000)
            row = {"timestamp_ms": elapsed_ms, "label": label, "workload": workload}
            row.update(parsed)
            writer.writerow(row)

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - start
                print(f"  [{elapsed:.1f}s / {dur}s] {frame_idx} frames coletados")

    print(f"Coleta finalizada: {frame_idx} frames -> {filepath}")
    return filepath


def generate_synthetic(config: dict, output_dir: str = "data") -> tuple[str, str]:
    """
    Gera dados sintéticos para teste rápido (sem precisar de perf/root).
    Simula padrões benignos e maliciosos com distribuições diferentes.
    """
    import numpy as np

    cfg = config["collect"]
    events = cfg["events"]
    n_events = len(events)
    interval_ms = cfg["interval_ms"]
    duration = cfg["duration_sec"]
    n_frames = int(duration * 1000 / interval_ms)

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    # --- Benigno: padrão "normal" com variação moderada ---
    benign_means = np.array([1e9, 8e8, 5e7, 1e6, 2e8, 5e6], dtype=np.float64)
    benign_stds = benign_means * 0.15

    benign_data = np.zeros((n_frames, n_events))
    for i in range(n_frames):
        # Simular transições de workload (mudança de regime a cada ~500 frames)
        regime = (i // 500) % 3
        scale = [1.0, 1.5, 0.7][regime]
        noise = rng.randn(n_events) * benign_stds * scale
        benign_data[i] = benign_means * scale + noise
        benign_data[i] = np.maximum(benign_data[i], 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benign_path = os.path.join(output_dir, f"benign_synthetic_{timestamp}.csv")
    with open(benign_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "label", "workload"] + events)
        writer.writeheader()
        for i in range(n_frames):
            row = {"timestamp_ms": i * interval_ms, "label": "benign", "workload": "synthetic"}
            for j, ev in enumerate(events):
                row[ev] = benign_data[i, j]
            writer.writerow(row)

    # --- Malicioso: padrão anômalo (I/O pesado, cache thrashing) ---
    n_mal_frames = n_frames // 5  # 20% da duração
    mal_data = np.zeros((n_mal_frames, n_events))

    # Ransomware: muitas instruções, cache-misses alto, branches erráticos
    mal_means = np.array([2e9, 2.5e9, 3e8, 8e7, 5e8, 4e7], dtype=np.float64)
    mal_stds = mal_means * 0.2

    for i in range(n_mal_frames):
        # Padrão de encrypt: bursts periódicos
        burst = 1.0 + 0.5 * np.sin(i * 0.1)
        noise = rng.randn(n_events) * mal_stds
        mal_data[i] = mal_means * burst + noise
        mal_data[i] = np.maximum(mal_data[i], 0)

    mal_path = os.path.join(output_dir, f"malicious_synthetic_{timestamp}.csv")
    with open(mal_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "label", "workload"] + events)
        writer.writeheader()
        for i in range(n_mal_frames):
            row = {"timestamp_ms": i * interval_ms, "label": "malicious", "workload": "synthetic_ransom"}
            for j, ev in enumerate(events):
                row[ev] = mal_data[i, j]
            writer.writerow(row)

    print(f"Dados sintéticos gerados:")
    print(f"  Benigno:   {n_frames} frames -> {benign_path}")
    print(f"  Malicioso: {n_mal_frames} frames -> {mal_path}")
    return benign_path, mal_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta PMU ou gera dados sintéticos")
    sub = parser.add_subparsers(dest="command")

    # Coleta real
    real = sub.add_parser("real", help="Coleta real via perf stat")
    real.add_argument("--label", required=True, choices=["benign", "malicious"])
    real.add_argument("--workload", required=True, help="Nome do workload (ex: idle, build, ransomware_x)")
    real.add_argument("--duration", type=int, default=None, help="Duração em segundos")
    real.add_argument("--config", default="configs/default.yaml")
    real.add_argument("--output-dir", default="data")

    # Dados sintéticos
    synth = sub.add_parser("synthetic", help="Gera dados sintéticos para teste")
    synth.add_argument("--config", default="configs/default.yaml")
    synth.add_argument("--output-dir", default="data")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)

    if args.command == "real":
        collect(config, args.label, args.workload, args.duration, args.output_dir)
    elif args.command == "synthetic":
        generate_synthetic(config, args.output_dir)
