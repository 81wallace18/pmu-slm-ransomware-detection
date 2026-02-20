"""
Treino benign-only do SLM (+ LoRA + Stage 1) e do LSTM baseline.

Uso:
    python src/train.py --model slm --data-dir data/tokenized
    python src/train.py --model lstm --data-dir data/tokenized
    python src/train.py --model slm-stage1 --data-dir data/tokenized
    python src/train.py --model slm-lora --data-dir data/tokenized --base-checkpoint checkpoints/slm_best.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from model import PMUSLM, apply_lora, save_lora
from baseline import LSTMForecaster


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_training(model, train_loader, val_loader, optimizer, scheduler,
                  config, output_dir, checkpoint_name, is_lm: bool = True):
    """Loop de treino genérico para SLM (cross-entropy) e LSTM (L1)."""
    cfg_train = config["train"]
    device = next(model.parameters()).device
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg_train["epochs"]):
        model.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_data in train_loader:
            if is_lm:
                (batch,) = batch_data
                batch = batch.to(device)
                input_ids = batch[:, :-1]
                target_ids = batch[:, 1:]
                _, loss = model(input_ids, target_ids)
            else:
                x, y = batch_data
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = torch.nn.functional.l1_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if is_lm:
                    (batch,) = batch_data
                    batch = batch.to(device)
                    input_ids = batch[:, :-1]
                    target_ids = batch[:, 1:]
                    _, loss = model(input_ids, target_ids)
                else:
                    x, y = batch_data
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = torch.nn.functional.l1_loss(pred, y)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        dt = time.time() - t0
        if is_lm:
            print(f"Epoch {epoch+1:3d}/{cfg_train['epochs']} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"ppl_train={np.exp(train_loss):.2f} ppl_val={np.exp(val_loss):.2f} | "
                  f"{dt:.1f}s")
        else:
            print(f"Epoch {epoch+1:3d}/{cfg_train['epochs']} | "
                  f"train_L1={train_loss:.4f} val_L1={val_loss:.4f} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, checkpoint_name))
        else:
            patience_counter += 1
            if patience_counter >= cfg_train["patience"]:
                print(f"Early stopping na epoch {epoch+1}")
                break

    print(f"\nMelhor val_loss: {best_val_loss:.4f}")
    if is_lm:
        print(f"  (ppl={np.exp(best_val_loss):.2f})")
    print(f"Modelo salvo em {output_dir}/{checkpoint_name}")


# ── SLM Training ──────────────────────────────────────────────────────────────

def train_slm(config: dict, data_dir: str, output_dir: str,
              stage1: bool = False):
    """Treina SLM (ou Stage 1 compacto) com next-token prediction."""
    device = get_device()
    cfg_train = config["train"]

    if stage1:
        cfg_model = config["stage1"]
        checkpoint_name = "slm_stage1_best.pt"
        print("=== Treinando SLM Stage 1 (compacto) ===")
    else:
        cfg_model = config["slm"]
        checkpoint_name = "slm_best.pt"
        print("=== Treinando SLM ===")

    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    train_data = np.load(os.path.join(data_dir, "train.npy"))
    val_data = np.load(os.path.join(data_dir, "val.npy"))
    print(f"Train: {train_data.shape}, Val: {val_data.shape}")

    train_ds = TensorDataset(torch.from_numpy(train_data))
    val_ds = TensorDataset(torch.from_numpy(val_data))
    train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_train["batch_size"], shuffle=False)

    model = PMUSLM(
        vocab_size=info["vocab_size"],
        n_embd=cfg_model["n_embd"],
        n_head=cfg_model["n_head"],
        n_layer=cfg_model["n_layer"],
        block_size=info["seq_len"],
        dropout=cfg_model["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_train["epochs"])

    os.makedirs(output_dir, exist_ok=True)
    _run_training(model, train_loader, val_loader, optimizer, scheduler,
                  config, output_dir, checkpoint_name, is_lm=True)


def train_slm_lora(config: dict, data_dir: str, output_dir: str,
                   base_checkpoint: str):
    """Fine-tune SLM com LoRA (para adaptação a nova CPU/workload)."""
    device = get_device()
    cfg_model = config["slm"]
    cfg_lora = config["lora"]
    cfg_train = config["train"]

    print("=== Fine-tuning SLM com LoRA ===")

    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    # Carregar modelo base
    model = PMUSLM(
        vocab_size=info["vocab_size"],
        n_embd=cfg_model["n_embd"],
        n_head=cfg_model["n_head"],
        n_layer=cfg_model["n_layer"],
        block_size=info["seq_len"],
        dropout=0.0,  # sem dropout durante LoRA (o adapter tem o seu)
    ).to(device)
    model.load_state_dict(torch.load(base_checkpoint, map_location=device, weights_only=True))

    # Aplicar LoRA
    model = apply_lora(
        model,
        rank=cfg_lora["rank"],
        alpha=cfg_lora["alpha"],
        dropout=cfg_lora["dropout"],
        target_modules=cfg_lora["target_modules"],
    )

    train_data = np.load(os.path.join(data_dir, "train.npy"))
    val_data = np.load(os.path.join(data_dir, "val.npy"))

    train_ds = TensorDataset(torch.from_numpy(train_data))
    val_ds = TensorDataset(torch.from_numpy(val_data))
    train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_train["batch_size"], shuffle=False)

    # Só otimizar parâmetros treináveis (LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg_train["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_train["epochs"])

    os.makedirs(output_dir, exist_ok=True)
    _run_training(model, train_loader, val_loader, optimizer, scheduler,
                  config, output_dir, "slm_lora_best.pt", is_lm=True)

    # Salvar apenas os adapters LoRA
    save_lora(model, os.path.join(output_dir, "lora_adapters.pt"))


# ── LSTM Training ─────────────────────────────────────────────────────────────

def train_lstm(config: dict, data_dir: str, output_dir: str):
    """Treina LSTM forecasting baseline em dados benignos contínuos."""
    device = get_device()
    cfg_lstm = config["lstm"]
    cfg_train = config["train"]
    events = config["collect"]["events"]
    n_events = len(events)
    ws = cfg_lstm["window_size"]

    print("=== Treinando LSTM Baseline ===")

    benign = np.load(os.path.join(data_dir, "benign_continuous.npy"))
    print(f"Dados contínuos benignos: {benign.shape}")

    windows = []
    for i in range(len(benign) - ws):
        windows.append(benign[i:i + ws])
    windows = np.array(windows, dtype=np.float32)
    print(f"Janelas LSTM: {windows.shape}")

    n = len(windows)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_x = torch.from_numpy(windows[:n_train, :-1, :])
    train_y = torch.from_numpy(windows[:n_train, -1, :])
    val_x = torch.from_numpy(windows[n_train:n_train + n_val, :-1, :])
    val_y = torch.from_numpy(windows[n_train:n_train + n_val, -1, :])

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_train["batch_size"], shuffle=False)

    model = LSTMForecaster(
        n_features=n_events,
        hidden_size=cfg_lstm["hidden_size"],
        num_layers=cfg_lstm["num_layers"],
        dropout=cfg_lstm["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTMForecaster: {n_params / 1e6:.2f}M parâmetros")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_train["epochs"])

    os.makedirs(output_dir, exist_ok=True)
    _run_training(model, train_loader, val_loader, optimizer, scheduler,
                  config, output_dir, "lstm_best.pt", is_lm=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino benign-only")
    parser.add_argument("--model", required=True,
                        choices=["slm", "lstm", "slm-stage1", "slm-lora"])
    parser.add_argument("--data-dir", default="data/tokenized")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-checkpoint", default=None,
                        help="Checkpoint base para LoRA fine-tuning")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model == "slm":
        train_slm(config, args.data_dir, args.output_dir)
    elif args.model == "slm-stage1":
        train_slm(config, args.data_dir, args.output_dir, stage1=True)
    elif args.model == "slm-lora":
        if not args.base_checkpoint:
            print("ERRO: --base-checkpoint é obrigatório para slm-lora")
            exit(1)
        train_slm_lora(config, args.data_dir, args.output_dir, args.base_checkpoint)
    else:
        train_lstm(config, args.data_dir, args.output_dir)
