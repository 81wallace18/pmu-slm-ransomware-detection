"""
SLM para detecção de anomalias em telemetria PMU.

Transformer decoder-only (estilo GPT-2) treinado com next-token prediction
em dados benignos. Score de anomalia = NLL/perplexity da janela.

Inclui:
- LoRA adapters para portabilidade entre CPUs/workloads
- Variante Stage 1 (compacta, quantizável) para on-host
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Linear layer com LoRA adapter.
    Congela pesos originais e treina apenas A e B (rank baixo).
    W_new = W_frozen + (B @ A) * (alpha / rank)
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Congelar pesos originais
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
               dropout: float = 0.05, target_modules: list[str] | None = None) -> nn.Module:
    """
    Aplica LoRA adapters aos módulos lineares do modelo.

    Args:
        model: modelo base
        rank: rank do adapter
        alpha: scaling factor
        dropout: dropout do adapter
        target_modules: nomes dos módulos para aplicar LoRA (None = todos os Linear)
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "head"]

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            mod_name = parts[-1]
            if mod_name in target_modules:
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                lora = LoRALinear(module, rank, alpha, dropout)
                setattr(parent, mod_name, lora)
                count += 1

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA aplicado: {count} módulos, {n_trainable}/{n_total} "
          f"params treináveis ({100*n_trainable/n_total:.1f}%)")
    return model


def save_lora(model: nn.Module, path: str):
    """Salva apenas pesos LoRA (adapters)."""
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_state[name] = param.data
    torch.save(lora_state, path)
    print(f"LoRA salvo: {len(lora_state)} tensores -> {path}")


def load_lora(model: nn.Module, path: str, device: torch.device):
    """Carrega pesos LoRA no modelo."""
    lora_state = torch.load(path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    model_state.update(lora_state)
    model.load_state_dict(model_state)
    print(f"LoRA carregado: {len(lora_state)} tensores de {path}")


# ── Transformer Blocks ───────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ── SLM Principal ────────────────────────────────────────────────────────────

class PMUSLM(nn.Module):
    """
    Small Language Model para telemetria PMU.
    Decoder-only transformer com next-token prediction.
    """

    def __init__(self, vocab_size: int, n_embd: int = 128, n_head: int = 4,
                 n_layer: int = 4, block_size: int = 300, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"PMUSLM: {n_params / 1e6:.2f}M parâmetros")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequência {T} > block_size {self.block_size}"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    def compute_nll(self, idx: torch.Tensor) -> torch.Tensor:
        """
        NLL média por token para cada janela. Usado como score de anomalia.

        Args:
            idx: (B, T) sequência de tokens

        Returns:
            nll: (B,) NLL média por token
        """
        B, T = idx.shape
        input_ids = idx[:, :-1]
        target_ids = idx[:, 1:]

        logits, _ = self.forward(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        nll = -target_log_probs.mean(dim=1)
        return nll

    def compute_perplexity(self, idx: torch.Tensor) -> torch.Tensor:
        """Perplexity por janela = exp(NLL)."""
        return torch.exp(self.compute_nll(idx))


# ── 2-Stage Pipeline ─────────────────────────────────────────────────────────

class TwoStageDetector:
    """
    Pipeline de detecção em 2 estágios.

    Stage 1 (on-host): SLM compacto quantizado, alta sensibilidade (warning).
    Stage 2 (servidor): SLM maior, alta especificidade (anomaly confirmation).

    Janelas que passam do limiar no Stage 1 são enviadas ao Stage 2.
    Só dispara anomaly se ambos concordam.
    """

    def __init__(self, stage1_model: PMUSLM, stage2_model: PMUSLM,
                 device: torch.device,
                 stage1_percentile: float = 90,
                 stage2_percentile: float = 99,
                 ema_alpha: float = 0.3):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.device = device
        self.stage1_percentile = stage1_percentile
        self.stage2_percentile = stage2_percentile
        self.ema_alpha = ema_alpha
        self.stage1_threshold: float | None = None
        self.stage2_threshold: float | None = None

    def calibrate(self, benign_data: torch.Tensor, batch_size: int = 256):
        """Calibra limiares em dados benignos."""
        s1_scores = self._score_model(self.stage1, benign_data, batch_size)
        s2_scores = self._score_model(self.stage2, benign_data, batch_size)

        self.stage1_threshold = float(np.percentile(s1_scores, self.stage1_percentile))
        self.stage2_threshold = float(np.percentile(s2_scores, self.stage2_percentile))

        print(f"Stage 1 threshold (p{self.stage1_percentile}): {self.stage1_threshold:.4f}")
        print(f"Stage 2 threshold (p{self.stage2_percentile}): {self.stage2_threshold:.4f}")

    def detect(self, data: torch.Tensor, batch_size: int = 256) -> dict:
        """
        Detecção em 2 estágios.

        Returns:
            dict com scores, warnings (stage1), anomalies (stage1+stage2),
            e taxa de passagem stage1->stage2.
        """
        import numpy as np  # noqa: local import para evitar circular

        s1_scores = self._score_model(self.stage1, data, batch_size)
        s1_ema = self._apply_ema(s1_scores)

        # Stage 1: warning
        stage1_flags = s1_ema > self.stage1_threshold

        # Stage 2: só roda nas janelas que passaram stage 1
        s2_scores = np.full_like(s1_scores, np.nan)
        suspect_idx = np.where(stage1_flags)[0]

        if len(suspect_idx) > 0:
            suspect_data = data[suspect_idx]
            s2_suspect = self._score_model(self.stage2, suspect_data, batch_size)
            s2_scores[suspect_idx] = s2_suspect

        # Anomaly: stage1 + stage2 acima do limiar
        anomaly_flags = np.zeros(len(data), dtype=bool)
        for idx in suspect_idx:
            if s2_scores[idx] > self.stage2_threshold:
                anomaly_flags[idx] = True

        passthrough_rate = len(suspect_idx) / len(data) if len(data) > 0 else 0

        return {
            "stage1_scores": s1_scores,
            "stage1_ema": s1_ema,
            "stage2_scores": s2_scores,
            "warnings": stage1_flags,
            "anomalies": anomaly_flags,
            "passthrough_rate": passthrough_rate,
            "n_stage1_triggered": int(stage1_flags.sum()),
            "n_stage2_confirmed": int(anomaly_flags.sum()),
        }

    def _score_model(self, model: PMUSLM, data, batch_size: int) -> "np.ndarray":
        import numpy as np
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                if isinstance(data, np.ndarray):
                    batch = torch.from_numpy(data[i:i + batch_size]).to(self.device)
                else:
                    batch = data[i:i + batch_size].to(self.device)
                nll = model.compute_nll(batch)
                scores.append(nll.cpu().numpy())
        return np.concatenate(scores)

    def _apply_ema(self, scores: "np.ndarray") -> "np.ndarray":
        import numpy as np
        ema = np.zeros_like(scores)
        ema[0] = scores[0]
        for i in range(1, len(scores)):
            ema[i] = self.ema_alpha * scores[i] + (1 - self.ema_alpha) * ema[i - 1]
        return ema


def quantize_model(model: PMUSLM) -> nn.Module:
    """Quantiza modelo para INT8 (CPU) via PyTorch dynamic quantization."""
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Embedding}, dtype=torch.qint8
    )
    return quantized


# Importação necessária para TwoStageDetector
import numpy as np  # noqa: E402
