# Arquitetura do Sistema — Implementação Real

Documento atualizado para refletir o que foi efetivamente implementado.

---

## 1) Arquitetura geral (2 estágios)

### Host monitorado (Stage 1 — leve / resiliente)

* Coleta PMU system-wide via `perf stat` a cada 10ms
* Tokenização: bins quantílicos (32 bins por evento) + tokens de ritmo (burst/steady/drop)
* Score de anomalia: NLL/perplexity do SLM compacto quantizado (INT8)
* Se score > limiar warning: alerta local + envia janela ao servidor

**Implementação:** `src/collect.py` (coleta) + `src/tokenize_pmu.py` (tokenização) + `src/model.py:TwoStageDetector` (Stage 1)

### Servidor (Stage 2 — confirmador)

* Recebe janelas suspeitas do Stage 1
* Roda SLM maior + EMA + CUSUM change-point
* Confirma anomaly com baixa taxa de falso positivo
* Multi-resolução (10ms/100ms/1s) com fusão configurável (max, weighted_avg, vote)

**Implementação:** `src/model.py:TwoStageDetector` (Stage 2) + `src/change_point.py` (CUSUM + fusão)

---

## 2) Componentes do projeto

### (A) Collector (`src/collect.py`)

* Coleta via `perf stat -x ";" -a` com subprocess
* Parsing de saída CSV com 6 eventos: cycles, instructions, cache-references, cache-misses, branch-instructions, branch-misses
* Modo real (requer root + perf) e modo sintético (para teste sem hardware)
* Saída: CSV com timestamp_ms, label, workload, + valores por evento

### (B) Tokenizer (`src/tokenize_pmu.py`)

* `PMUTokenizer`: transforma valores contínuos em tokens discretos
* Bins por quantis do benigno (mais robusto que min-max)
* Tokens de ritmo: derivada local em janela de 5 frames → burst (↑), steady (=), drop (↓)
* Vocabulário: `n_events * n_bins + n_events * 3 + 3` (bins + ritmo + especiais)
* Multi-resolução: `create_multiresolution_windows()` agrega frames para 100ms e 1s
* Gera dataset duplo: tokenizado (para SLM) + contínuo normalizado (para LSTM)

### (C) SLM (`src/model.py`)

* `PMUSLM`: transformer decoder-only (estilo GPT-2 compacto)
* Default: 128 embed, 4 heads, 4 layers (~0.8M params)
* Stage 1: 64 embed, 2 heads, 2 layers (~0.2M params)
* Next-token prediction benign-only
* Score: `compute_nll()` → NLL média por token na janela
* LoRA: `LoRALinear` + `apply_lora()` para adaptação por CPU/workload
* Quantização: `quantize_model()` via PyTorch dynamic quantization (INT8)
* `TwoStageDetector`: orquestra Stage 1 (warning) → Stage 2 (anomaly confirmation)

### (D) Baseline LSTM (`src/baseline.py`)

* `LSTMForecaster`: LSTM 2-layer que prevê próximo frame dado N-1
* `HARDLiteDetector`:
  * Erro L1 por feature
  * WMVE com pesos AUC-ROC por feature (`_compute_auc_weights()`)
  * EMA suavização (`_apply_ema()`)
  * 2 níveis de alerta: warning (p90) e anomaly (p99)
  * `detect()` retorna scores, warnings, anomalies

### (E) Treino (`src/train.py`)

* 4 modos: `slm`, `slm-stage1`, `slm-lora`, `lstm`
* Treino benign-only com early stopping e cosine annealing
* LoRA: congela modelo base, treina apenas adapters (~0.1% dos params)
* Salva checkpoints + adapters LoRA separados

### (F) Avaliação (`src/eval.py`)

* Métricas: AUC-ROC, PR-AUC, TPR@FPR fixo
* FPR/dia: converte fração de janelas para alarmes/dia em operação contínua
* ToD: tempo até detecção em ms, com p50 e p95 (simula múltiplas execuções)
* CUSUM change-point nos scores maliciosos
* Multi-resolução: scoring em 10ms/100ms/1s + fusão configurável
* Overhead: latência p50/p95/p99, CPU%, memória, contagem de parâmetros
* Bootstrap CI (95%) para AUC-ROC, PR-AUC, FPR/dia
* Tabela final: LSTM vs SLM com delta absoluto e percentual

---

## 3) Stack implementada

| Componente | Tecnologia |
|---|---|
| Coleta PMU | `perf stat` via subprocess (Python) |
| Linguagem | Python (todo o projeto) |
| Treino/inferência | PyTorch (sem HuggingFace) |
| Fine-tuning | LoRA implementado from scratch |
| Quantização | `torch.ao.quantization.quantize_dynamic` (INT8) |
| Armazenamento | CSV (coleta) + NumPy .npy (datasets) + JSON (metadata) |
| Config | YAML (`configs/*.yaml`) |
| Avaliação | scikit-learn (métricas) + matplotlib (gráficos) |

**Decisão pragmática:** Tudo em Python, sem Rust/Go para o collector, sem gRPC/Protobuf, sem vLLM. Foco em reprodutibilidade acadêmica, não em produção.

---

## 4) Tamanho do SLM

### Implementado

* **Stage 2 (servidor):** ~0.8M params (128 embed, 4 heads, 4 layers)
* **Stage 1 (host):** ~0.2M params (64 embed, 2 heads, 2 layers)

### Experimento de scaling (`configs/model_scaling.yaml`)

| Config | embed | heads | layers | Params estimados |
|---|---|---|---|---|
| Tiny | 64 | 2 | 2 | ~0.2M |
| Small | 128 | 4 | 4 | ~0.8M |
| Medium | 256 | 8 | 6 | ~5M |
| Large | 512 | 8 | 8 | ~20M |

**Nota:** Modelos de 0.5B–3B mencionados no planejamento original não se aplicam. Para telemetria PMU tokenizada (vocabulário ~200 tokens, sequências de ~600 tokens), modelos acima de 20M provavelmente são overkill. O experimento de scaling determinará o sweet spot.

---

## 5) Comparabilidade com HARD-Lite

### Mesmo coletor, dois datasets

`src/tokenize_pmu.py` gera a partir da mesma coleta:

1. **Dataset tokenizado** (SLM): bins + ritmo → janelas de tokens achatadas
2. **Dataset contínuo normalizado** (LSTM): z-score no benigno → janelas (N, ws, n_features)

### Mesmos parâmetros

* Frame: 10ms
* Janela: 50 frames = 500ms
* Split temporal: 70% treino / 15% val / 15% teste
* Limiar: calibrado só com benigno (percentis)

---

## 6) Metodologia implementada

### Treino benign-only

* Coleta benigna em múltiplos workloads (sintético: idle + build + IO; real: configurável)
* Split temporal (não aleatório) para drift realista
* SLM: next-token prediction (cross-entropy loss)
* LSTM: forecasting do próximo frame (L1 loss)

### Score e decisão

* **SLM:** NLL/perplexity por janela → EMA → 2 limiares (warning p90 / anomaly p99)
* **LSTM:** erro L1 por feature → WMVE (pesos AUC) → EMA → 2 limiares
* **CUSUM:** detecta mudança de regime nos scores suavizados
* **Multi-resolução:** scores em 3 escalas temporais, fundidos por max/weighted_avg/vote

### Portabilidade (LoRA)

* Treinar modelo base em CPU A
* Coletar benigno em CPU B
* Fine-tune apenas LoRA adapters (~0.1% dos parâmetros)
* Salvar/carregar adapters separadamente (`save_lora()` / `load_lora()`)

---

## 7) Métricas reportadas

| Métrica | Onde |
|---|---|
| AUC-ROC | `compute_metrics()` |
| PR-AUC | `compute_metrics()` |
| TPR@FPR fixo (0.001, 0.01, 0.05) | `compute_metrics()` |
| FPR/dia (warning e anomaly) | `compute_metrics()` — convertido para alarmes/dia |
| ToD p50/p95 (warning e anomaly) | `_compute_tod()` — em ms |
| CPU% | `measure_overhead()` — latência / duração da janela |
| Latência p95 (ms) | `measure_overhead()` |
| Parâmetros (M) | `measure_overhead()` |
| Bootstrap CI 95% | `bootstrap_metrics()` |
| CUSUM ToD | `CUSUM.detect_with_reference()` |

---

## 8) Organização do repositório

```
thiago/
├── configs/                    # Hiperparâmetros (YAML)
│   ├── default.yaml            # Config completa padrão
│   ├── fast_debug.yaml         # Teste rápido
│   ├── ablation_no_rhythm.yaml # Ablation: sem tokens de ritmo
│   ├── ablation_no_multires.yaml # Ablation: sem multi-resolução
│   ├── ablation_no_cusum.yaml  # Ablation: sem CUSUM
│   ├── intel.yaml              # Eventos PMU Intel
│   ├── amd.yaml                # Eventos PMU AMD
│   └── model_scaling.yaml      # Experimento de tamanho do SLM
├── src/
│   ├── collect.py              # Coleta PMU + dados sintéticos
│   ├── tokenize_pmu.py         # Tokenização + multi-resolução
│   ├── model.py                # SLM + LoRA + quantização + 2 estágios
│   ├── baseline.py             # LSTM + WMVE(AUC) + EMA + 2 níveis
│   ├── train.py                # Treino (SLM, Stage 1, LoRA, LSTM)
│   ├── eval.py                 # Avaliação + overhead + bootstrap
│   └── change_point.py         # CUSUM + fusão multi-resolução
├── data/                       # Dados (gerados pelo pipeline)
├── checkpoints/                # Modelos treinados
├── results/                    # Métricas + gráficos
├── docs/
│   ├── BASELINE.md             # Definição baseline vs proposta
│   └── exemplo.md              # Este arquivo
├── run_phase0.sh               # Pipeline completo
├── requirements.txt
└── README.md
```

---

## 9) Como usar o hardware disponível

* **HPC:** Geração de dados benignos em escala, ablations em paralelo, sweeps de hiperparâmetros
* **GPUs 24GB:** Treino do SLM (todos os tamanhos cabem facilmente), LoRA fine-tuning
* **Servidor Stage 2:** Inferência contínua monitorando múltiplos hosts
* **Host Stage 1:** SLM quantizado INT8 rodando em CPU (sem GPU)

---

## 10) Próximos passos

1. ~~Fixar eventos PMU~~ → Feito (6 eventos genéricos + configs Intel/AMD)
2. ~~Definir janela 10ms/50 frames~~ → Feito
3. ~~Pipeline de dataset duplo~~ → Feito (`tokenize_pmu.py`)
4. ~~Treinar baseline LSTM e SLM~~ → Feito (`train.py`)
5. ~~Tabela inicial~~ → Feito (`eval.py`)
6. **Rodar com dados reais** (coleta em bare metal com perf)
7. **Testar portabilidade** (treino Intel → teste AMD com LoRA)
8. **Experimento de scaling** (curva tamanho × acurácia × latência)
9. **Ablations** (ritmo, multi-res, CUSUM)
10. **Redigir paper** com tabelas e gráficos gerados
