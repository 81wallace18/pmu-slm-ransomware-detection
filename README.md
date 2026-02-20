# PMU-SLM: Hardware Language Model para Detecção de Ransomware

Detecção de ransomware em tempo real via Hardware Performance Counters (PMU) usando um Small Language Model (SLM) benign-only. Compara com baseline LSTM (estilo HARD-Lite).

## Estrutura

```
├── configs/
│   ├── default.yaml              # Config completa padrão
│   ├── fast_debug.yaml           # Teste rápido (5 epochs, sem bootstrap)
│   ├── ablation_no_rhythm.yaml   # Ablation: sem tokens de ritmo
│   ├── ablation_no_multires.yaml # Ablation: sem multi-resolução
│   ├── ablation_no_cusum.yaml    # Ablation: sem CUSUM
│   ├── intel.yaml                # Eventos PMU Intel
│   ├── amd.yaml                  # Eventos PMU AMD
│   └── model_scaling.yaml        # Experimento de tamanho do SLM
├── src/
│   ├── collect.py                # Coleta PMU (perf stat) + dados sintéticos
│   ├── tokenize_pmu.py           # Tokenização: bins + ritmo + multi-resolução
│   ├── model.py                  # SLM + LoRA + quantização INT8 + 2 estágios
│   ├── baseline.py               # LSTM + WMVE(AUC) + EMA + 2 níveis
│   ├── train.py                  # Treino benign-only (SLM, Stage 1, LoRA, LSTM)
│   ├── eval.py                   # Avaliação + overhead + bootstrap CI
│   └── change_point.py           # CUSUM change-point + fusão multi-resolução
├── data/                         # Dados gerados pelo pipeline
├── checkpoints/                  # Modelos treinados
├── results/                      # Métricas + gráficos
├── docs/
│   ├── BASELINE.md               # Definição baseline vs proposta
│   └── exemplo.md                # Arquitetura detalhada do sistema
├── run_phase0.sh                 # Pipeline completo (dados sintéticos)
└── requirements.txt
```

## Requisitos

```bash
pip install -r requirements.txt
```

PyTorch >= 2.1 com suporte CUDA (opcional, funciona em CPU).

## Uso Rápido (Fase 0 — Dados Sintéticos)

```bash
./run_phase0.sh
```

Pipeline completo:
1. Gera dados sintéticos (benigno + malicioso)
2. Tokeniza (bins quantílicos + tokens de ritmo + multi-resolução)
3. Treina SLM principal (Stage 2)
4. Treina SLM Stage 1 (compacto, quantizável)
5. Treina LSTM baseline (HARD-Lite)
6. Avalia e compara (EMA, 2 níveis, CUSUM, overhead, CPU%, bootstrap CI)

Para teste rápido (debug):
```bash
./run_phase0.sh  # editar para usar --config configs/fast_debug.yaml
```

## Uso com Dados Reais

### Coleta PMU (requer root + `perf`)

```bash
# Benigno: múltiplos workloads
sudo python3 src/collect.py real --label benign --workload idle --duration 300
sudo python3 src/collect.py real --label benign --workload build --duration 300
sudo python3 src/collect.py real --label benign --workload browser --duration 300

# Malicioso (em VM isolada)
sudo python3 src/collect.py real --label malicious --workload ransomware_x --duration 60

# Para Intel com eventos extras:
sudo python3 src/collect.py real --label benign --workload idle --duration 300 --config configs/intel.yaml
```

### Tokenização

```bash
python3 src/tokenize_pmu.py \
    --benign "data/benign_*.csv" \
    --malicious "data/malicious_*.csv" \
    --output-dir data/tokenized
```

### Treino

```bash
# SLM principal (Stage 2 / servidor)
python3 src/train.py --model slm --data-dir data/tokenized

# SLM Stage 1 (compacto, para on-host)
python3 src/train.py --model slm-stage1 --data-dir data/tokenized

# LSTM baseline
python3 src/train.py --model lstm --data-dir data/tokenized

# LoRA: adaptar para nova CPU/workload
python3 src/train.py --model slm-lora --data-dir data/tokenized \
    --base-checkpoint checkpoints/slm_best.pt
```

### Avaliação

```bash
python3 src/eval.py --data-dir data/tokenized --checkpoint-dir checkpoints
```

### Ablations

```bash
# Sem tokens de ritmo
python3 src/train.py --model slm --config configs/ablation_no_rhythm.yaml --output-dir checkpoints/no_rhythm
python3 src/eval.py --config configs/ablation_no_rhythm.yaml --output-dir results/no_rhythm

# Sem multi-resolução
python3 src/eval.py --config configs/ablation_no_multires.yaml --output-dir results/no_multires

# Sem CUSUM
python3 src/eval.py --config configs/ablation_no_cusum.yaml --output-dir results/no_cusum
```

## Arquitetura

### Pipeline SLM (proposta)

```
PMU (10ms) → Tokenização (bins + ritmo) → SLM benign-only → NLL/perplexity
                                                              ↓
                                                    EMA + 2 níveis (warning/anomaly)
                                                    CUSUM change-point detection
                                                    Multi-resolução (10/100/1000ms)
```

### 2 Estágios

| | Stage 1 (on-host) | Stage 2 (servidor) |
|---|---|---|
| Modelo | SLM compacto (~0.2M) | SLM principal (~0.8M) |
| Quantização | INT8 | FP32 |
| Sensibilidade | Alta (warning, p90) | Baixa FPR (anomaly, p99) |
| Rede | Funciona desconectado | Recebe janelas suspeitas |
| Função | Early warning | Confirmação |

### Baseline LSTM (HARD-Lite)

```
PMU (10ms) → z-score → LSTM forecasting → Erro L1 por feature
                                              ↓
                                    WMVE (pesos AUC por feature)
                                    EMA + 2 níveis (warning/anomaly)
```

## Métricas

| Métrica | Descrição | Onde |
|---|---|---|
| AUC-ROC | Discriminação geral | `compute_metrics()` |
| PR-AUC | Precisão vs recall | `compute_metrics()` |
| TPR@FPR | Recall em FPR fixo (0.001, 0.01, 0.05) | `compute_metrics()` |
| FPR/dia | Alarmes falsos por dia (warning e anomaly) | `compute_metrics()` |
| ToD p50/p95 | Tempo até detecção em ms | `_compute_tod()` |
| CPU% | % de CPU por janela de inferência | `measure_overhead()` |
| Latência p95 | Tempo de inferência por janela (ms) | `measure_overhead()` |
| Bootstrap CI | Intervalo de confiança 95% | `bootstrap_metrics()` |
| CUSUM ToD | Change-point no score malicioso | `CUSUM.detect_with_reference()` |

### Tabela Final

A avaliação gera automaticamente a tabela comparativa:

```
Métrica                  LSTM            SLM          Delta        %
──────────────────────────────────────────────────────────────────────
FPR/day (anomaly)       X.XXXX         X.XXXX       +X.XXXX    +X.X%
ToD p50 (anomaly)       X.XXXX         X.XXXX       +X.XXXX    +X.X%
ToD p95 (anomaly)       X.XXXX         X.XXXX       +X.XXXX    +X.X%
AUC-ROC                 X.XXXX         X.XXXX       +X.XXXX    +X.X%
PR-AUC                  X.XXXX         X.XXXX       +X.XXXX    +X.X%
CPU%                    X.XXXX         X.XXXX       +X.XXXX    +X.X%
Latência p95 (ms)       X.XXXX         X.XXXX       +X.XXXX    +X.X%
```

## Portabilidade (LoRA)

```
CPU Intel (treino) → Modelo base → CPU AMD (coleta benigna) → LoRA adapter → Modelo adaptado
                                                                (~0.1% params)
```

1. Treinar modelo base em CPU A
2. Coletar dados benignos no novo ambiente (CPU B)
3. `python3 src/train.py --model slm-lora --base-checkpoint checkpoints/slm_best.pt`
4. Adapters salvos separadamente (`lora_adapters.pt`)

## Configs

| Config | Uso |
|---|---|
| `default.yaml` | Experimento completo com todas as features |
| `fast_debug.yaml` | Validação rápida do pipeline (5 epochs, sem bootstrap) |
| `ablation_no_rhythm.yaml` | Medir impacto dos tokens de ritmo |
| `ablation_no_multires.yaml` | Medir impacto da multi-resolução |
| `ablation_no_cusum.yaml` | Medir impacto do CUSUM |
| `intel.yaml` | Eventos PMU Intel (+ específicos comentados) |
| `amd.yaml` | Eventos PMU AMD (+ específicos comentados) |
| `model_scaling.yaml` | Curva tamanho do SLM vs acurácia vs latência |
