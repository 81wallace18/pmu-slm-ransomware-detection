# PMU-SLM: Hardware Language Model para Detecção de Ransomware

Detecção de ransomware em tempo real via Hardware Performance Counters (PMU) usando um Small Language Model (SLM) benign-only. Compara com baseline LSTM (estilo HARD-Lite).

## Estrutura

```
thiago/
├── configs/
│   └── default.yaml           # Hiperparâmetros centralizados
├── src/
│   ├── collect.py             # Coleta PMU (perf stat) + dados sintéticos
│   ├── tokenize_pmu.py        # Tokenização: bins quantílicos + tokens de ritmo
│   ├── model.py               # SLM (transformer decoder-only) + LoRA + 2 estágios
│   ├── baseline.py            # LSTM forecaster + WMVE(AUC) + EMA + 2 níveis
│   ├── train.py               # Treino benign-only (SLM, Stage 1, LoRA, LSTM)
│   ├── eval.py                # Avaliação completa + overhead + bootstrap
│   └── change_point.py        # CUSUM + fusão multi-resolução
├── run_phase0.sh              # Pipeline completo (dados sintéticos)
├── requirements.txt
└── docs/
    ├── BASELINE.md            # Definição do paper base e proposta
    └── exemplo.md             # Arquitetura e stack
```

## Requisitos

```bash
pip install -r requirements.txt
```

Necessita PyTorch >= 2.1 com suporte CUDA (opcional, funciona em CPU).

## Uso Rápido (Fase 0 — Dados Sintéticos)

```bash
./run_phase0.sh
```

Executa o pipeline completo com dados sintéticos:
1. Gera dados benignos + maliciosos simulados
2. Tokeniza (bins quantílicos + tokens de ritmo + multi-resolução)
3. Treina SLM (benign-only, next-token prediction)
4. Treina SLM Stage 1 (compacto, para on-host)
5. Treina LSTM baseline (HARD-Lite: forecasting + WMVE + EMA)
6. Avalia e compara (2 níveis, CUSUM, overhead, bootstrap CI)

Resultados em `results/`.

## Uso com Dados Reais

### Coleta PMU (requer root e `perf`)

```bash
# Benigno: múltiplos workloads
sudo python3 src/collect.py real --label benign --workload idle --duration 300
sudo python3 src/collect.py real --label benign --workload build --duration 300
sudo python3 src/collect.py real --label benign --workload browser --duration 300

# Malicioso (em VM isolada)
sudo python3 src/collect.py real --label malicious --workload ransomware_x --duration 60
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

# LoRA (adaptar para nova CPU/workload)
python3 src/train.py --model slm-lora --data-dir data/tokenized \
    --base-checkpoint checkpoints/slm_best.pt
```

### Avaliação

```bash
python3 src/eval.py --data-dir data/tokenized --checkpoint-dir checkpoints
```

## Arquitetura

### Pipeline SLM (proposta)

```
PMU (10ms) → Tokenização (bins + ritmo) → SLM benign-only → NLL/perplexity
                                                              ↓
                                                    EMA + 2 níveis (warning/anomaly)
                                                    CUSUM change-point
                                                    Multi-resolução (10/100/1000ms)
```

### 2 Estágios

- **Stage 1 (on-host)**: SLM compacto quantizado (INT8). Alta sensibilidade (warning). Funciona desconectado.
- **Stage 2 (servidor)**: SLM maior. Confirma anomalias do Stage 1. Reduz FPs.

### Baseline LSTM (HARD-Lite)

```
PMU (10ms) → Normalização z-score → LSTM forecasting → Erro L1 por feature
                                                          ↓
                                                 WMVE (pesos AUC por feature)
                                                 EMA + 2 níveis (warning/anomaly)
```

## Métricas

| Métrica | Descrição |
|---------|-----------|
| AUC-ROC | Qualidade geral de discriminação |
| PR-AUC | Precisão vs recall (melhor para classes desbalanceadas) |
| TPR@FPR | Recall em taxas fixas de falso positivo |
| FPR/dia | Alarmes falsos por dia em operação benigna contínua |
| ToD p50/p95 | Tempo até detecção (ms) com mediana e p95 |
| CPU% | Percentual de CPU consumido pela inferência |
| Latência p95 | Tempo de inferência por janela (ms) |
| Bootstrap CI | Intervalo de confiança 95% via reamostragem |

## Portabilidade (LoRA)

Para adaptar o modelo a uma nova CPU ou workload:

1. Coletar dados benignos no novo ambiente
2. Treinar LoRA adapter (`--model slm-lora`)
3. Salva apenas os adapters (~0.1% dos parâmetros)

## Configuração

Todos os hiperparâmetros estão em `configs/default.yaml`:
- Coleta: eventos PMU, intervalo, duração
- Tokenização: bins, janela, ritmo
- Modelos: SLM (principal e Stage 1), LSTM, LoRA
- Avaliação: limiares, CUSUM, multi-resolução, bootstrap, overhead
