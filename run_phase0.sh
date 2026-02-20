#!/usr/bin/env bash
# ==============================================================================
# Fase 0: Prova de Conceito — Pipeline Completo
#
#   1. Gera dados sintéticos (benigno + malicioso)
#   2. Tokeniza e prepara datasets (com tokens de ritmo + multi-resolução)
#   3. Treina SLM (benign-only)
#   4. Treina SLM Stage 1 (compacto, para on-host)
#   5. Treina LSTM baseline (benign-only)
#   6. Avalia e compara (EMA, 2 níveis, CUSUM, overhead, bootstrap)
#
# Uso:
#   chmod +x run_phase0.sh
#   ./run_phase0.sh
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")"

echo "=========================================="
echo " FASE 0: Prova de Conceito"
echo "=========================================="
echo ""

# 1. Gerar dados sintéticos
echo "[1/6] Gerando dados sintéticos..."
python src/collect.py synthetic --output-dir data
echo ""

# 2. Tokenizar e preparar datasets
echo "[2/6] Tokenizando (bins + ritmo + multi-resolução)..."
python src/tokenize_pmu.py \
    --benign "data/benign_synthetic_*.csv" \
    --malicious "data/malicious_synthetic_*.csv" \
    --output-dir data/tokenized
echo ""

# 3. Treinar SLM (modelo principal / Stage 2)
echo "[3/6] Treinando SLM (benign-only)..."
python src/train.py --model slm --data-dir data/tokenized --output-dir checkpoints
echo ""

# 4. Treinar SLM Stage 1 (compacto)
echo "[4/6] Treinando SLM Stage 1 (compacto)..."
python src/train.py --model slm-stage1 --data-dir data/tokenized --output-dir checkpoints
echo ""

# 5. Treinar LSTM baseline
echo "[5/6] Treinando LSTM baseline (benign-only)..."
python src/train.py --model lstm --data-dir data/tokenized --output-dir checkpoints
echo ""

# 6. Avaliar e comparar
echo "[6/6] Avaliando (EMA + 2 níveis + CUSUM + overhead + bootstrap)..."
python src/eval.py \
    --data-dir data/tokenized \
    --checkpoint-dir checkpoints \
    --output-dir results
echo ""

echo "=========================================="
echo " FASE 0 COMPLETA"
echo "=========================================="
echo ""
echo "Resultados em: results/"
echo "  - results/results.json            (métricas + overhead + CI)"
echo "  - results/slm_histogram.png       (histograma SLM com 2 limiares)"
echo "  - results/lstm_histogram.png      (histograma LSTM com 2 limiares)"
echo "  - results/roc_comparison.png      (ROC comparativo)"
echo "  - results/pr_comparison.png       (PR comparativo)"
echo "  - results/slm_timeline_cusum.png  (score + CUSUM)"
echo "  - results/lstm_timeline_cusum.png (score + CUSUM)"
