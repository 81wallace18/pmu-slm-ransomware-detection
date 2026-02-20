Perfeito — vou te passar **a arquitetura do sistema + stack + como organizar o repositório + fluxo de treino/inferência + como garantir comparabilidade com o HARD-Lite**, **sem escrever código**.

---

## 1) Arquitetura geral que vamos usar

### Visão macro (2 estágios, “Lite de verdade”)

**Host monitorado (Stage 1 – leve / resiliente)**

* Coleta PMU (10ms) + agregações simples
* Tokenização rápida (bins) + score de anomalia (NLL/perplexity do SLM **quantizado**)
* Se “suspeito”: dispara **warning** local e envia janela para o servidor

**Servidor (Stage 2 – confirmador / mais forte)**

* Recebe janelas suspeitas + (opcional) amostras benignas periódicas
* Roda SLM maior (ou ensemble) + multi-resolução + change-point
* Decide **anomaly** com baixa taxa de falso positivo
* Envia evento para SIEM/SOC (Wazuh, etc.)

**Por que isso é bom academicamente**

* Você mantém o “espírito HARD-Lite” (offload + real-time) **mas entrega robustez** (alerta mesmo se rede cair) e “SLM-first”.

---

## 2) Componentes (módulos) do projeto

### (A) Collector Agent (host)

Objetivo: coletar e preparar o mínimo possível, com overhead baixo.

**Funções**

* Ler PMU via `perf_event_open`/`perf` (Linux) com um conjunto fixo de eventos
* Agregar por janela curta (ex.: 10ms) e formar *frames*
* Montar *windows* (ex.: 50 frames = 500ms) para inferência
* Fazer normalização leve (z-score/robust) e discretização (bins)
* Rodar Stage 1 (modelo pequeno quantizado) e sinalizar

**Decisões importantes**

* Coleta por **CPU-core** vs agregado total (agregado é mais simples e mais “lite”; por core dá mais sinal mas aumenta custo).
* Associar com processo/cgroup (opcional) — se você quiser “per-process detection”, entra eBPF/cgroups, mas isso complica e pode virar outro paper.

### (B) Tokenizer / Serializer (shared lib)

Objetivo: transformar telemetria num “idioma” estável para o SLM.

**Abordagem recomendada**

* Para cada evento: transformar valor em **bin** (ex.: 0–63) por quantis do benigno (mais robusto que min-max).
* Criar tokens do tipo: `E5_B12` (evento 5, bin 12) ou formato mais compacto.
* Incluir tokens de **ritmo/tempo** (ex.: “burst”, “steady”) se você for atacar evasão por delay.

**Resultado:** janela vira uma sequência de tokens (tipo log estruturado).

### (C) Inference Engine (Stage 1 e Stage 2)

* Stage 1: runtime ultraleve no host
* Stage 2: runtime robusto no servidor (com GPU)

### (D) Trainer (offline)

* Treino benign-only do SLM
* Calibração de limiar (threshold) por host/workload
* Export: checkpoints + LoRA adapters + quantização

### (E) Evaluation Harness (pra comparar com HARD-Lite)

* Rodar baseline LSTM e seu SLM **com os mesmos dados**
* Geração de tabelas de métricas e gráficos (ToD, FPR/day, PR-AUC etc.)
* Ablations (tokenização, multiresolução, quantização, etc.)

---

## 3) Stack / tecnologias recomendadas (sem código)

### Coleta PMU (Linux)

* `perf_event_open` (baixo nível) ou `perf`/`libpfm` (facilita listar eventos)
* Se quiser isolação por workload/cgroup: **eBPF** (mas eu só colocaria se virar objetivo explícito)

### Linguagens (pragmático)

* **Collector**: Rust ou Go (latência baixa, binário estável, fácil deploy)
* **Pipeline/treino/avaliação**: Python (PyTorch/Transformers)

### Transporte

* gRPC + Protobuf (simples e rápido) **ou** NATS (se quiser fila leve)
* SSH tunnel como no paper é ok, mas gRPC+mTLS é mais “produto”

### Armazenamento offline (dataset)

* Parquet (colunar, eficiente) + metadata (json/yaml)
* Logs por máquina/workload/versão do coletor

### Treino/inferência (SLM)

* Treino: PyTorch + HuggingFace Transformers
* Fine-tuning: **LoRA/QLoRA** (pra adaptar por CPU/workload sem custo absurdo)
* Quantização Stage 1:

  * `llama.cpp` (GGUF) ou
  * ONNX Runtime (se preferir pipeline mais “ML deploy”)
* Stage 2 servidor:

  * vLLM (se você quiser throughput alto) ou inferência HF normal

---

## 4) Escolha do SLM (tamanho e por quê, com teu hardware)

Você tem GPUs 24GB + nó com 32GB VRAM + HPC gigante. O sweet spot:

### Modelos alvo (faixa)

* **1B–3B parâmetros** como “modelo central” (treino/servidor)
* **0.3B–1B** como “modelo stage 1” (host quantizado)

### Por que essa faixa

* 24GB VRAM é excelente pra **LoRA/QLoRA** em 1.5B–3B com batch/seq adequados
* Stage 1 precisa ser rápido e barato; 4-bit em 0.5B–1B costuma ficar ótimo

**Estratégia boa de paper**

* Mostrar curva: 0.5B vs 1.5B vs 3B (acurácia × custo × latência)

---

## 5) Como vamos reproduzir o HARD-Lite (baseline) sem bagunçar o projeto

### Regra: mesmo coletor, dois pipelines

Você coleta PMU uma vez, e gera dois “datasets” derivados:

1. **Baseline HARD-Lite dataset**

   * Normalizado contínuo (float) por evento
   * Janelas N=50 (igual paper)
   * Serve pro LSTM forecasting

2. **Ours SLM dataset**

   * Discretizado/tokenizado
   * Mesmas janelas e timestamps
   * Serve pro SLM (NLL/perplexity)

Assim você garante comparabilidade e não “inventa” outra coleta.

---

## 6) Metodologia do nosso pipeline (o miolo)

### Treino benign-only

* Coletar benigno em múltiplos workloads (idle + dev + build + browser + IO pesado etc.)
* Split temporal: treino / validação / teste (pra drift realista)
* Treinar SLM para prever tokens (next-token) ou masked tokens

### Score e limiar

* Score por janela: **NLL/perplexity**
* Limiar: percentil no benigno (p95/p99) + calibrado por FPR alvo
* Suavização: EMA (pra ficar comparável ao HARD-Lite)
* Change-point: detectar “mudança de regime” no score (pra delay/padding e ruído)

### Multi-resolução (se entrar como contribuição)

* Rodar scores em 10ms/100ms/1s e fundir (max, média ponderada, ou voto)

---

## 7) Métricas e onde você provavelmente vai “ganhar %”

Se você fizer bem feito, onde costuma dar salto real:

* **FPR/day** (produção) ↓ grande
* **TPR@FPR fixo** ↑
* **ToD p95** ↓ (menos “demora nos casos difíceis”)
* **Overhead host** ↓ com quantização/Stage 1
* **Portabilidade** ↑ com adapters/normalização (Intel↔AMD)

---

## 8) Organização do repositório (arquitetura de engenharia)

Sugestão de pastas (pra ficar “paper-ready”):

* `collector/` (agent do host)
* `schemas/` (protobuf + contratos)
* `pipeline/`

  * `preprocess/` (windowing, normalização, tokenização)
  * `datasets/` (parquet builders)
* `models/`

  * `baseline_lstm/`
  * `slm_detector/`
* `train/` (scripts de treino e calibração)
* `inference/`

  * `stage1_host/` (quantizado)
  * `stage2_server/`
* `eval/` (métricas, tabelas, ablações)
* `deploy/` (docker, systemd, helm se quiser)
* `docs/` (figuras do paper, diagramas, config)

---

## 9) Como usar teus servidores/HPC de forma inteligente

* **HPC grande**: geração de benigno (workloads), preprocess pesado, hyperparam sweeps, ablações
* **GPUs 24GB**: LoRA/QLoRA do SLM 1B–3B
* **Servidor stage2**: inferência contínua (monitorando múltiplos hosts)
* **Stage1**: roda no host com quantização (CPU-friendly)

---

## 10) Decisões que eu tomaria agora (pra você avançar rápido)

1. Fixar um **conjunto mínimo de eventos PMU** (familias: cycles, instructions, cache, branch, tlb, stalls)
2. Definir janela: **10ms frame**, **50 frames/window** (igual paper pra comparar)
3. Implementar pipeline de dataset duplo (float e token)
4. Treinar baseline LSTM e o SLM pequeno (0.5B–1.5B)
5. Fechar tabela inicial: ToD, FPR/day, PR-AUC, overhead

---

Se você me disser **qual Linux e como você pretende coletar (bare metal vs VM vs ambos)**, eu ajusto as escolhas (por exemplo, perf no host vs inside VM, e quais eventos são mais “portáveis”). Mas mesmo sem isso, a arquitetura acima já está consistente e pronta pra virar design do sistema + seção de metodologia do paper.
