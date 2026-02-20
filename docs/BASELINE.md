# BASELINE

Fechado — vamos “montar” o **paper base** e já definir **o nosso** com as diferenças, mantendo o formato que você pediu.
* * *
# 1) Paper base (HARD-Lite)
## Análise
Framework de **detecção em tempo real** de ransomware via **Hardware Performance Counters (PMU)**, usando **semi-supervisionado (treino só benigno)** e modelagem temporal. Arquitetura **hierárquica**: coleta no host monitorado e **classificação offload** em servidor, com ensemble e suavização para reduzir falsos positivos.
## Problema
*   Assinaturas estáticas falham para variantes novas.
*   Monitoramento “alto nível” (arquivos/rede) pode ser burlado/ruidoso.
*   Precisa detectar **rápido** com **baixo overhead** e com o detector mais protegido do atacante.
## Motivação
*   Ransomware cresce e muda rápido.
*   PMU captura sinais micro/arquiteturais difíceis de “maquiar” sem custo.
*   Offload aumenta segurança do classificador e escalabilidade.
## Objetivos
*   Detectar ransomware (incl. _zero-day_) por **desvio do comportamento benigno**.
*   Operar em **tempo real** com **boa acurácia** e **poucos FPs**.
*   Ser escalável via servidor central.
## Relevância
*   “Deployability”: pensa em produção (overhead, separação de módulos, múltiplas máquinas, VM).
*   Discute ameaças ao detector (ataques/adversarial).
## Aplicação
Proteção de endpoints/servidores (Linux principalmente) e **detecção em VMs** (monitorando via host).
## Cenário
*   Coleta PMU em alta frequência (ex.: 10ms).
*   Envio em blocos (ex.: 500ms) via canal seguro, sem armazenar log no host.
*   Classificador roda no servidor (mais protegido).
## Metodologia
*   **LSTM forecasting**: prevê próximo passo dado uma janela (ex.: N=50).
*   **Score**: erro de predição por feature.
*   **Limiar**: por feature (ex.: percentil 95 do benigno).
*   **WMVE** (ensemble por voto ponderado) + **EMA** (suavização) + 2 estados de alerta (warning/anomaly).
*   Feature selection guiada por AUC-ROC por evento/contador.
## Métricas
*   AUC-ROC (seleção de eventos).
*   **Time to Detection (ToD)**: até warning/anomaly.
*   FPR/FNR (principalmente em benigno prolongado).
*   Overhead (CPU/memória/impacto em benchmark) e escalabilidade (qtd de modelos/hosts).
## Contribuição
*   Pipeline completo em tempo real PMU→LSTM→ensemble/smoothing.
*   Arquitetura distribuída (offload) com foco em overhead/segurança/escalabilidade.
*   Evidência em Intel/AMD e caso de VM.
## Trabalhos futuros (do paper)
*   Otimizar classificador (ex.: quantização inteira).
*   Otimizar coleta/relay (reduzir overhead de processamento/comunicação).
*   Expandir para outros ataques/malwares e fortalecer segurança do deployment.
* * *
# 2) Nosso paper (proposta) — “o que vamos fazer de diferente”
A ideia é ficar **comparável** (mesmo cenário/coleta/janelas), mas trocar o “miolo” e atacar pontos fracos reconhecidos no estado da arte.
## Análise
Vamos propor um **SLM como “Hardware Language Model”**: transformar telemetria (PMU e derivados) em **tokens** e usar um **SLM benign-only**. A detecção vira **NLL/perplexity + change detection**, com um desenho **2-estágios** (on-host leve + servidor confirmador).
## Problema
Além do problema do HARD-Lite, vamos endereçar explicitamente:
*   **Generalização fraca** (muda CPU/workload → FP explode).
*   **Robustez em produção** (cargas concorrentes, ruído, _concept drift_).
*   **Portabilidade** (eventos PMU diferentes entre Intel/AMD/microarquiteturas).
## Motivação
*   SLMs pequenos + LoRA/quantização permitem **adaptação rápida** e **inferência barata**.
*   Tokenização + perplexity dá um score mais “universal” do que erro de um forecast LSTM específico.
*   Estado da arte em HPC-based detection sofre com “cautionary tales” de generalização; vamos medir isso de forma dura.
## Objetivos
*   Melhorar **FPR em benigno longo** (meta principal de produção).
*   Melhorar **ToD** sem sacrificar FPR.
*   Mostrar **portabilidade** (Intel↔AMD) via adapters/normalização.
*   Entregar um modo “lite de verdade”: **early warning on-host**.
## Relevância
*   Contribuição alinhada ao que revisores cobram hoje: **robustez, portabilidade, avaliação realista, custo**.
*   Facilita integração em SOC: score + explicação textual opcional (sem virar “caixa preta” total).
## Aplicação
Mesmas aplicações do HARD-Lite, mas com:
*   **Modo desconectado** (on-host alerta mesmo sem servidor/rede).
*   Melhor suporte a ambientes com workloads variáveis.
## Cenário
*   Mantemos: PMU em 10ms, janelas e streaming (para comparabilidade).
*   Acrescentamos: **stage 1 no host** (SLM quantizado) + **stage 2 no servidor** (SLM maior/ensemble) para reduzir FPs.
## Metodologia
**Delta direto vs HARD-Lite**
1. **Representação**: time-series → **tokens** (bins por quantil/z-score + tokens de “ritmo”/tempo).
2. **Modelo**: LSTM forecasting → **SLM benign-only** (next token / masked).
3. **Score**: erro L1 por feature → **NLL/perplexity** por janela + smoothing.
4. **Decisão**: manter EMA/2-níveis para ficar comparável, mas adicionar **change-point** e **multi-resolução** (10ms/100ms/1s) para robustez.
5. **Portabilidade**: usar **famílias de eventos** + normalização + **LoRA adapters** por CPU/workload.
## Métricas (e onde vamos buscar “X% melhor”)
**Mesmas do baseline + extras de estado da arte**
*   Qualidade: AUC-ROC e **PR-AUC**, **TPR@FPR fixo**.
*   Produção: **FPR por hora/dia** em benigno longo (ponto central).
*   Tempo: ToD (warning/anomaly) com mediana e p95.
*   Custo: latência de inferência (p95), CPU/RAM, tráfego rede (se offload).
*   Portabilidade: treino em CPU A / teste em CPU B; e adaptação com LoRA (quanto dado/tempo precisa).
*   Robustez: cargas concorrentes e drift (antes/depois de adaptação).
## Contribuição (o “claim” do nosso paper)
*   **Primeiro (ou um dos primeiros) pipeline PMU→SLM benign-only** para ransomware com avaliação realista.
*   **2-estágios** (on-host quantizado + servidor confirmador) com trade-off claro.
*   **Portabilidade** via normalização + adapters (Intel/AMD).
*   **Robustez** via multi-resolução + change detection (reduz impacto de evasões baseadas em tempo e de ruído de workload).
## Trabalhos futuros (nossos)
*   Federated adaptation de adapters/LoRA entre máquinas (sem centralizar dados).
*   Explicabilidade melhor (gerar sumário técnico consistente por evento/janela).
*   Extensão para outras classes de ataques (sem retrabalho total do pipeline).
* * *
## Como garantir comparação “justa” e poder dizer “somos X% melhores”
*   **Mesmo coletor / mesma taxa / mesmas janelas / mesmo conjunto de ataques**.
*   Limiar calibrado **só com benigno** em ambos.
*   Reportar: diferença absoluta + % (e idealmente IC via bootstrap).
*   Tabela final: Baseline HARD-Lite vs Ours em **FPR/day**, **ToD p50/p95**, **PR-AUC**, **CPU%** e **latência p95**.