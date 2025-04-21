# Detecção de atividades suspeitas em sistemas

O Objetivo aqui é treinar um modelo de ML para detectar atividades suspeitas em redes. Para isso, iremos usar o dataset CIDDS.

***O que é: CIDDS é uma coleção de conjuntos de dados (datasets) projetados para a avaliação e pesquisa de Sistemas de Detecção de Intrusão (IDS - Intrusion Detection Systems).***

***Origem: Foi desenvolvido na Universidade de Coburg (Hochschule Coburg), na Alemanha.***

***Propósito: O objetivo principal do CIDDS é fornecer dados de tráfego de rede realistas e rotulados (indicando qual tráfego é normal e qual é malicioso/ataque) para que pesquisadores e desenvolvedores possam:***

- *Treinar e testar algoritmos de detecção de intrusão*.
- *Comparar o desempenho de diferentes sistemas IDS*.
- *Desenvolver novas técnicas de detecção*.

### **Características:**

- **Dados Realistas**: Tenta simular o tráfego de uma rede de pequena empresa.
- **Rótulos**: Cada fluxo de tráfego é classificado como normal, suspeito ou pertencente a um tipo específico de ataque (como DoS, Port Scan, Brute Force, etc.).
- **Diversidade**: Inclui diferentes tipos de ataques e tráfego normal gerado por usuários simulados.
- **Versões**: Existem diferentes versões, como o CIDDS-001, que é o mais conhecido.

### **Importância**

É uma ferramenta valiosa para a comunidade de segurança cibernética, pois datasets públicos e bem rotulados são essenciais para o avanço da pesquisa em detecção de intrusões.

> Obs: Nós vamos usar uma fração da versão CIDDS-001, referente somente a uma semana (uma semana bem turbulenta).

### Dicionário de Dados - CIDDS-001

- **session_start_time**: Timestamp em que a sessão de rede começou.
- **duration**: Duração da sessão (em segundos).
- **network_protocol**: Protocolo de comunicação usado (TCP, UDP, ICMP, etc.).
- **source_ip_addres**: Endereço IP de origem do tráfego.
- **source_port**: Porta de origem do tráfego.
- **dest_ip_address**: Endereço IP de destino do tráfego.
- **dest_port**: Porta de destino do tráfego.
- **total_packets_used**: Número total de pacotes transmitidos na sessão.
- **bytes_flow**: Quantidade total de bytes transferidos.
- **flows**: Quantidade de fluxos (connections) relacionados à sessão.
- **network_flags**: Sinalizadores usados no protocolo (como SYN, ACK, FIN).
- **tos**: Tipo de Serviço (Type of Service) – define prioridade e tratamento do pacote na rede.
- **class**: Rótulo de classificação da conexão, podendo ser 'normal' ou 'attack'.
- **attack_type**: Tipo de ataque detectado (por exemplo: portscan, bruteforce, etc.).
- **attack_id**: Identificador único do ataque.
- **attack_description**: Descrição detalhada do ataque, se aplicável.

---

### Entendendo as Flags:


| Flag | Significado         |
|------|---------------------|
| S    | SYN (início de conexão)     |
| F    | FIN (fim de conexão)        |
| R    | RST (reset)         |
| P    | PSH (push)          |
| A    | ACK (acknowledge)   |
| U    | URG (urgent)        |
| .    | "nada" / sem flag nessa posição |

Exemplo:
- `.AP.SF` → conexão com ACK, PSH, SYN, FIN. Meio estranha, parece agressiva.  
- `....S.` → só SYN → geralmente início de conexão TCP.  
- `......` → nenhum flag → estranho ou inválido.

---
# Resultados dos Modelos

## Supervisionado (RandomForestClassifier)

### **Confusion Matrix** 

|                   | Predito: 0 (Normal) | Predito: 1 (Ataque) |
|-------------------|---------------------|----------------------|
| **Real: 0 (Normal)** | 60 (TN)              | 14.806 (FP)           |
| **Real: 1 (Ataque)** | 270 (FN)             | 30.264 (TP)           |

- **Falsos Negativos (FN)**: apenas **270**, ou seja, está **pegando 99,1% dos ataques**, o que é **excelente** para um cenário de segurança.
- **Falsos Positivos (FP)**: **14.806**, ou seja, muito tráfego normal está sendo classificado como ataque, **mas prefiro errar pelo excesso de zelo, então isso está dentro da estratégia**.

---

###  **Métricas principais**

- **Recall (para ataque - classe 1)**: `0.9911` 
  → Isso é o que mais importa para a nossa estratégia. Está altíssimo.  
- **Precision**: `0.9911`  
  → Isso aqui está meio **ilusório**, pois só diz o quão certo o modelo está quando prediz "ataque", e como tem muitos FPs vindos da classe 0 (normal), o número é puxado porque o recall é altíssimo na classe 1.
- **F1-score classe 1 (ataque)**: `0.80` 
  → Um ótimo equilíbrio.
- **Accuracy**: `0.6679`  
  → Não é o mais relevante aqui, mas está ok.
- **ROC AUC**: `1.0` 
  → Provavelmente por conta da separação do score de probabilidade, o modelo consegue fazer bem a distinção entre classes em termos de ranking.

---

### **Pra finalizar**
- Este **modelo está extremamente conservador**, priorizando **detecção máxima de anomalias**.
- O **recall (99.1%)** garante que **quase nenhum ataque passa batido**.
- O trade-off é a altíssima quantidade de **falsos positivos (14.806)** — o que, dependendo da aplicação, pode significar muitos alertas desnecessários, mas se o sistema ou equipe consegue lidar com isso, **tudo certo**.

---

### **Status geral**
- **RandomForestClassifier**
- Modelo salvo com versionamento: ex: `model_v20250409_081244.pkl`
- Log estruturado com métricas salvas em JSON (rastreabilidade).

---

### **Análise das métricas**

#### **Confusion Matrix**

| Real / Previsto | Normal (0) | Ataque (1) |
|-----------------|------------|-------------|
| **Normal (0)**  | 54 (TN)    | 14.812 (FP) |
| **Ataque (1)**  | 275 (FN)   | 30.259 (TP) |

#### **Recall (classe 1)**: `0.9909`
- Detecção altíssima de ataques, o que **mantém o objetivo principal intacto**.
  
####  **Precision**: `0.671`
- Como esperado, caiu um pouco por causa do aumento nos falsos positivos.
  
####  **F1-score**: `0.80`
- Ainda excelente, combinando precisão e recall.
  
####  **Accuracy**: `0.667`
- Continua não sendo o foco, mas estável.

#### ❗ **ROC AUC Score**: `0.475`
- **Aqui está a única queda importante**. Antes estava 1.0, agora caiu bastante.
  - Isso indica que os **scores de probabilidade** do modelo não estão separando bem as classes.
  - Mas **o modelo ainda está performando bem nas decisões finais (baseado no `predict`), que é o que importa pro o caso**.

---

### Interpretação

**Treinamos um modelo que segue a filosofia de segurança que definimos: "errar por zelo"**.

**Quase todos os ataques são detectados**, com recall de 99%.

⚠️ **Alerta:** o **ROC AUC caiu**, o que pode ser um sinal de que:
- O modelo **decide bem, mas não tem boa separação nas probabilidades**.
- Isso **não é necessariamente ruim** se estiver **usando `predict()` ao invés de `predict_proba()`**.
---

## Não-Supervisionado (IsolationForest)

## 1. **IsolationForest com `contamination=0.5` (17 features)**

| Métrica        | Valor   |
|----------------|---------|
| Recall         | 0.6595  |
| Precision      | 0.8930  |
| F1-score       | 0.7587  |
| Accuracy       | 0.72    |

**Balanceado**: Este modelo **detecta bem ataques (recall)**, e ainda **com ótima precisão (0.89)**.  
**F1-score 0.75**.  

---

## 2. **IsolationForest com `contamination=0.5` e só 11 features**

| Métrica        | Valor   |
|----------------|---------|
| Recall         | 0.2762  |
| Precision      | 0.9817  |
| F1-score       | 0.4311  |
| Accuracy       | 0.51    |

**Recall despencou**: o modelo está ignorando ataques.  
**Precisão aumentou muito**: está super conservador (marca ataque só com altíssima certeza).  

Isso sugere que **as features excluídas tinham alta capacidade de separação** (provavelmente ajudavam a distinguir ataques).

---

## Conclusão do Método não-supervisionado:

### Melhor resultado até agora:
**`contamination=0.5` com 17 features**: Melhor F1, ótimo recall e precisão.

---

## Modelos em Produção com API + Mensageria para Retreino

A partir de agora, **ambos os modelos (supervisionado e não-supervisionado)** estão disponíveis **em produção via uma API com FastAPI**, hospedada no **Google Cloud Run**.

### Documentação Swagger da API
- Acesse aqui: [Swagger UI](https://anomaly-detection-api-1049636244984.us-central1.run.app/docs)

### ⚙️ Como funciona:
- A API possui **um único endpoint**, onde o usuário envia os dados de rede.
- É possível escolher **qual modelo será usado para a predição** (`RandomForest` ou `IsolationForest`).
- A resposta contém a predição: `0` ou `1` para o modelo supervisionado e `0` `-1` para o  modelo não-supervisionado.

---

## 📨 Sistema de Mensageria: Pub/Sub + Storage para Retreino

### Objetivo:
- **Todos os dados de entrada enviados à API também são armazenados** para uso futuro em retreino do modelo.

### Pipeline:
1. **A API publica os dados recebidos** em um tópico do **Google Cloud Pub/Sub**.
2. Um serviço **subscriber (também deployado no Cloud Run)** escuta esse tópico.
3. O subscriber **salva os dados recebidos em formato `.csv` em uma Google Cloud Storage bucket**.
4. Esses dados armazenados serão usados para **retreinar os modelos com novos padrões de tráfego de rede**.

---

## 📈 Monitoramento

A API também conta com **monitoramento de métricas operacionais**, como:
- **Latência** (tempo de resposta da predição)
- **Throughput** (número de requisições por segundo)
- **Tamanho do modelo (em MB)**

---

## ✅ Benefícios dessa arquitetura

- **Alta disponibilidade** com Google Cloud Run.
- **Escalabilidade automática** da API e do serviço subscriber.
- **Ciclo contínuo de melhoria dos modelos**, com coleta automática de novos dados reais.
- **Rastreabilidade**: cada input é logado e salvo com segurança.
- **Flexibilidade**: escolha entre detecção conservadora (RandomForest) ou balanceada (IsolationForest).
---

### Produto final

**Temos agora**:
-  Um modelo supervisionado salvo com versionamento.
-  Um modelo não-supervisionado salvo com versionamento.
-  Logs e métricas registradas.
-  Performance alinhada com os objetivos.
-  Um modelo supervisionado "conservador", pronto pra alertar até a menor suspeita de ataque.
-  Um modelo não-supervisionado que detecta bem ataques e com ótima precisão.
-  Ambos os modelos em produção
-  Dados novos salvos na (servirão para retreino) bucket
-  Sistema de Mensageria com Google Cloud Pub/Sub