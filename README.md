# Low-Connectivity Learning Analytics: Predictive Models for School Dropout Prevention in Mozambique

Este repositório contém o código e os dados sintéticos para o artigo científico: **"Low-Connectivity Learning Analytics: Lightweight Predictive Models for School Dropout Prevention in Mozambique"**. O trabalho apresenta o modelo ML-Evasion-LB, otimizado para ambientes de baixa largura de banda e recursos limitados.

---

## 1. Status do Artigo e Citação

| Detalhe | Valor |
| :--- | :--- |
| **Título Completo** | Equitable Dropout Prediction via Asynchronous Data: A Low-Bandwidth Machine Learning Approach in Mozambique |
| **Revista Alvo** | JEDM - Journal of Educational Data Mining |
| **Status Atual** | Em submissão (Under Review) |
| **Link para o Artigo** | https://www.escritadahistoria.com/p/atribuicao-de-doi-nas-publicacoes.html |
| **Autores** | [Seu Nome Completo] |

---

## 2. Visão Geral do Projeto (ML-Evasion-LB)

O objetivo principal deste projeto é provar a viabilidade de sistemas de Alerta Precoce (*Early Warning Systems*) em contextos de profunda desigualdade digital, como Moçambique.

O modelo **Random Forest** (ML-Evasion-LB) utiliza um conjunto minimalista de apenas três *features* assíncronas para alcançar alta precisão preditiva:

* **DAP** (Performance in Periodic Assessments)
* **FET** (Frequency of Task Submission)
* **TF** (Truancy/Absence Tendency)

### Resultados Chave de Sustentabilidade e Performance:

| Métrica | Resultado | Relevância |
| :--- | :--- | :--- |
| **AUC** | 0.95 | Excelente poder discriminatório. |
| **Recall (Sensibilidade)** | > 92.1% | Prioriza a identificação correta de alunos em risco para intervenção. |
| **Tempo de Execução** | < 150 ms | Viável para arquiteturas **offline-first** e baixo custo. |

---

## 3. Replicando os Resultados

Siga estas instruções passo a passo para configurar o ambiente e executar o modelo ML-Evasion-LB.

### Pré-requisitos

* Python 3.8+
* `pip` (gerenciador de pacotes Python)

### Passo 1: Clonar o Repositório

Abra seu terminal ou prompt de comando e execute:

```bash
git clone [https://github.com/](https://github.com/)[Seu-Nome-Usuário]/ML-Evasion-LB-Mozambique.git
cd ML-Evasion-LB-Mozambique
