# 🏥 Sistema Especialista Híbrido: Diagnóstico Médico

> **Integração entre Conhecimento Humano (IA Simbólica) e Aprendizado de Máquina (IA Numérica).**

Este projeto foi desenvolvido como requisito avaliativo da disciplina **T_TT050A_2025S2 - Sistemas de Apoio à Decisão**.

O sistema propõe uma abordagem híbrida para o diagnóstico de Câncer de Mama (utilizando o dataset *Breast Cancer Wisconsin*), unindo a precisão estatística de modelos de Machine Learning com a explicabilidade e segurança de Sistemas Especialistas baseados em regras.

---

## 👥 Grupo 10

| Nome | RA |
|------|----|
| **Hugo Strassa** | 246710 |
| Daniel Aniceto Rosell | 283988 |
| Davie Schimidt Fonseca | 259908 |
| Gabriel Sorensen M Traina | 283997 |
| Kaue Samuel Oliveira da Silva | 178449 |
| Kauã Henrique da Silva Andrade | 246165 |

---

## 🧠 Como Funciona a Hibridização

O sistema opera em 3 camadas para garantir um diagnóstico seguro:

1.  **IA Numérica (Estatística):** Uma Árvore de Decisão (`DecisionTreeClassifier`) treinada analisa os dados brutos e fornece uma predição baseada em padrões matemáticos.
2.  **IA Simbólica (Regras):** Um Motor de Inferência (`Experta`) aplica regras médicas explícitas (ex: concavidade severa, tamanho do tumor) para detectar riscos clínicos.
3.  **Motor Híbrido (Decisão):** O sistema cruza os dois resultados.
    * Se ambos concordam, a confiança é alta.
    * **Diferencial:** Se o ML prevê "Benigno" mas as Regras detectam "Alto Risco", o sistema **interrompe a automação** e emite um alerta de incerteza, recomendando revisão humana.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Machine Learning:** `scikit-learn`
* **Sistema Especialista:** `experta` (fork moderno do Pyknow)
* **Manipulação de Dados:** `pandas`, `numpy`

---

## 🚀 Como Rodar o Projeto

### Pré-requisitos

Certifique-se de ter o Python instalado. Recomenda-se o uso de um ambiente virtual (conda ou venv).

```bash
# Instale as dependências
pip install pandas numpy scikit-learn experta
python sistema_hibrido.py
