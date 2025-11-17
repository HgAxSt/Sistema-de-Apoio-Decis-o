#!/usr/bin/env python
# coding: utf-8

# In[6]:


import collections.abc
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

print(">>> INICIANDO SISTEMA...")
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target 


y = np.where(y == 0, 1, 0) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f" [OK] Dados carregados com sucesso.")
print(f"      - Total de Pacientes: {X.shape[0]}")
print(f"      - Características analisadas por paciente: {X.shape[1]}")
print(f"      - Pacientes para Treino: {X_train.shape[0]}")
print(f"      - Pacientes para Teste: {X_test.shape[0]}")
print("-" * 30)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

print(">>> TREINANDO IA ESTATÍSTICA (Árvore de Decisão)...")

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f" [OK] Modelo treinado.")
print(f"      - Acurácia (Acertos gerais): {acc*100:.1f}%")
print(f"      - Recall (Capacidade de detectar câncer): {rec*100:.1f}%")

def get_ml_prediction(paciente_data):
    reshaped_data = paciente_data.values.reshape(1, -1)
    
    predicao = clf.predict(reshaped_data)[0] # 0 ou 1
    probabilidade = clf.predict_proba(reshaped_data).max() # Ex: 0.95 (95% de certeza)
    
    return predicao, probabilidade

print("-" * 30)


# In[8]:


from experta import *

print(">>> INICIALIZANDO MOTOR DE REGRAS (IA Simbólica)...")

class Sintomas(Fact):
    """
    Representa os dados clínicos do paciente que chegam ao sistema.
    O experta usa isso para verificar se as regras batem.
    """
    pass

class DiagnosticoMedico(KnowledgeEngine):
    
    def __init__(self):
        super().__init__()
        self.risco_simbolico = 0
        self.explicacao = []

    @Rule(Sintomas(mean_radius=P(lambda x: x > 15.0)))
    def regra_tumor_grande(self):
        self.risco_simbolico += 1
        self.explicacao.append("ALERTA: Nódulo com raio médio elevado (> 15.0).")

    @Rule(Sintomas(mean_texture=P(lambda x: x > 21.0)))
    def regra_textura_irregular(self):
        self.risco_simbolico += 1
        self.explicacao.append("ALERTA: Textura do nódulo apresenta irregularidade alta.")

    @Rule(Sintomas(worst_concavity=P(lambda x: x > 0.25)))
    def regra_concavidade_severa(self):
        self.risco_simbolico += 2  # Peso maior (2 pontos) para essa regra!
        self.explicacao.append("PERIGO: Concavidade severa detectada (indicador forte).")

    def get_analise_simbolica(self):
        # Lógica de Pontuação (Score)
        if self.risco_simbolico >= 3:
            return "ALTO RISCO", self.explicacao
        elif self.risco_simbolico >= 1:
            return "MÉDIO RISCO", self.explicacao
        else:
            return "BAIXO RISCO", ["Nenhum sinal crítico detectado pelas regras explícitas."]

print(" [OK] Motor de regras construído.")
print("-" * 30)


# In[9]:


print(">>> EXECUTANDO SISTEMA HÍBRIDO...")

def sistema_hibrido_diagnostico(paciente_idx):
    try:
        paciente_dados = X_test.iloc[paciente_idx]
        diagnostico_real = y_test[paciente_idx] # O gabarito (só para a gente conferir)
    except:
        print("ID inválido.")
        return

    print(f"\n==================================================")
    print(f" ANÁLISE DO PACIENTE ID: {paciente_idx}")
    print(f"==================================================")

    pred_ml, prob_ml = get_ml_prediction(paciente_dados)
    
    veredito_ml = "MALIGNO" if pred_ml == 1 else "BENIGNO"
    print(f"[1] Parecer Estatístico (ML): {veredito_ml} (Confiança: {prob_ml*100:.1f}%)")

    engine = DiagnosticoMedico()
    engine.reset()
    
    engine.declare(Sintomas(
        mean_radius=paciente_dados['mean radius'],
        mean_texture=paciente_dados['mean texture'],
        worst_concavity=paciente_dados['worst concavity']
    ))
    engine.run()
    
    risco_simbolico, explicacao = engine.get_analise_simbolica()
    print(f"[2] Parecer Simbólico (Regras): {risco_simbolico}")

    decisao_final = ""
    nivel_confianca = ""
    recomendacao = ""

    # Cenario 1: ML diz Câncer
    if pred_ml == 1: 
        decisao_final = "POSITIVO PARA MALIGNIDADE"
        if risco_simbolico == "ALTO RISCO":
            nivel_confianca = "MUITO ALTA"
            recomendacao = "Início imediato de protocolo oncológico."
        elif risco_simbolico == "MÉDIO RISCO":
            nivel_confianca = "ALTA"
            recomendacao = "Investigação aprofundada recomendada."
        else:
            nivel_confianca = "MODERADA"
            recomendacao = "Padrão estatístico suspeito, mas sem sinais clínicos clássicos. Biópsia sugerida."

   # Cenario 2: ML diz Benigno
    else: 
        if risco_simbolico == "ALTO RISCO" or risco_simbolico == "MÉDIO RISCO":
            
            decisao_final = "INCERTO (ALERTA DE CONFLITO)"
            nivel_confianca = "BAIXA"
            recomendacao = "ATENÇÃO: ML sugere benigno, mas Regras detectaram sinais suspeitos (Concavidade/Textura). REVISÃO HUMANA OBRIGATÓRIA."
            
        else:
            decisao_final = "NEGATIVO PARA MALIGNIDADE"
            nivel_confianca = "ALTA"
            recomendacao = "Acompanhamento de rotina."


    print("\n---------------- RELATÓRIO FINAL DO SISTEMA ----------------")
    print(f"DIAGNÓSTICO:  {decisao_final}")
    print(f"CONFIANÇA:    {nivel_confianca}")
    print(f"AÇÃO:         {recomendacao}")
    print("\nJUSTIFICATIVA (Base de Conhecimento):")
    if not explicacao:
        print(" - Nenhum sinal crítico específico detectado pelas regras.")
    else:
        for nota in explicacao:
            print(f" - {nota}")
    
    print("-" * 52)
    print(f"(Gabarito Real: {'MALIGNO' if diagnostico_real == 1 else 'BENIGNO'})")


sistema_hibrido_diagnostico(0)

sistema_hibrido_diagnostico(20)


# In[ ]:




