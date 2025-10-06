
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import time

# --- 1. Simulação dos Dados (N=500) ---
# O dataset é simulado para ter as distribuições e proporções da Tabela 2
# Features: FET (0-100), DAP (0-20), TF (0-20)
# Target: Dropout (18% = 1)
np.random.seed(42) 
N = 500
dropout_rate = 0.18
n_dropout = int(N * dropout_rate)
n_retention = N - n_dropout

# Gerar dados simulados para a classe Retention (0)
df_retention = pd.DataFrame({
    'FET': np.clip(np.random.normal(68, 18, n_retention), 0, 100),
    'DAP': np.clip(np.random.normal(11.5, 2.5, n_retention), 0, 20),
    'TF': np.clip(np.random.poisson(4, n_retention), 0, 20),
    'Dropout': 0
})

# Gerar dados simulados para a classe Dropout (1) - Pior desempenho/Engajamento
df_dropout = pd.DataFrame({
    'FET': np.clip(np.random.normal(45, 15, n_dropout), 0, 100),
    'DAP': np.clip(np.random.normal(7.0, 2.0, n_dropout), 0, 20),
    'TF': np.clip(np.random.poisson(8, n_dropout), 0, 20),
    'Dropout': 1
})

# Combinar e embaralhar o dataset
df = pd.concat([df_retention, df_dropout]).sample(frac=1, random_state=42).reset_index(drop=True)
X = df[['FET', 'DAP', 'TF']]
y = df['Dropout']

# --- 2. Definição de Parâmetros e Modelo (Random Forest) ---
# Parâmetros conforme o Apêndice (Section 7)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 4,
    'criterion': 'gini',
    'class_weight': 'balanced', # Estratégia adicional para classe desbalanceada
    'random_state': 42
}

# Threshold otimizado para maximizar F1-Score da minoria (Recall)
OPTIMAL_THRESHOLD = 0.41 
N_SPLITS = 5
sm = SMOTE(k_neighbors=5, random_state=42)
scaler = StandardScaler()

# Listas para armazenar métricas de cada fold
auc_scores = []
recall_scores = []
f1_scores = []

# --- 3. Pseudo-Algoritmo ML-Evasion-LB com Cross-Validation ---

# O 20% Holdout Set seria separado antes, mas aqui simulamos apenas o 5-fold no total para demonstrar o CV
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print(f"Iniciando Stratified {N_SPLITS}-Fold Cross-Validation...")
start_time = time.time()

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # FASE 1: Pré-processamento
    # Z-score Standardization (Apenas no treino, para evitar Data Leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # FASE 2: Balanceamento (SMOTE) no conjunto de treino
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    # print(f"Fold {fold+1} - Classe original: {Counter(y_train)}, Classe após SMOTE: {Counter(y_res)}")

    # FASE 3: Treinamento e Predição
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_res, y_res)
    
    # Previsão de Probabilidades (necessário para AUC e otimização de Threshold)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Aplicação do Threshold Otimizado (0.41)
    y_pred_threshold = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
    
    # FASE 4: Avaliação das Métricas
    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    
    auc_scores.append(auc)
    recall_scores.append(recall)
    f1_scores.append(f1)

# --- 4. Relatório de Desempenho (Métricas do Artigo) ---
end_time = time.time()
runtime = (end_time - start_time) * 1000 / N_SPLITS # Tempo médio por fold em ms

print("\n" + "="*50)
print("RELATÓRIO DE DESEMPENHO ML-EVASION-LB (5-FOLD CV)")
print("="*50)
print(f"Modelo: Random Forest (Parâmetros: {RF_PARAMS['n_estimators']} estimators, depth {RF_PARAMS['max_depth']})")
print(f"Estratégia: SMOTE + Threshold Otimizado ({OPTIMAL_THRESHOLD})")
print("-" * 50)
print(f"AUC Média: {np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
print(f"Recall (Sensibilidade) Média: {np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}")
print(f"F1-Score Média: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"\nTempo de Execução (Média/Fold): {runtime:.2f} ms") # Simulação da Métrica de Sustentabilidade
print("="*50)
