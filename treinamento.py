
import os
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CARREGAMENTO DOS DADOS
# ==========================================

print("Carregando a base Breast Cancer...")

dataset = fetch_ucirepo(id=14)

X = dataset.data.features.copy()
y = dataset.data.targets.squeeze()

correcoes = {
    "tumor-size": {
        "9-May": "5-9",
        "14-Oct": "10-14"
    },
    "inv-nodes": {
        "5-Mar": "3-5",
        "8-Jun": "6-8",
        "11-Sep": "9-11",
        "14-Dec": "12-14"
    }
}

X = X.replace(correcoes)

print("\nDimensões:", X.shape)
print("\nValores ausentes:")
print(X.isnull().sum())

print("\nDistribuição das classes:")
print(y.value_counts())

# ==========================================
# 2. DIVISÃO TREINO E TESTE
# ==========================================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTreino:", X_treino.shape)
print("Teste:", X_teste.shape)

# ==========================================
# 3. PRÉ-PROCESSAMENTO
# ==========================================

# Todos os atributos serão tratados como
# categóricos, inclusive deg-malig.

colunas = X.columns.tolist()

preprocessador = ColumnTransformer(
    transformers=[
        (
            "categoricos",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            colunas
        )
    ]
)

# ==========================================
# 4. ALGORITMOS E HIPERPARÂMETROS
# ==========================================

modelos = {
    "Regressão Logística": (
        LogisticRegression(max_iter=2000),
        {
            "modelo__C": [0.1, 1, 10]
        }
    ),

    "Árvore de Decisão": (
        DecisionTreeClassifier(random_state=42),
        {
            "modelo__max_depth": [3, 5, 10, None],
            "modelo__min_samples_leaf": [1, 3, 5]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "modelo__n_estimators": [100, 200],
            "modelo__max_depth": [3, 5, None]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "modelo__n_neighbors": [3, 5, 7, 9]
        }
    ),

    "SVM": (
        SVC(),
        {
            "modelo__C": [0.1, 1, 10],
            "modelo__kernel": ["linear", "rbf"]
        }
    )
}

# ==========================================
# 5. TREINAMENTO E VALIDAÇÃO CRUZADA
# ==========================================

validacao = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

resultados = []
melhor_modelo = None
melhor_acuracia = -1
melhor_nome = ""

for nome, (algoritmo, parametros) in modelos.items():

    print(f"\nTreinando: {nome}")

    pipeline = Pipeline(steps=[
        ("preprocessamento", preprocessador),
        ("modelo", algoritmo)
    ])

    busca = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=validacao,
        scoring="accuracy",
        n_jobs=-1
    )

    busca.fit(X_treino, y_treino)

    acuracia = busca.best_score_

    print(f"Acurácia média CV: {acuracia:.4f}")
    print("Melhores parâmetros:", busca.best_params_)

    resultados.append({
        "Modelo": nome,
        "Acurácia CV": acuracia
    })

    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_modelo = busca.best_estimator_
        melhor_nome = nome

# ==========================================
# 6. COMPARAÇÃO DOS MODELOS
# ==========================================

tabela = pd.DataFrame(resultados)

tabela = tabela.sort_values(
    by="Acurácia CV",
    ascending=False
)

print("\n===== COMPARAÇÃO DOS MODELOS =====")
print(tabela.to_string(index=False))

print("\nMelhor modelo:", melhor_nome)
print(f"Melhor acurácia CV: {melhor_acuracia:.4f}")

# ==========================================
# 7. AVALIAÇÃO FINAL
# ==========================================

previsoes = melhor_modelo.predict(X_teste)

acuracia_teste = accuracy_score(
    y_teste,
    previsoes
)

print("\n===== AVALIAÇÃO FINAL =====")

print(f"Acurácia no teste: {acuracia_teste:.4f}")

print("\nRelatório de classificação:")
print(
    classification_report(
        y_teste,
        previsoes,
        zero_division=0
    )
)

classes = [
    "no-recurrence-events",
    "recurrence-events"
]

print("\nMatriz de confusão:")
print(
    confusion_matrix(
        y_teste,
        previsoes,
        labels=classes
    )
)

# ==========================================
# 8. SALVAR MODELO
# ==========================================

os.makedirs("modelos", exist_ok=True)

joblib.dump(
    melhor_modelo,
    "modelos/melhor_modelo.joblib"
)

tabela.to_csv(
    "modelos/comparacao_modelos.csv",
    index=False
)

print("\nModelo salvo com sucesso!")
print("Arquivo: modelos/melhor_modelo.joblib")
print("Comparação: modelos/comparacao_modelos.csv")