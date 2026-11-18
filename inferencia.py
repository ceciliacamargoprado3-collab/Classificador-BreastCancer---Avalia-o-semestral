
import os
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo

# ==========================================
# 1. CARREGAR O MODELO TREINADO
# ==========================================

CAMINHO_MODELO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "modelos",
    "melhor_modelo.joblib"
)

if not os.path.exists(CAMINHO_MODELO):
    print("Modelo não encontrado!")
    print("Execute primeiro: python treinamento.py")
    raise SystemExit(1)

modelo = joblib.load(CAMINHO_MODELO)

# ==========================================
# 2. OBTER OS ATRIBUTOS VÁLIDOS
# ==========================================

dataset = fetch_ucirepo(id=14)
X = dataset.data.features.copy()

# Corrigir intervalos convertidos em datas
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

colunas = X.columns.tolist()

opcoes = {}

for coluna in colunas:
    opcoes[coluna] = sorted(
        X[coluna].dropna().astype(str).unique().tolist()
    )

# ==========================================
# 3. RECEBER DADOS DA NOVA PACIENTE
# ==========================================

print("\n===== CLASSIFICADOR BREAST CANCER =====")
print("Informe os dados da nova paciente.\n")

paciente = {}

for coluna in colunas:

    valores = opcoes[coluna]

    print(f"\nAtributo: {coluna}")

    for indice, valor in enumerate(valores, start=1):
        print(f"{indice} - {valor}")

    while True:
        escolha = input("Escolha uma opção: ").strip()

        if escolha.isdigit():
            indice = int(escolha)

            if 1 <= indice <= len(valores):
                valor = valores[indice - 1]

                if coluna == "deg-malig":
                    paciente[coluna] = int(valor)
                else:
                    paciente[coluna] = valor

                break

        print("Opção inválida. Tente novamente.")

# ==========================================
# 4. PREPARAR DADOS PARA INFERÊNCIA
# ==========================================

nova_paciente = pd.DataFrame(
    [paciente],
    columns=colunas
)

# ==========================================
# 5. REALIZAR A PREVISÃO
# ==========================================

previsao = modelo.predict(nova_paciente)[0]

print("\n===== RESULTADO DA INFERÊNCIA =====")

print("\nDados informados:")
print(nova_paciente.to_string(index=False))

print("\nClasse prevista:", previsao)

if previsao == "no-recurrence-events":
    print("Indicativo: SEM recorrência.")

elif previsao == "recurrence-events":
    print("Indicativo: COM recorrência.")

print("\nAtenção: resultado exclusivamente acadêmico.")
print("Não deve ser utilizado para diagnóstico clínico.")