# Classificador Breast Cancer — Avaliação Semestral

## Descrição

Projeto desenvolvido em Python para classificação de eventos de recorrência do câncer de mama utilizando técnicas de Machine Learning.

O sistema possui dois módulos: treinamento e inferência.

## Base de dados

Foi utilizada a base Breast Cancer da UCI Machine Learning Repository.

Fonte: https://archive.ics.uci.edu/dataset/14/breast+cancer

A base possui 286 registros, nove atributos de entrada e duas classes de saída.

## Módulo de treinamento

O arquivo `treinamento.py` realiza o carregamento dos dados, tratamento de valores ausentes, codificação de atributos categóricos, separação entre treino e teste, treinamento de cinco algoritmos e comparação de desempenho por validação cruzada.

Algoritmos utilizados:

- Regressão Logística
- Árvore de Decisão
- Random Forest
- KNN
- SVM

O melhor modelo é salvo na pasta `modelos` utilizando Joblib.

## Resultados

| Algoritmo | Acurácia média CV |
|---|---|
| SVM | 75,88% |
| Random Forest | 75,45% |
| KNN | 73,70% |
| Regressão Logística | 72,39% |
| Árvore de Decisão | 71,05% |

O melhor algoritmo foi o **SVM com kernel RBF**, apresentando acurácia de **75,86% no conjunto de teste**.

O modelo apresentou recall de 29% para a classe de recorrência, indicando limitações na identificação desses casos.

## Módulo de inferência

O arquivo `inferencia.py` permite informar os nove atributos de uma nova paciente e obter a classificação prevista:

- `no-recurrence-events`: sem recorrência.
- `recurrence-events`: com recorrência.

## Como executar

**1. Instalar as dependências**

```bash
pip install -r requirements.txt
```

**2. Executar o treinamento**

```bash
python treinamento.py
```

**3. Executar a inferência**

```bash
python inferencia.py
```

## Tecnologias

Python, Pandas, Scikit-learn, UCI ML Repository e Joblib.

## Observação

Este projeto foi desenvolvido exclusivamente para fins acadêmicos e não deve ser utilizado para diagnóstico ou decisões clínicas.