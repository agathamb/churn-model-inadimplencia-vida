import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import argparse
import re
from Levenshtein import ratio
from config import (
    TREINO_PATH, TESTE_PATH, PREPROCESSADO_PATH, TARGET,
    NUMERIC_COLS, CATEGORICAL_COLS,
    REBALANCEAMENTO, REBALANCEAMENTO_PARAMETROS,
    USAR_VIF, VIF_THRESHOLD, PREPROCESS_CAT,
    USE_SAMPLE_WEIGHTS, SAMPLE_WEIGHT_RULES,
    TRAIN_TEST_SPLIT_PARAMS
)
from sklearn.model_selection import train_test_split
import unicodedata
from jellyfish import jaro_winkler_similarity
import string
import warnings
warnings.filterwarnings("ignore")


def carregar_dados(path):
    print(f"📥 Tentando carregar arquivo de: {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def calcular_vif(df):
    vif_data = pd.DataFrame()
    vif_data["variavel"] = df.columns
    vif_data["vif"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data


def remover_multicolinearidade(X, threshold):
    removidas = []
    while True:
        vif = calcular_vif(X)
        max_vif = vif["vif"].max()
        if max_vif > threshold:
            to_remove = vif.sort_values("vif", ascending=False).iloc[0]["variavel"]
            X = X.drop(columns=[to_remove])
            removidas.append(to_remove)
        else:
            break
    return X, removidas

# Função para padronizar texto
def normalizar_texto(texto):
    if pd.isnull(texto):
        return np.nan
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    return texto

def aplicar_preprocessamento_categorico(df: pd.DataFrame, regras: dict):
    for coluna, valores_validos in regras.items():
        if coluna in df.columns:
            valores_validos_norm = [normalizar_texto(v) for v in valores_validos]
            col_norm = df[coluna].apply(normalizar_texto)
            df[coluna] = df[coluna].where(col_norm.isin(valores_validos_norm), np.nan)


def aplicar_rebalanceamento(X, y):
    metodo = REBALANCEAMENTO.lower()
    print(f"🔄 Aplicando balanceamento: {metodo}")

    if metodo == "smote":
        smote = SMOTE(**REBALANCEAMENTO_PARAMETROS["smote"])
        return smote.fit_resample(X, y)

    elif metodo == "undersample":
        under = RandomUnderSampler(**REBALANCEAMENTO_PARAMETROS["undersample"])
        return under.fit_resample(X, y)

    elif metodo == "oversample":
        over = RandomOverSampler(**REBALANCEAMENTO_PARAMETROS["oversample"])
        return over.fit_resample(X, y)

    elif metodo == "class_weight" or metodo == "none":
        print("⚖ Usando class_weight ou sem balanceamento direto.")
        return X, y

    else:
        raise ValueError(f"❌ Método de rebalanceamento inválido: {metodo}")


def aplicar_pesos(df):
    if not USE_SAMPLE_WEIGHTS:
        return None
    print("⚖ Aplicando regras de peso...")
    pesos = pd.Series(1.0, index=df.index)
    for regra in SAMPLE_WEIGHT_RULES:
        cond = regra["cond"]
        peso = float(regra["peso"])
        pesos.loc[df.query(cond).index] = peso
    return pesos


def main():
    df = carregar_dados(TREINO_PATH)

    print("🧹 Aplicando pré-processamento categórico...")
    aplicar_preprocessamento_categorico(df, PREPROCESS_CAT)

    if TARGET not in df.columns:
        raise ValueError(f"❌ Coluna target '{TARGET}' não encontrada.")

    # Separar em treino e teste antes do rebalanceamento
    stratify = df[TARGET] if TRAIN_TEST_SPLIT_PARAMS.get("stratify", True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[TARGET]),
        df[TARGET],
        test_size=TRAIN_TEST_SPLIT_PARAMS["test_size"],
        stratify=stratify,
        random_state=TRAIN_TEST_SPLIT_PARAMS["random_state"]
    )
    
    print(f"✂️ Dados divididos em treino ({1-TRAIN_TEST_SPLIT_PARAMS['test_size']:.0%}) e teste ({TRAIN_TEST_SPLIT_PARAMS['test_size']:.0%})")

    # Rebalanceamento apenas no conjunto de treino
    X_train_bal, y_train_bal = aplicar_rebalanceamento(X_train, y_train)

    # (Opcional) Adicionar sample weights
    pesos_train = aplicar_pesos(pd.concat([X_train_bal, y_train_bal], axis=1)) if USE_SAMPLE_WEIGHTS else None

    # Recompor DataFrames
    df_train = X_train_bal.copy()
    df_train[TARGET] = y_train_bal
    
    df_test = X_test.copy()
    df_test[TARGET] = y_test

    print(f"💾 Salvando base de treino pré-processada em: {PREPROCESSADO_PATH}")
    df_train.to_csv(PREPROCESSADO_PATH, index=False)
    
    print(f"💾 Salvando base de teste em: {TESTE_PATH}")
    df_test.to_csv(TESTE_PATH, index=False)

    print(f"💾 Salvando base de treino em: {TREINO_PATH}")
    df_train.to_csv(TREINO_PATH, index=False)


    print("✅ Pré-processamento concluído com sucesso!")


if __name__ == "__main__":
    main()