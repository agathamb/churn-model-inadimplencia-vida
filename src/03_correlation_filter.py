import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import os
from config import (
    PREPROCESSADO_PATH,
    PREPROCESSADO_FILTRADO_PATH,
    FEATURES_PATH,
    TARGET,
    SELECAO_VARIAVEIS
)
import warnings
warnings.filterwarnings("ignore")

def selecionar_por_correlacao(df, target):
    metodo = SELECAO_VARIAVEIS["metodo_correlacao"]
    estrategia = SELECAO_VARIAVEIS["estrategia_correlacao"]

    num = df.select_dtypes(include=[np.number]).copy()
    if target not in num.columns:
        # garante que o TARGET entra no subconjunto numérico
        df[target] = pd.to_numeric(df[target], errors="coerce")
        num = df.select_dtypes(include=[np.number]).copy()

    correlacoes = num.corr(method=metodo)[target].drop(target)
    correlacoes_abs = correlacoes.abs()

    if estrategia == "top_n":
        top_n = SELECAO_VARIAVEIS["correlacao_top_n"]
        selecionadas = correlacoes_abs.sort_values(ascending=False).head(top_n).index.tolist()

    elif estrategia == "limiar":
        limiar = SELECAO_VARIAVEIS["correlacao_limiar"]
        selecionadas = correlacoes_abs[correlacoes_abs >= limiar].index.tolist()

    elif estrategia == "selectkbest":
        k = SELECAO_VARIAVEIS["correlacao_k_best"]
        X = df.drop(columns=[target])
        y = df[target]
        X_num = X.select_dtypes(include=np.number).fillna(0)
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_num, y)
        selecionadas = X_num.columns[selector.get_support()].tolist()

    else:
        raise ValueError(f"❌ Estratégia de correlação desconhecida: {estrategia}")

    return selecionadas

def main():
    print("📥 Lendo base preprocessada...")
    df = pd.read_csv(PREPROCESSADO_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"❌ Coluna target '{TARGET}' não encontrada.")

    print("🔍 Selecionando variáveis via correlação...")
    selecionadas = selecionar_por_correlacao(df, TARGET)

    print(f"✅ Variáveis selecionadas ({len(selecionadas)}): {selecionadas}")

    # Filtrar base
    df_filtrado = df[selecionadas + [TARGET]]

    # Salvar base filtrada
    os.makedirs(os.path.dirname(PREPROCESSADO_FILTRADO_PATH), exist_ok=True)
    df_filtrado.to_csv(PREPROCESSADO_FILTRADO_PATH, index=False)
    print(f"💾 Base filtrada salva em: {PREPROCESSADO_FILTRADO_PATH}")

    # Salvar lista de features
    pd.DataFrame(selecionadas, columns=["features"]).to_csv(FEATURES_PATH, index=False)
    print(f"📝 Lista de features salva em: {FEATURES_PATH}")

if __name__ == "__main__":
    main()
