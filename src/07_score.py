import os
import pandas as pd
import joblib

from config import (
    ATIVOS_PATH,
    FINAL_MODEL_PATH,
    SCORED_PATH,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)

def score_ativos():
    print("📥 Lendo base de ativos...")
    df = pd.read_csv(ATIVOS_PATH)

    print("📦 Carregando modelo final (pipeline)...")
    modelo = joblib.load(FINAL_MODEL_PATH)

    print("🔠 Preparando colunas cruas (iguais ao treino)...")
    X = df.copy()

    # Garante que TODAS as colunas esperadas existam
    faltantes = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c not in X.columns]
    for c in faltantes:
        if c in NUMERIC_COLS:
            X[c] = 0
        else:
            X[c] = "DESCONHECIDO"

    # Tipagem mínima: numéricas coerentes e categóricas como string
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(str).fillna("DESCONHECIDO")

    # Seleciona e ordena como no treino
    X = X[NUMERIC_COLS + CATEGORICAL_COLS]

    print("📈 Gerando previsões...")
    df["Scored Labels"] = modelo.predict(X)
    # Para classificadores binários que possuem predict_proba
    if hasattr(modelo, "predict_proba"):
        df["Scored Probabilities"] = modelo.predict_proba(X)[:, 1]
    else:
        # Fallback: quando não há predict_proba (ex.: alguns modelos)
        # Usa decisão em 0/1 como probabilidade “proxy”
        df["Scored Probabilities"] = df["Scored Labels"].astype(float)

    # Define a coluna de identificação
    id_col = "NUM_CERTIFICADO" if "NUM_CERTIFICADO" in df.columns else (
        "CERTIFICADO" if "CERTIFICADO" in df.columns else None
    )
    if id_col is None:
        # Se não houver identificador, cria um incremental
        print("⚠️ Coluna de identificação não encontrada (NUM_CERTIFICADO/CERTIFICADO). "
              "Criando ID incremental.")
        id_col = "CERTIFICADO"
        df[id_col] = range(1, len(df) + 1)

    # Saída padronizada
    colunas_saida = [id_col, "Scored Labels", "Scored Probabilities"]
    resultado = df[colunas_saida]

    print(f"💾 Salvando resultado em: {SCORED_PATH}")
    os.makedirs(os.path.dirname(SCORED_PATH), exist_ok=True)
    resultado.to_csv(SCORED_PATH, index=False)

    print("✅ Scoring finalizado.")

if __name__ == "__main__":
    score_ativos()
