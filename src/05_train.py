import pandas as pd
import os
import joblib
import json
from sklearn.pipeline import Pipeline
from config import (
    TREINO_PATH, FINAL_MODEL_PATH, BEST_PARAMS_PATH,
    TARGET, COLS_TO_REMOVE, NUMERIC_COLS, CATEGORICAL_COLS
    
)
from pipeline_utils import create_preprocessor, create_model

def treinar_modelo_final():
    print("📥 Lendo dados para treino final...")
    df = pd.read_csv(TREINO_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"❌ Coluna target '{TARGET}' não encontrada.")

    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET]

    print("🔧 Carregando melhores parâmetros do tuning...")
    with open(BEST_PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    preprocessor = create_preprocessor()
    model = create_model().set_params(**best_params)

    pipeline_final = Pipeline([
        ('pre', preprocessor),
        ('clf', model)
    ])

    print("🧠 Treinando modelo com todos os dados disponíveis...")
    pipeline_final.fit(X, y)

    print(f"💾 Salvando modelo final em: {FINAL_MODEL_PATH}")
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline_final, FINAL_MODEL_PATH)

    print("✅ Treinamento final concluído com sucesso!")

if __name__ == "__main__":
    treinar_modelo_final()