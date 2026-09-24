import pandas as pd
import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from config import (
    TREINO_PATH, BEST_MODEL_PATH, BEST_PARAMS_PATH,
    TARGET, NUMERIC_COLS, CATEGORICAL_COLS,
    MODELO_BASE, MODEL_PARAMS, GRID_SEARCH_PARAMS,
    METRICA_AVALIACAO, CV_PARAMS, TUNING_PARAMS, SEARCH_METHOD,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from custom_transformers import (
    NullNumericImputer, NullCategoricalImputer
)
from hyperopt import STATUS_OK, STATUS_FAIL
from pipeline_utils import create_preprocessor

from sklearn.metrics import f1_score

def tunar_modelo():
    print("📥 Lendo dados para tunagem...")
    df = pd.read_csv(TREINO_PATH)
    print("[DEBUG] Colunas do DataFrame original:", df.columns.tolist())

    if TARGET not in df.columns:
        raise ValueError(f"❌ Coluna target '{TARGET}' não encontrada.")

    # Verificação explícita das colunas
    missing_num = set(NUMERIC_COLS) - set(df.columns)
    missing_cat = set(CATEGORICAL_COLS) - set(df.columns)
    if missing_num:
        print(f"⚠️ Colunas numéricas faltando: {missing_num}")
    if missing_cat:
        print(f"⚠️ Colunas categóricas faltando: {missing_cat}")

    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET]
    print("[DEBUG] Colunas usadas em X (antes do split):", X.columns.tolist())
    print("[DEBUG] Shape de X (antes do split):", X.shape)

    print("🔀 Separando treino/teste para tunagem...")
    stratify = y if TUNING_PARAMS.get("stratify", False) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TUNING_PARAMS["test_size"],
        stratify=stratify,
        random_state=TUNING_PARAMS["random_state"]
    )
    print("[DEBUG] Colunas em X_train:", X_train.columns.tolist())
    print("[DEBUG] Shape de X_train:", X_train.shape)
    print("[DEBUG] Colunas em X_test:", X_test.columns.tolist())
    print("[DEBUG] Shape de X_test:", X_test.shape)

    # Pipeline robusto com verificação de tipos
    preprocessor = create_preprocessor()

    # Debug do preprocessor: fit e nomes das features
    print("[DEBUG] Fit do preprocessor em X_train...")
    preprocessor.fit(X_train)
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        elif hasattr(preprocessor, 'transformers_'):
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        feature_names.extend(trans.get_feature_names_out(cols))
                    except Exception:
                        feature_names.extend(cols)
                else:
                    feature_names.extend(cols)
        else:
            feature_names = []
        print(f"[DEBUG] Nomes das features após pre-processamento: {feature_names if feature_names else 'Não disponível.'}")
    except Exception as e:
        print(f"[DEBUG] Não foi possível obter nomes das features: {e}")
    try:
        X_train_transf = preprocessor.transform(X_train)
        print(f"[DEBUG] Shape de X_train após pre-processamento: {X_train_transf.shape}")
    except Exception as e:
        print(f"[DEBUG] Erro ao transformar X_train: {e}")

    model_class = {
        "xgboost": __import__("xgboost").XGBClassifier,
        "randomforest": __import__("sklearn.ensemble").ensemble.RandomForestClassifier,
        "logisticregression": __import__("sklearn.linear_model").linear_model.LogisticRegression,
        "extratrees": __import__("sklearn.ensemble").ensemble.ExtraTreesClassifier,
        "gradientboosting": __import__("sklearn.ensemble").ensemble.GradientBoostingClassifier,
        "lgbm": __import__("lightgbm").LGBMClassifier
    }[MODELO_BASE]

    if SEARCH_METHOD == "hyperopt":
        print("⚡ Iniciando tunagem com Hyperopt...")
        from hyperopt import fmin, tpe, hp, Trials, space_eval

        space = {}
        model_grid = GRID_SEARCH_PARAMS.get(MODELO_BASE, {})

        if "n_estimators" in model_grid:
            space["n_estimators"] = hp.choice("n_estimators", model_grid["n_estimators"])
        if "max_depth" in model_grid:
            space["max_depth"] = hp.choice("max_depth", model_grid["max_depth"])
        if "num_leaves" in model_grid:
            space["num_leaves"] = hp.choice("num_leaves", model_grid["num_leaves"])
        if "penalty" in model_grid:
            space["penalty"] = hp.choice("penalty", model_grid["penalty"])
        if "learning_rate" in model_grid:
            lr_min = min(model_grid["learning_rate"])
            lr_max = max(model_grid["learning_rate"])
            space["learning_rate"] = hp.uniform("learning_rate", lr_min, lr_max)
        if "subsample" in model_grid:
            sub_min = min(model_grid["subsample"])
            sub_max = max(model_grid["subsample"])
            space["subsample"] = hp.uniform("subsample", sub_min, sub_max)
        if "C" in model_grid:
            c_min = min(model_grid["C"])
            c_max = max(model_grid["C"])
            space["C"] = hp.uniform("C", c_min, c_max)

        # Remove random_state do space se existir
        if 'random_state' in space:
            del space['random_state']

        space.update({
            k: v for k, v in MODEL_PARAMS[MODELO_BASE].items()
            if k not in space and k != 'random_state'
        })

        def objective(params):
            try:
                if 'random_state' in params:
                    params.pop('random_state')
                model = model_class(**params, random_state=42)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_train)
                score = -f1_score(y_train, preds, average="macro")
                return {'loss': score, 'status': STATUS_OK}
            except Exception as e:
                print(f"❌ Erro no trial: {str(e)}")
                return {'loss': float('inf'), 'status': STATUS_FAIL}

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials
        )

        best_params = space_eval(space, best)
        if 'random_state' in best_params:
            best_params.pop('random_state')

        final_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_class(**best_params, random_state=42))
        ])

        final_model.fit(X_train, y_train)

        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        joblib.dump(final_model, BEST_MODEL_PATH)
        with open(BEST_PARAMS_PATH, 'w') as f:
            json.dump(best_params, f, indent=4)

        print(f"✅ Tunagem concluída. Modelo salvo em {BEST_MODEL_PATH}")

    else:
        raise ValueError("❌ Método de busca não suportado")

if __name__ == "__main__":
    tunar_modelo()