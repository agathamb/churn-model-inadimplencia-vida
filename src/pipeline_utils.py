from custom_transformers import (
    NullNumericImputer, NullCategoricalImputer
)
from config import (
    MODELO_BASE, MODEL_PARAMS,
    NUMERIC_COLS, CATEGORICAL_COLS
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def create_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', NullNumericImputer(columns=NUMERIC_COLS, modelo=MODELO_BASE))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', NullCategoricalImputer(columns=CATEGORICAL_COLS, modelo=MODELO_BASE)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor

def create_model():
    model_class = {
        "xgboost": __import__("xgboost").XGBClassifier,
        "randomforest": __import__("sklearn.ensemble").ensemble.RandomForestClassifier,
        "logisticregression": __import__("sklearn.linear_model").linear_model.LogisticRegression,
        "extratrees": __import__("sklearn.ensemble").ensemble.ExtraTreesClassifier,
        "gradientboosting": __import__("sklearn.ensemble").ensemble.GradientBoostingClassifier,
        "lgbm": __import__("lightgbm").LGBMClassifier
    }[MODELO_BASE]
    return model_class(**MODEL_PARAMS[MODELO_BASE])