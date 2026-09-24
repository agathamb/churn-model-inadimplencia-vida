import os
from pathlib import Path
from scipy.stats import randint, uniform
from datetime import datetime, timedelta

# =====================
# ⚠️ execuçãp
# =====================

hoje = datetime.today()
mes_anterior = hoje.replace(day=1) - timedelta(days=1)
ANO_MES = mes_anterior.strftime("%Y%m")
#ANO_MES = "202601"

#def get_cluster_instance() -> str:
#    """
#    Retorna  o nome do cluster ou instance atual.
#    """
#    current_path = Path(os.path.realpath(os.getcwd()))
#    parts = list(current_path.parts)

#    # 'clusters' sempre vem antes do nome do cluster
#    idx = parts.index("clusters")
##    return parts[idx + 1]

#CLUSTER_INSTANCE = get_cluster_instance()


#########################################################################
import os
from pathlib import Path

def get_cluster_instance() -> str:
    """
    Retorna o nome do cluster ou da instância atual, 
    funcionando tanto em notebooks quanto em jobs do Azure ML.
    """

    # Caminho do arquivo atual (ou do working dir, como fallback)
    try:
        base_path = Path(__file__).resolve()
    except NameError:
        base_path = Path(os.getcwd()).resolve()

    parts = list(base_path.parts)

    # Caso esteja rodando em ambiente com 'clusters' no path (ex: Notebook)
    if "clusters" in parts:
        idx = parts.index("clusters")
        return parts[idx + 1]

    # Variáveis de ambiente disponíveis em jobs do Azure ML
    cluster_name = os.getenv("AZUREML_COMPUTE_CLUSTER_NAME")
    node_id = os.getenv("AZ_BATCH_NODE_ID")
    run_id = os.getenv("AZUREML_RUN_ID")

    # Prioridade de identificação
    if cluster_name:
        return cluster_name
    elif node_id:
        return node_id
    elif run_id:
        return run_id.split("-")[0]
    else:
        return "unknown_instance"

# ======================================
# 🧩 Construção automática do MAIN_DIR
# ======================================

CLUSTER_INSTANCE = get_cluster_instance()

# Caminho base provável do Azure ML Notebooks
notebook_path = Path(f"/mnt/batch/tasks/shared/LS_root/mounts/clusters/{CLUSTER_INSTANCE}/code/Models/")

# Caminho base provável de execução em job
job_path = Path("/mnt/batch/tasks/shared/LS_root/mounts/code/Models/")

# Caminho de fallback local (ex.: execução fora do Azure)
local_path = Path.cwd() / "Models"

# Seleciona o caminho existente
if notebook_path.exists():
    MAIN_DIR = str(notebook_path)
elif job_path.exists():
    MAIN_DIR = str(job_path)
else:
    MAIN_DIR = str(local_path)

print(f"[INFO] MAIN_DIR detectado: {MAIN_DIR}")
print(f"[INFO] CLUSTER_INSTANCE: {CLUSTER_INSTANCE}")

#########################################################################





# =====================
# 📁 Caminhos padrão
# =====================

#MAIN_DIR = f"/mnt/batch/tasks/shared/LS_root/mounts/clusters/{CLUSTER_INSTANCE}/code/Models/"
PROJECT_NAME = "churn-model-inadimplencia-vida"
TRAIN_PATH  = "train"
INFER_PATH = "infer"

DATA_DIR = f"{MAIN_DIR}/{PROJECT_NAME}/data/{INFER_PATH}"
MODEL_DIR = f"{MAIN_DIR}/{PROJECT_NAME}/data/{TRAIN_PATH}"
TREINO_PATH = f"{MODEL_DIR}/treino.csv"
TESTE_PATH = f"{MODEL_DIR}/teste.csv"
ATIVOS_PATH = f"{DATA_DIR}/ativos.csv"
PREPROCESSADO_PATH = f"{MODEL_DIR}/preprocessado.csv"
PREPROCESSADO_FILTRADO_PATH = f"{MODEL_DIR}/preprocessado_filtrado.csv"
FEATURES_PATH = f"{MODEL_DIR}/features.csv"
FEATURES_TOP10_PATH = f"{MODEL_DIR}/features.csv"
BEST_PARAMS_PATH = f"{MODEL_DIR}/melhores_parametros.json"
BEST_MODEL_PATH = f"{MODEL_DIR}/melhor_modelo.pkl"
FINAL_MODEL_PATH = f"{MODEL_DIR}/modelo_final.pkl"
EVAL_PATH = f"{MODEL_DIR}/avaliacao.json"
SCORED_PATH = f"{DATA_DIR}/scored_ativos.csv"
BIN_DIST_PATH = f"{DATA_DIR}/bin_distribution.csv"
HTML_RELATORIO_PATH = f"{MODEL_DIR}/relatorio_final.html"
TUNING_INPUT_PATH = PREPROCESSADO_FILTRADO_PATH  # ou PREPROCESSADO_PATH
FEATURES_TREINADAS_PATH = f"{MODEL_DIR}/features.csv"
LOG_DIR = f"{MAIN_DIR}/{PROJECT_NAME}/logs"

LOG_FILE_TRAIN = f"{LOG_DIR}/exec_train_{datetime.today().strftime('%Y%m%d')}.log"
LOG_FILE_PRED = f"{LOG_DIR}/exec_pred_{datetime.today().strftime('%Y%m%d')}.log"

# =====================
# 🔐 Segurança
# =====================
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=mlwiaprod0012869149986;AccountKey=qCwHEzRawhkcKLZ76ixlnYgmHFuaYR3YXqXgDC012xJEom2enirv4hVIePVzwVGNCdlZAXwOjr1++AStdsbyug==;EndpointSuffix=core.windows.net"
SAS_TOKEN = "?sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2032-05-22T22:37:18Z&st=2025-05-22T14:37:18Z&spr=https&sig=aS%2FnuROWBg4ocnAyoLjMA6IZtTh47aDG%2B2KWy6Lf4vI%3D"
# =====================
# 📤 Azure Blob Storage
# =====================
STORAGE_ACCOUNT = "mlwiaprod0012869149986"
CONTAINER_NAME = "azureml-models"
TREINO_BLOB_NAME = f"EXTRACTION/{ANO_MES}/ML_INADIMPLENCIA_VIDA_{ANO_MES}.txt"
ATIVOS_BLOB_NAME = f"EXTRACTION/{ANO_MES}/ML_ATIVOS_INADIMPLENCIA_VIDA_{ANO_MES}.txt"
CONTAINER_URL = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}"
SAS_URL = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER_NAME}?{SAS_TOKEN}"
# =====================
# 🎯 Variáveis de modelo
# =====================
TARGET = "CANCELADO_INADIMPLENCIA"

ID_TRACKING = 'NUM_CERTIFICADO'

NUMERIC_COLS = [
    "VLR_IS","VLR_PREMIO", "QTD_VIDAS_SEGURADAS", "Beneficiarios", "VLR_RENDA_INDIVIDUAL"
]

CATEGORICAL_COLS = [
    "COD_PRODUTO", "NOM_REGIAO_AGENCIA", "DES_ESTADO_CIVIL", "IND_GENERO", "COD_CANAL_VENDA", "DES_FORMA_PAGAMENTO"
]

PREPROCESS_CAT = {
    "DES_ESTADO_CIVIL":[
        'Solteiro', 'Casado', 'Outros', 'Divorciado', 'Viuvo'
    ],
    "NOM_REGIAO_AGENCIA": [
        "NORDESTE", "SÃO PAULO", "SUL", "SUDESTE", "CENTRO-OESTE", "NORTE"
    ],
    "IND_GENERO": [
        "F", "M", "N"
    ],
    "DES_FORMA_PAGAMENTO":[
        "Débito em Conta C", "Débito em Conta P", "Boleto", "Cartão de Crédito"
    ] 

}

COLS_TO_REMOVE = []


# ============================================================
# Pós-processamento analítico - Quartis e Grupos (após o score)
# ============================================================

# Colunas do dataset (devem existir no TREINO e no SCORED)
# Ajuste os nomes exatamente como estão nos seus CSVs:
COL_VALOR_PAGO_SEGURO = "VLR_PREMIO"          # exemplo: prêmio/valor pago
COL_TEMPO_PERMANENCIA_MESES = "QTD_MESES_ATIVO"   # ajuste para o nome real no seu dado

# Colunas a serem criadas no SCORED
COL_GRUPO = "GRUPO"                     # conforme você pediu
COL_DESCRICAO_GRUPO = "DESC_GRUPO" # coluna de descrição

# (Opcional) Guardar também os quartis calculados no scored (útil p/ debug e BI)
CRIAR_COLUNAS_QUARTIS = False
COL_QUARTIL_VALOR_PAGO = "quartil_valor_pago"
COL_QUARTIL_TEMPO_MESES = "quartil_tempo_meses"

# Default para quem não se enquadra nas 4 regras
GRUPO_DEFAULT = 0
DESCRICAO_GRUPO_DEFAULT = "PERFIL CENTRAL DA BASE"

# Descrições dos grupos
DESCRICOES_GRUPOS = {
    1: "MAIOR TICKET MEDIO E MAIOR TEMPO DE PERMANENCIA NA CVP",
    2: "MAIOR TICKET MEDIO E TEMPO DE PERMANENCIA MEDIANO",
    3: "MAIOR TEMPO DE PERMANENCIA NA CVP E MENOR TICKET MEDIO",
    4: "MAIOR TICKET MEDIO E MENOR TEMPO DE PERMANENCIA NA CVP",
}

ANALYSIS_XLSX_PATH = f"{DATA_DIR}/analysis.xlsx"

METRICA_AVALIACAO = "f1_macro"
TOP_N_FEATURES = 10

#Opções de preenchimento de valores nulos
NULOS_PIPELINE = {
    "lgbm": {
        "numeric": {"impute": -9999, "flag": False},  # valor negativo, não precisa flag
        "categorical": {"impute": "DESCONHECIDO", "flag": False}
    },
    "xgboost": {
        "numeric": {"impute": -9999, "flag": False},
        "categorical": {"impute": "DESCONHECIDO", "flag": False}
    },
    "randomforest": {
        "numeric": {"impute": -9999, "flag": False},
        "categorical": {"impute": "DESCONHECIDO", "flag": False}
    },
    "extratrees": {
        "numeric": {"impute": -9999, "flag": False},
        "categorical": {"impute": "DESCONHECIDO", "flag": False}
    },
    "gradientboosting": {
        "numeric": {"impute": -9999, "flag": False},
        "categorical": {"impute": "DESCONHECIDO", "flag": False}
    },
    "logisticregression": {
        "numeric": {"impute": 0, "flag": True},  # 0 pode ser valor real, então precisa flag
        "categorical": {"impute": "DESCONHECIDO", "flag": False}
    }
}

# =====================
# 🧠 Seleção de Variáveis
# =====================
SELECAO_VARIAVEIS = {
    "usar_importancia_modelo": False,           # A: Usa importância das features do modelo (XGBoost, RF)
    "usar_correlacao": False,                   # B: Usa correlação com o target
    "usar_shap": True,                         # C: Usa SHAP para interpretabilidade
    "metodo_vetorizacao": "tfidf",              # D: Método de vetorização: "bert", "tfidf"
    # Configurações para seleção via correlação
    "metodo_correlacao": "pearson",            # Tipo de correlação: "pearson", "spearman", "kendall"
    "estrategia_correlacao": "top_n",          # Estratégia: "top_n", "limiar", "selectkbest"
    "correlacao_top_n": 10,                    # Se "top_n": mantém as N maiores correlações absolutas
    "correlacao_limiar": 0.1,                  # Se "limiar": mantém variáveis com correlação >= limiar
    "correlacao_k_best": 10                    # Se "selectkbest": número de features a selecionar com SelectKBest
}





# ------------------------
# Configuração de suavização do score
# ------------------------

# Se True aplica suavização nas probabilidades
USE_SCORE_SMOOTHING = True

# Método de suavização: "temperature", "shrink" ou "both"
SMOOTHING_METHOD = "temperature"  

# Parâmetro de temperature scaling (T>1 achata os picos)
TEMPERATURE = 1.6  

# Parâmetro de shrink para 0.5 (0.0 = sem shrink, até 0.3 é razoável)
LAMBDA_SHRINK = 0.1  

# Evita logit infinito
EPSILON = 1e-6




# =====================
# ⚖️ Rebalanceamento
# =====================
REBALANCEAMENTO = "undersample"  # Opções: smote | undersample | oversample | class_weight | None

# Parâmetros avançados para cada método
REBALANCEAMENTO_PARAMETROS = {

    "smote": {
        "sampling_strategy": "auto",     # Proporção desejada da classe minoritária após o SMOTE (ex: 0.5, "minority", "not majority")
        "k_neighbors": 5,                # Número de vizinhos para gerar amostras sintéticas (quanto menor, menos diversidade)
        "n_jobs": -1,                    # Número de threads (use -1 para usar todos os núcleos)
        "random_state": 42               # Fixar aleatoriedade para reprodutibilidade
    },

    "undersample": {
        "sampling_strategy": 0.5,     # Proporção final da classe minoritária ("auto", float, dict). "auto" reduz a classe majoritária ao tamanho da minoritária.
        "replacement": False,            # Se True, amostras podem ser selecionadas mais de uma vez
        "random_state": 42               # Controle de aleatoriedade do undersampling
    },

    "oversample": {
        "sampling_strategy": "auto",     # Proporção desejada da classe majoritária após oversampling
        "replacement": True,             # Se True, permite duplicar amostras já replicadas
        "random_state": 42               # Controle de aleatoriedade
    }
}

# =====================
# 🧮 Multicolinearidade
# =====================
USAR_VIF = False  # True para aplicar filtro de multicolinearidade, False para pular
VIF_THRESHOLD = 5.0  # Limite para remoção de variáveis com alta colinearidade

# =====================
# ⚖️ Regras de pesos por amostra
# =====================
USE_SAMPLE_WEIGHTS = False
SAMPLE_WEIGHT_RULES = [
    {"cond": "REGIAO == 'SUL'", "peso": 2.0}
    #,{"cond": "RESGATE > 10000", "peso": 3.0}
]


# =====================
# 🧠 Modelo base e opções
# =====================
MODELO_BASE = "xgboost"  # Opções: xgboost, randomforest, logisticregression, extratrees, gradientboosting, lgbm

# 🔍 Método de busca
SEARCH_METHOD = "hyperopt"  # Opções: grid, random, hyperopt


# =====================
# ⚙️ Parâmetros por modelo
# =====================
MODEL_PARAMS = {
    "xgboost": {
        "eval_metric": "logloss",       # Métrica de avaliação interna (ex: "logloss", "auc", "error")
        "random_state": 42,             # Define semente para reprodutibilidade dos resultados
        "n_jobs": -1                    # Nº de threads: -1 usa todos os núcleos (quanto maior, mais rápido)
    },
    "randomforest": {
        "n_estimators": 100,            # Nº de árvores na floresta (maior = mais robusto e lento)
        "max_depth": None,              # Profundidade máxima da árvore (None = ilimitada; menor = menos overfitting)
        "random_state": 42,             # Semente aleatória para reprodutibilidade
        "class_weight": "balanced"      # Ajuste de peso entre classes (opções: None, "balanced", "balanced_subsample")
    },
    "logisticregression": {
        "solver": "liblinear",          # Algoritmo de otimização (ex: "liblinear", "saga", "lbfgs", "newton-cg")
        "penalty": "l2",                # Tipo de regularização (ex: "l1", "l2", "elasticnet", "none")
        "random_state": 42,             # Reprodutibilidade
        "class_weight": "balanced"      # Ajuste automático para classes desbalanceadas (opções: None, "balanced")
    },
    "extratrees": {
        "n_estimators": 100,            # Nº de árvores no ensemble (maior = mais robusto, porém mais lento)
        "random_state": 42              # Define aleatoriedade da construção da floresta
    },
    "gradientboosting": {
        "n_estimators": 100,            # Nº de boosting stages (quanto maior, mais complexo)
        "learning_rate": 0.1,           # Taxa de aprendizado (menor = mais robusto e lento)
        "max_depth": 3,                 # Profundidade de cada árvore (menor = menos overfitting)
        "random_state": 42              # Reprodutibilidade
    },
    "lgbm": {
        "n_estimators": 100,            # Nº de boosting rounds (maior = mais aprendizado)
        "learning_rate": 0.1,           # Taxa de aprendizado (menor = mais conservador)
        "random_state": 42,             # Aleatoriedade
        "num_leaves": 31                # Nº máximo de folhas por árvore (maior = mais complexidade)
    }
}

RANDOM_SEARCH_PARAMS = {
    "n_iter": 50,                     # Nº de combinações a serem testadas (maior = mais chances de achar bom modelo)
    "param_distributions": {
        "n_estimators": randint(100, 300),    # Nº de árvores: mais árvores = maior capacidade, porém mais lento
        "max_depth": randint(3, 9),           # Profundidade: maior = mais complexidade, menor = menor overfitting
        "learning_rate": uniform(0.01, 0.2),  # Taxa de aprendizado: menor = mais conservador, maior = mais rápido
        "subsample": uniform(0.6, 0.4)        # Subamostragem: menor = mais regularização, maior = menos bias
    }
}

GRID_SEARCH_PARAMS = {
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    },
    "randomforest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 20]
    },
    "logisticregression": {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    },
    "extratrees": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    },
    "gradientboosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },
    "lgbm": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 50]
    }
}

# =====================
# 📊 Validação, teste e execução
# =====================
CV_PARAMS = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42
}

TRAIN_TEST_SPLIT_PARAMS = {
    "test_size": 0.2,
    "stratify": True,
    "random_state": 42
}

TUNING_PARAMS = {
    "test_size": 0.1,
    "stratify": True,
    "random_state": 42
}

GRIDSEARCH_PARAMS = {
    "verbose": 1,
    "n_jobs": -1
}


# =====================
# 📤 Upload para Blob
# =====================

UPLOAD_MAP = {
    SCORED_PATH: f"METRICS/{ANO_MES}/ML_SCORED_INADIMPLENCIA_VIDA_{ANO_MES}.csv",

}
