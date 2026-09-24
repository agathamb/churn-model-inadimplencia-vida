# Pipeline de Machine Learning para Inadimplência em Seguro de Vida

Pipeline em Python para treinamento, avaliação e inferência de um modelo preditivo de inadimplência em seguros de vida. O projeto integra ingestão de dados no Azure Blob Storage, pré-processamento, balanceamento de classes, ajuste de hiperparâmetros, treinamento, geração de scores, segmentação analítica por quartis e publicação dos resultados.

> **Importante:** todas as configurações operacionais estão centralizadas em `config.py`. Antes da execução, revise o período de referência, os caminhos, as credenciais, os arquivos do Blob Storage, as variáveis, as regras de negócio e os parâmetros do modelo.

> **Segurança:** o `config.py` analisado contém credenciais diretamente no código. Remova os segredos, faça a rotação das credenciais expostas e utilize Azure Key Vault, identidade gerenciada ou variáveis de ambiente antes de versionar ou compartilhar o projeto.

---

## Sumário

- [Objetivo](#objetivo)
- [Visão geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Execução](#execução)
- [Pipeline de treinamento](#pipeline-de-treinamento)
- [Pipeline de inferência](#pipeline-de-inferência)
- [Pós-processamento analítico](#pós-processamento-analítico)
- [Relatório analítico em Excel](#relatório-analítico-em-excel)
- [Entradas e saídas](#entradas-e-saídas)
- [Variáveis do modelo](#variáveis-do-modelo)
- [Modelos suportados](#modelos-suportados)
- [Métricas e avaliação](#métricas-e-avaliação)
- [Logs e rastreabilidade](#logs-e-rastreabilidade)
- [Segurança](#segurança)
- [Pontos de atenção técnicos](#pontos-de-atenção-técnicos)
- [Solução de problemas](#solução-de-problemas)
- [Checklist operacional](#checklist-operacional)
- [Recomendações de evolução](#recomendações-de-evolução)

---

## Objetivo

O modelo tem como objetivo estimar a probabilidade associada à inadimplência em seguros de vida. A saída pode apoiar análises de risco, priorização de carteiras e estudos de perfis, sempre observando as regras de negócio, governança, privacidade e validação aplicáveis.

A variável-alvo configurada é:

```python
TARGET = "CANCELADO_INADIMPLENCIA"
```

O identificador principal utilizado no acompanhamento dos registros é:

```python
ID_TRACKING = "NUM_CERTIFICADO"
```

---

## Visão geral

O projeto possui dois fluxos principais:

- **Treinamento:** ingestão, preparação dos dados, divisão entre treino e teste, balanceamento, ajuste de hiperparâmetros, treinamento final e avaliação.
- **Inferência:** ingestão dos ativos, aplicação do pipeline treinado, geração de probabilidades, classificação por grupos analíticos e upload do resultado.

Há também dois componentes complementares:

- `09_bin_distribution.py`, para distribuição dos scores em faixas de 5%;
- `analysis.py`, para geração de um relatório Excel com análises por grupo, classe predita e limites de probabilidade.

Esses dois componentes não são chamados atualmente pelos orquestradores principais e devem ser executados separadamente, caso sejam necessários.

---

## Arquitetura

### Fluxo de treinamento

```text
Azure Blob Storage
        |
        v
01_ingestion_train.py
        |
        v
02_preprocess.py
        |
        v
03_correlation_filter.py
        |
        v
04_tune.py
        |
        v
05_train.py
        |
        v
06_evaluate.py
        |
        v
Modelo, parâmetros, métricas e relatório PDF
```

### Fluxo de inferência

```text
Azure Blob Storage
        |
        v
01_ingestion_predict.py
        |
        v
07_score.py
        |
        v
08_post_process.py
        |
        v
10_upload_files.py
        |
        v
Resultado publicado no Azure Blob Storage
```

### Componentes analíticos opcionais

```text
scored_ativos.csv
        |
        +--> 09_bin_distribution.py --> bin_distribution.csv + PNG
        |
        +--> analysis.py --> analysis.xlsx
```

---

## Estrutura do projeto

```text
.
├── 01_ingestion_predict.py   # Download da base de ativos
├── 01_ingestion_train.py     # Download da base histórica de treinamento
├── 02_preprocess.py          # Tratamento categórico, split e balanceamento
├── 03_correlation_filter.py  # Seleção opcional de variáveis numéricas
├── 04_tune.py                # Ajuste de hiperparâmetros com Hyperopt
├── 05_train.py               # Treinamento do pipeline final
├── 06_evaluate.py            # Métricas, gráficos e relatório PDF
├── 07_score.py               # Geração de classes e probabilidades
├── 08_post_process.py        # Segmentação por quartis e grupos
├── 09_bin_distribution.py    # Distribuição dos scores em faixas
├── 10_upload_files.py        # Upload dos resultados
├── analysis.py               # Relatório analítico em Excel
├── config.py                 # Configuração central
├── custom_transformers.py    # Imputadores personalizados
├── pipeline_utils.py         # Construção do pré-processador e modelo
├── main_train.py             # Orquestração do treinamento
├── main_predict.py           # Orquestração da inferência
├── requirements.txt          # Dependências do ambiente
└── README.md                 # Documentação do projeto
```

### Estrutura esperada de artefatos

```text
Models/
└── churn-model-inadimplencia-vida/
    ├── data/
    │   ├── train/
    │   │   ├── treino.csv
    │   │   ├── teste.csv
    │   │   ├── preprocessado.csv
    │   │   ├── preprocessado_filtrado.csv
    │   │   ├── features.csv
    │   │   ├── melhores_parametros.json
    │   │   ├── melhor_modelo.pkl
    │   │   ├── modelo_final.pkl
    │   │   ├── avaliacao.json
    │   │   ├── relatorio_final.pdf
    │   │   ├── relatorio_final_confusion_matrix.png
    │   │   └── relatorio_final_roc_curve.png
    │   └── infer/
    │       ├── ativos.csv
    │       ├── scored_ativos.csv
    │       ├── bin_distribution.csv
    │       ├── bin_distribution.png
    │       └── analysis.xlsx
    └── logs/
        ├── exec_train_YYYYMMDD.log
        └── exec_pred_YYYYMMDD.log
```

---

## Pré-requisitos

- Python 3.10 recomendado;
- ambiente Linux ou Azure Machine Learning;
- acesso ao container configurado no Azure Blob Storage;
- permissão de leitura nos arquivos de entrada;
- permissão de gravação no caminho de publicação;
- credenciais disponibilizadas por mecanismo seguro;
- arquivos com as colunas esperadas pelo `config.py`;
- dependências instaladas pelo `requirements.txt`.

### Bibliotecas principais

- pandas;
- NumPy;
- scikit-learn;
- imbalanced-learn;
- XGBoost;
- LightGBM;
- Hyperopt;
- statsmodels;
- Matplotlib;
- FPDF;
- joblib;
- azure-storage-blob;
- openpyxl;
- python-Levenshtein;
- jellyfish.

---

## Instalação

### Com Conda

```bash
conda create -n inadimplencia-vida python=3.10 -y
conda activate inadimplencia-vida
pip install -r requirements.txt
```

### Com venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Validar o ambiente

```bash
python --version
pip check
```

Para registrar o ambiente efetivamente instalado:

```bash
pip freeze > requirements-lock.txt
```

---

## Configuração

Todas as variáveis são centralizadas em `config.py`.

### Período de referência

O período é definido automaticamente como o mês anterior à execução:

```python
hoje = datetime.today()
mes_anterior = hoje.replace(day=1) - timedelta(days=1)
ANO_MES = mes_anterior.strftime("%Y%m")
```

Para reprocessamentos, o período pode ser fixado manualmente. A alteração deve ser documentada e revertida após o processamento.

### Identificação do projeto

```python
PROJECT_NAME = "churn-model-inadimplencia-vida"
```

### Detecção do diretório principal

O código tenta identificar o ambiente nesta ordem:

1. caminho de notebooks do Azure ML;
2. caminho de jobs do Azure ML;
3. diretório local `Models` como fallback.

Na inicialização são exibidos:

```text
[INFO] MAIN_DIR detectado: ...
[INFO] CLUSTER_INSTANCE: ...
```

### Arquivos no Blob Storage

```python
TREINO_BLOB_NAME = f"EXTRACTION/{ANO_MES}/ML_INADIMPLENCIA_VIDA_{ANO_MES}.txt"
ATIVOS_BLOB_NAME = f"EXTRACTION/{ANO_MES}/ML_ATIVOS_INADIMPLENCIA_VIDA_{ANO_MES}.txt"
```

### Publicação do scoring

```python
UPLOAD_MAP = {
    SCORED_PATH: f"METRICS/{ANO_MES}/ML_SCORED_INADIMPLENCIA_VIDA_{ANO_MES}.csv",
}
```

### Modelo e otimização

```python
MODELO_BASE = "xgboost"
SEARCH_METHOD = "hyperopt"
METRICA_AVALIACAO = "f1_macro"
```

> No código atual, o objetivo do Hyperopt utiliza F1 macro, mas calcula a métrica sobre os próprios dados usados no ajuste de cada tentativa. Consulte os pontos de atenção técnicos.

### Divisão dos dados

```python
TRAIN_TEST_SPLIT_PARAMS = {
    "test_size": 0.2,
    "stratify": True,
    "random_state": 42,
}
```

O pré-processamento separa 20% dos registros para teste e aplica rebalanceamento apenas ao conjunto de treino.

### Balanceamento

```python
REBALANCEAMENTO = "undersample"
```

Métodos suportados:

- `smote`;
- `undersample`;
- `oversample`;
- `class_weight`;
- `none`.

### Tratamento de valores ausentes

`NULOS_PIPELINE` define, por tipo de modelo:

- valor de imputação numérica;
- valor de imputação categórica;
- criação opcional de indicador de ausência.

Os transformadores estão implementados em `custom_transformers.py` e integrados ao `ColumnTransformer` de `pipeline_utils.py`.

---

## Execução

Execute os comandos no diretório onde estão os scripts, pois os orquestradores utilizam nomes relativos.

### Treinamento completo

```bash
python main_train.py
```

Ordem executada:

```text
01_ingestion_train.py
02_preprocess.py
03_correlation_filter.py
04_tune.py
05_train.py
06_evaluate.py
```

### Inferência completa

```bash
python main_predict.py
```

Ordem executada:

```text
01_ingestion_predict.py
07_score.py
08_post_process.py
10_upload_files.py
```

### Distribuição dos scores

```bash
python 09_bin_distribution.py
```

### Relatório analítico Excel

```bash
python analysis.py
```

### Execução isolada

```bash
python 02_preprocess.py
python 04_tune.py
python 05_train.py
python 07_score.py
```

Uma etapa isolada depende dos artefatos gerados pelas etapas anteriores.

---

## Pipeline de treinamento

### 1. Ingestão da base histórica

Arquivo: `01_ingestion_train.py`

Responsabilidades:

- baixar o arquivo histórico do Blob Storage;
- detectar automaticamente o separador entre vírgula, ponto e vírgula, barra vertical e tabulação;
- testar as codificações `utf-8`, `utf-8-sig`, `latin1` e `cp1252`;
- utilizar leitura tolerante a linhas problemáticas;
- manter os campos inicialmente como texto;
- salvar o conteúdo em `TREINO_PATH`.

A chamada principal usa detecção automática de separador:

```python
download_blob(
    CONTAINER_URL,
    TREINO_BLOB_NAME,
    SAS_TOKEN,
    TREINO_PATH,
    sep=None,
)
```

### 2. Pré-processamento

Arquivo: `02_preprocess.py`

Responsabilidades:

- carregar a base histórica;
- normalizar textos categóricos para validação;
- invalidar categorias que não estejam nas listas permitidas;
- validar a existência da variável-alvo;
- separar treino e teste com estratificação opcional;
- aplicar rebalanceamento somente no treino;
- gerar a base preprocessada;
- gerar a base de teste;
- sobrescrever `TREINO_PATH` com o conjunto de treino rebalanceado.

#### Categorias validadas

O dicionário `PREPROCESS_CAT` contém regras para:

- estado civil;
- região da agência;
- gênero;
- forma de pagamento.

Valores fora das listas permitidas são substituídos por valor ausente e posteriormente tratados pelo pipeline.

#### Atenção sobre o arquivo original

O script salva o conjunto de treino processado novamente em `TREINO_PATH`. Portanto, o arquivo local inicialmente baixado deixa de representar a extração original após o pré-processamento.

### 3. Seleção por correlação

Arquivo: `03_correlation_filter.py`

O script:

- carrega `PREPROCESSADO_PATH`;
- força o alvo para formato numérico, quando necessário;
- calcula correlações somente com colunas numéricas;
- aplica `top_n`, limiar ou `SelectKBest`;
- salva `preprocessado_filtrado.csv`;
- salva `features.csv`.

> Embora essa etapa seja executada pelo `main_train.py`, o treinamento final usa diretamente as listas `NUMERIC_COLS` e `CATEGORICAL_COLS`, não o arquivo `features.csv`.

### 4. Ajuste de hiperparâmetros

Arquivo: `04_tune.py`

O script:

- carrega a base de treino;
- valida colunas numéricas e categóricas;
- realiza uma nova divisão entre treino e teste para a tunagem;
- cria o pré-processador central;
- imprime informações de diagnóstico;
- cria um Pipeline com pré-processador e classificador;
- executa Hyperopt;
- salva o melhor pipeline em `melhor_modelo.pkl`;
- salva os melhores parâmetros em `melhores_parametros.json`.

O espaço de busca é montado a partir de `GRID_SEARCH_PARAMS` para o modelo selecionado.

### 5. Treinamento final

Arquivo: `05_train.py`

O script:

- carrega `TREINO_PATH`;
- seleciona as variáveis numéricas e categóricas configuradas;
- carrega `melhores_parametros.json`;
- cria o pré-processador;
- cria o estimador-base;
- aplica os melhores parâmetros;
- treina um Pipeline completo;
- salva `modelo_final.pkl`.

O artefato final inclui pré-processamento e classificador, reduzindo o risco de divergência entre treino e inferência.

### 6. Avaliação

Arquivo: `06_evaluate.py`

O script:

- carrega `PREPROCESSADO_PATH`;
- carrega `modelo_final.pkl`;
- seleciona as variáveis configuradas;
- cria valores padrão para colunas ausentes;
- gera classes e probabilidades;
- calcula métricas;
- salva `avaliacao.json`;
- gera matriz de confusão;
- gera curva ROC;
- gera relatório PDF.

O relatório atual não inclui SHAP.

---

## Pipeline de inferência

### 1. Ingestão dos ativos

Arquivo: `01_ingestion_predict.py`

A ingestão:

- baixa a base de ativos;
- detecta o separador;
- testa codificações comuns;
- mantém os valores como texto;
- salva `ativos.csv`.

### 2. Geração dos scores

Arquivo: `07_score.py`

O script:

- carrega a base de ativos;
- carrega o pipeline final;
- cria colunas ausentes com valores padrão;
- converte variáveis numéricas;
- converte variáveis categóricas para texto;
- ordena as colunas conforme a configuração;
- gera classes e probabilidades;
- define o identificador de saída;
- salva `scored_ativos.csv`.

A prioridade do identificador é:

1. `NUM_CERTIFICADO`;
2. `CERTIFICADO`;
3. identificador incremental criado pelo script.

Saída padronizada:

```text
<ID>
Scored Labels
Scored Probabilities
```

Quando o estimador não possui `predict_proba`, a classe prevista é utilizada como aproximação de probabilidade.

### 3. Pós-processamento

Arquivo: `08_post_process.py`

O script:

- lê treino, ativos e scoring;
- detecta o separador dos arquivos;
- valida as colunas obrigatórias;
- converte prêmio e tempo de permanência para numérico;
- aprende limites de quartis com a base de treino;
- classifica os ativos;
- gera grupo e descrição;
- incorpora as classificações ao scoring por identificador;
- preenche valores padrão para registros sem correspondência;
- sobrescreve o arquivo de scoring usando `|` como separador.

### 4. Upload

Arquivo: `10_upload_files.py`

O script:

- percorre o dicionário `UPLOAD_MAP`;
- ignora arquivos locais inexistentes;
- envia os arquivos ao Blob Storage;
- utiliza sobrescrita no destino.

---

## Pós-processamento analítico

O pós-processamento utiliza duas dimensões:

```python
COL_VALOR_PAGO_SEGURO = "VLR_PREMIO"
COL_TEMPO_PERMANENCIA_MESES = "QTD_MESES_ATIVO"
```

Os quartis são calculados na base de treino e aplicados à base de ativos.

Valores negativos de tempo ou valor são mantidos na base, mas ficam fora do cálculo dos quartis. Essa regra deve ser validada com o negócio para cada variável.

### Grupos

| Grupo | Regra implementada | Descrição configurada |
|---:|---|---|
| 1 | Maior quartil de prêmio e maior quartil de permanência | Maior ticket médio e maior tempo de permanência na CVP |
| 2 | Maior quartil de prêmio e permanência nos quartis 2 ou 3 | Maior ticket médio e tempo de permanência mediano |
| 3 | Maior quartil de permanência e menor quartil de prêmio | Maior tempo de permanência na CVP e menor ticket médio |
| 4 | Maior quartil de prêmio e menor quartil de permanência | Maior ticket médio e menor tempo de permanência na CVP |
| 0 | Demais registros | Perfil central da base |

A segmentação é analítica e complementar ao score preditivo. O grupo não substitui a classe nem a probabilidade do modelo.

---

## Relatório analítico em Excel

Arquivo: `analysis.py`

Execução:

```bash
python analysis.py
```

Saída padrão:

```text
analysis.xlsx
```

Abas geradas:

1. `01_limites_quartis_treino`;
2. `02_qtd_por_grupo_e_label`;
3. `03_prob_media_por_grupo_e_label`;
4. `04_thresholds_por_label`;
5. `05_thresholds_por_grupo_e_label`;
6. `06_investigar_meses_negativos`;
7. `07_amostra_por_grupo_label`.

### Limites analisados

O relatório calcula contagens e percentuais para probabilidades maiores ou iguais a:

- 0,65;
- 0,70;
- 0,80;
- 0,90.

### Amostragem

A rotina seleciona, quando disponíveis, dois exemplos com classe 1 e dois exemplos com classe 0 por grupo, priorizando as maiores probabilidades.

---

## Entradas e saídas

### Entradas de treinamento

| Artefato | Configuração | Descrição |
|---|---|---|
| Base histórica | `TREINO_BLOB_NAME` | Extração mensal com alvo e variáveis explicativas |

### Entradas de inferência

| Artefato | Configuração | Descrição |
|---|---|---|
| Base de ativos | `ATIVOS_BLOB_NAME` | População mensal que receberá o score |
| Modelo final | `FINAL_MODEL_PATH` | Pipeline treinado e serializado |

### Saídas de treinamento

| Artefato | Configuração | Descrição |
|---|---|---|
| Treino processado | `TREINO_PATH` | Conjunto de treino após split e balanceamento |
| Teste | `TESTE_PATH` | Conjunto separado antes do balanceamento |
| Base preprocessada | `PREPROCESSADO_PATH` | Base de treino processada |
| Base filtrada | `PREPROCESSADO_FILTRADO_PATH` | Base após seleção por correlação |
| Variáveis | `FEATURES_PATH` | Lista gerada pela seleção |
| Melhor modelo | `BEST_MODEL_PATH` | Pipeline retornado pela tunagem |
| Melhores parâmetros | `BEST_PARAMS_PATH` | Parâmetros encontrados pelo Hyperopt |
| Modelo final | `FINAL_MODEL_PATH` | Pipeline usado na inferência |
| Métricas | `EVAL_PATH` | Métricas em JSON |
| Relatório | `relatorio_final.pdf` | Relatório de avaliação |

### Saídas de inferência e análise

| Artefato | Configuração | Descrição |
|---|---|---|
| Scoring | `SCORED_PATH` | Identificador, classe, probabilidade e grupos |
| Distribuição | `BIN_DIST_PATH` | Contagem por faixa de score |
| Relatório Excel | `ANALYSIS_XLSX_PATH` | Análises de perfis e limites |

---

## Variáveis do modelo

### Variáveis numéricas configuradas

```text
VLR_IS
VLR_PREMIO
QTD_VIDAS_SEGURADAS
Beneficiarios
VLR_RENDA_INDIVIDUAL
```

### Variáveis categóricas configuradas

```text
COD_PRODUTO
NOM_REGIAO_AGENCIA
DES_ESTADO_CIVIL
IND_GENERO
COD_CANAL_VENDA
DES_FORMA_PAGAMENTO
```

### Variáveis adicionais do pós-processamento

```text
NUM_CERTIFICADO
QTD_MESES_ATIVO
VLR_PREMIO
```

> `QTD_MESES_ATIVO` é necessária para o pós-processamento, embora não esteja na lista de variáveis preditivas configuradas. Ela deve existir nas bases de treino e ativos para que `08_post_process.py` funcione.

---

## Modelos suportados

| Valor de `MODELO_BASE` | Classe |
|---|---|
| `xgboost` | `XGBClassifier` |
| `randomforest` | `RandomForestClassifier` |
| `logisticregression` | `LogisticRegression` |
| `extratrees` | `ExtraTreesClassifier` |
| `gradientboosting` | `GradientBoostingClassifier` |
| `lgbm` | `LGBMClassifier` |

O modelo ativo é XGBoost.

---

## Métricas e avaliação

Métricas calculadas:

- F1-score;
- precisão;
- revocação;
- acurácia;
- ROC AUC;
- log loss;
- curva de calibração para regressão logística;
- PR AUC para LightGBM.

Gráficos:

- matriz de confusão;
- curva ROC.

O relatório PDF contém:

- modelo utilizado;
- dimensões da base;
- colunas consideradas;
- configurações de VIF, correlação, balanceamento e pesos;
- métricas;
- matriz de confusão;
- curva ROC.

---

## Logs e rastreabilidade

Arquivos:

```text
logs/exec_train_YYYYMMDD.log
logs/exec_pred_YYYYMMDD.log
```

Comportamento dos orquestradores:

- execução sequencial;
- gravação consolidada dos logs;
- interrupção na primeira etapa com código de erro;
- saída do treinamento exibida no console e gravada em arquivo;
- saída da inferência direcionada ao arquivo de log.

Acompanhamento:

```bash
tail -f logs/exec_train_YYYYMMDD.log
```

```bash
tail -f logs/exec_pred_YYYYMMDD.log
```

---

## Segurança

### Segredos

Nunca versione:

- connection strings;
- chaves de armazenamento;
- SAS Tokens;
- senhas;
- segredos de aplicações;
- certificados privados.

O código analisado possui credenciais embutidas em `config.py`. Faça a rotação antes de publicar o repositório.

### Variáveis de ambiente

Exemplo:

```python
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
SAS_TOKEN = os.getenv("AZURE_STORAGE_SAS_TOKEN")

if not SAS_TOKEN:
    raise ValueError("AZURE_STORAGE_SAS_TOKEN não configurada.")
```

### `.gitignore` recomendado

```gitignore
# Ambiente e segredos
.env
*.env
.venv/
venv/

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# Dados e artefatos
*.csv
*.txt
*.pkl
*.joblib
*.xlsx
*.pdf
*.png

# Logs
logs/
*.log

# IDE e sistema
.vscode/
.idea/
.DS_Store
```

---

## Pontos de atenção técnicos

### 1. Credenciais no código

O `config.py` contém segredos em texto claro. Remova, rotacione e consulte-os por mecanismo seguro.

### 2. Avaliação usa a base de treino processada

`02_preprocess.py` cria `TESTE_PATH`, mas `06_evaluate.py` utiliza `PREPROCESSADO_PATH`. Assim, a base reservada para teste não é usada pela avaliação atual.

### 3. Dupla divisão dos dados

O pré-processamento cria treino e teste. Depois, `04_tune.py` divide novamente o treino para tunagem. Isso é válido como estrutura de desenvolvimento, mas a avaliação final deve utilizar exclusivamente a base reservada em `TESTE_PATH`.

### 4. Hyperopt avalia no próprio conjunto de ajuste

A função objetivo treina em `X_train` e calcula F1 macro também em `X_train`. Isso favorece configurações com sobreajuste. Use validação cruzada ou `X_test` da tunagem.

### 5. Seleção de variáveis não afeta o modelo final

`03_correlation_filter.py` gera `features.csv`, mas `04_tune.py`, `05_train.py`, `06_evaluate.py` e `07_score.py` usam `NUMERIC_COLS + CATEGORICAL_COLS` diretamente.

### 6. Configuração de correlação desativada, etapa executada

`SELECAO_VARIAVEIS["usar_correlacao"]` está desativada, porém `main_train.py` executa `03_correlation_filter.py` sem consultar essa configuração.

### 7. Sobrescrita da extração local de treino

`02_preprocess.py` escreve a base rebalanceada novamente em `TREINO_PATH`. Preserve uma cópia imutável da extração original para auditoria e reprocessamento.

### 8. Rebalanceamento antes do Pipeline

O balanceamento é aplicado sobre o DataFrame bruto. SMOTE exige dados numéricos e não funcionará diretamente com as categóricas atuais. Para dados mistos, avalie SMOTENC ou aplique transformação compatível antes da reamostragem.

### 9. Pesos calculados, mas não persistidos

O pré-processamento calcula `pesos_train` quando habilitado, mas não salva nem utiliza o resultado no treinamento final.

### 10. Suavização configurada, mas não aplicada

`USE_SCORE_SMOOTHING`, `SMOOTHING_METHOD`, `TEMPERATURE`, `LAMBDA_SHRINK` e `EPSILON` estão definidos, mas `07_score.py` não aplica suavização às probabilidades.

### 11. Separador do scoring após pós-processamento

`08_post_process.py` sobrescreve o scoring com separador `|`. `09_bin_distribution.py` usa `pd.read_csv` sem informar esse separador, portanto pode interpretar o arquivo como uma única coluna.

### 12. Distribuição não faz parte do fluxo principal

`09_bin_distribution.py` não está no `main_predict.py`. Execute-o separadamente ou inclua-o de forma explícita no orquestrador, após alinhar o separador.

### 13. Relatório Excel não faz parte do fluxo principal

`analysis.py` também precisa ser executado separadamente. O arquivo não está listado em `main_predict.py`.

### 14. Título do gráfico incorreto

`09_bin_distribution.py` usa “Distribuição dos Scores de Resgate”. Altere para “Distribuição dos Scores de Inadimplência Vida por Faixa (%)”.

### 15. Possível inconsistência na tipagem categórica do scoring

`astype(str).fillna("DESCONHECIDO")` converte valores ausentes em texto antes do `fillna`, podendo gerar a string `nan`. Prefira preencher antes da conversão.

```python
X[c] = X[c].fillna("DESCONHECIDO").astype(str)
```

### 16. Colunas ausentes na avaliação

`06_evaluate.py` cria a lista de colunas ausentes depois de selecionar `df[NUMERIC_COLS + CATEGORICAL_COLS]`. Se uma coluna não existir, a seleção falhará antes da rotina de criação. Crie as colunas antes de selecionar e ordenar.

### 17. Descrição do pós-processamento

Os comentários indicam que as colunas analíticas devem existir no treino e no “scored”, mas a implementação lê essas colunas na base de ativos e faz merge no scored pelo identificador. A documentação operacional deve refletir o comportamento real.

### 18. `COLS_TO_REMOVE` não utilizado

A configuração é importada no treinamento, mas não é aplicada ao DataFrame.

### 19. Parâmetros de validação cruzada não utilizados

`CV_PARAMS` está configurado, porém a tunagem atual não executa validação cruzada.

### 20. Dependência de `n_jobs` no SMOTE

Dependendo da versão do imbalanced-learn, o parâmetro `n_jobs` pode não ser aceito por `SMOTE`. Valide a versão fixada no ambiente.

---

## Solução de problemas

### Arquivo não encontrado

Verifique:

- `MAIN_DIR`;
- `PROJECT_NAME`;
- diretório atual;
- disponibilidade dos arquivos no período;
- existência dos artefatos das etapas anteriores.

### Erro de coluna ausente

Confirme:

- `TARGET`;
- `ID_TRACKING`;
- `NUMERIC_COLS`;
- `CATEGORICAL_COLS`;
- `VLR_PREMIO`;
- `QTD_MESES_ATIVO`;
- capitalização dos cabeçalhos.

### Erro de leitura do CSV

Revise:

- separador;
- codificação;
- linhas malformadas;
- uso de `|` após o pós-processamento;
- vírgula decimal e ponto de milhar.

### Erro no Blob Storage

Valide:

- URL do container;
- validade do SAS Token;
- permissões de leitura e escrita;
- nome completo do blob;
- conectividade e proxy.

### Erro ao carregar o modelo

Valide:

- existência de `modelo_final.pkl`;
- compatibilidade entre versões;
- disponibilidade de `custom_transformers.py` durante a desserialização;
- integridade do artefato.

### Erro no Hyperopt

Consulte as mensagens de cada tentativa. O script registra falhas e retorna `STATUS_FAIL`. Verifique compatibilidade dos parâmetros com o algoritmo selecionado.

### Erro na geração do PDF

O FPDF clássico pode apresentar limitações com caracteres Unicode. Utilize fonte compatível ou normalize os textos.

### Erro no pós-processamento

Confirme a presença de:

```text
NUM_CERTIFICADO
VLR_PREMIO
QTD_MESES_ATIVO
```

Os quartis também exigem valores numéricos válidos e quantidade suficiente de valores distintos.

---

## Checklist operacional

### Antes do treinamento

- [ ] Credenciais removidas do código e rotacionadas.
- [ ] `ANO_MES` validado.
- [ ] Arquivo histórico disponível no Blob Storage.
- [ ] Diretório principal detectado corretamente.
- [ ] Target validado.
- [ ] Identificador validado.
- [ ] Variáveis numéricas e categóricas revisadas.
- [ ] Categorias válidas revisadas com o negócio.
- [ ] Estratégia de balanceamento revisada.
- [ ] Espaço de busca revisado.
- [ ] Ambiente e dependências validados.
- [ ] Política de preservação da extração original definida.

### Depois do treinamento

- [ ] Pipeline concluído sem erros.
- [ ] Logs revisados.
- [ ] Melhores parâmetros revisados.
- [ ] Modelo final gerado.
- [ ] Avaliação realizada em base independente.
- [ ] Matriz de confusão revisada.
- [ ] Curva ROC revisada.
- [ ] Artefatos versionados.
- [ ] Aprovação registrada antes de produção.

### Antes da inferência

- [ ] Base de ativos disponível.
- [ ] Modelo final disponível.
- [ ] Esquema dos ativos validado.
- [ ] Identificador presente.
- [ ] `VLR_PREMIO` presente.
- [ ] `QTD_MESES_ATIVO` presente.
- [ ] Destino do upload confirmado.
- [ ] Período de referência confirmado.

### Depois da inferência

- [ ] Quantidade de entrada e saída reconciliada.
- [ ] IDs incrementais não foram criados indevidamente.
- [ ] Classes e probabilidades validadas.
- [ ] Probabilidades entre 0 e 1.
- [ ] Distribuição por grupo revisada.
- [ ] Registros sem correspondência no pós-processamento revisados.
- [ ] Arquivo publicado no caminho correto.
- [ ] Log arquivado.
- [ ] Relatório analítico gerado, quando aplicável.

---

## Recomendações de evolução

1. avaliar o modelo final exclusivamente em `TESTE_PATH`;
2. usar validação cruzada no Hyperopt;
3. preservar a extração original em caminho imutável;
4. remover ou integrar efetivamente a seleção por correlação;
5. tornar o balanceamento compatível com dados categóricos;
6. persistir e utilizar pesos por amostra quando habilitados;
7. implementar ou remover a configuração de suavização;
8. padronizar o separador dos artefatos;
9. integrar distribuição e relatório Excel ao orquestrador, se fizerem parte do processo oficial;
10. adicionar validação de esquema, tipos, duplicidades, nulos e faixas;
11. incluir testes unitários e de integração;
12. substituir `print` por logging estruturado em todos os scripts;
13. registrar versão de código, dados, parâmetros e artefatos;
14. monitorar mudança de distribuição, estabilidade do score e desempenho;
15. mover segredos para Azure Key Vault ou identidade gerenciada;
16. automatizar a execução em jobs ou pipelines do Azure ML;
17. adicionar critérios formais de promoção, rollback e retreinamento.

---

## Governança

Alterações na população, variável-alvo, variáveis explicativas, categorias válidas, regras de balanceamento, parâmetros, quartis, grupos, limites de probabilidade ou arquivos publicados devem ser documentadas e validadas antes da promoção para produção.

O uso do score deve respeitar as políticas internas de segurança, privacidade, governança de dados, gestão de modelos e controles aplicáveis ao processo de negócio.
