# ============================================================
# 08_post_process.py
# Pós-processamento (pós-score):
# - Aprende quartis no TREINO
# - Classifica ATIVOS
# - Aplica classificação no SCORED (merge por ID)
# - Cria colunas: grupo (1..4/0) e descricao_grupo
# - Sobrescreve o SCORED
# ============================================================

import csv
import logging
import numpy as np
import pandas as pd
import config


# ------------------------------------------------------------
# Logging (Azure ML friendly)
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def _sniff_delimiter(path: str, default: str = ",") -> str:
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            sample = f.read(8192)
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return default


def _read_csv_auto(path: str) -> pd.DataFrame:
    sep = _sniff_delimiter(path)
    logger.info(f"Lendo arquivo: {path} | separador detectado: '{sep}'")
    df = pd.read_csv(path, sep=sep)
    logger.info(f"Arquivo carregado: {path} | shape={df.shape}")
    logger.info(f"Colunas ({len(df.columns)}): {list(df.columns)[:30]}{'...' if len(df.columns)>30 else ''}")
    return df


def _to_numeric(series: pd.Series, col_name: str) -> pd.Series:
    logger.info(f"Convertendo coluna para numérico: {col_name}")
    return pd.to_numeric(series, errors="coerce")


def _fit_quartile_bins(train_series: pd.Series, nome_coluna: str):
    logger.info(f"Calculando quartis para coluna: {nome_coluna}")

    # Série original (não alterada)
    s_original = train_series.copy()

    # ✅ Regra analítica:
    # valores negativos NÃO entram no cálculo de quartis,
    # mas os registros continuam existindo na base
    negativos = s_original < 0
    qtde_negativos = negativos.sum()

    if qtde_negativos > 0:
        logger.warning(
            f"[SANIDADE] {qtde_negativos} registros com valores negativos em "
            f"{nome_coluna} foram ignorados no cálculo de quartis."
        )

    # Série usada APENAS para quartil
    s_quartil = s_original.where(s_original >= 0)

    s = s_quartil.dropna()
    logger.info(f"Qtd valores válidos para quartil (>=0 e sem NaN): {len(s)}")

    if s.empty:
        raise ValueError(
            f"Série vazia para cálculo de quartis após remover valores inválidos: {nome_coluna}"
        )

    # Cálculo dos quartis
    _, bins = pd.qcut(s, q=4, retbins=True, duplicates="drop")
    bins = np.unique(np.array(bins, dtype=float))
    logger.info(f"Bins calculados (antes ajuste): {bins}")

    if len(bins) < 3:
        raise ValueError(f"Bins insuficientes para quartis na coluna {nome_coluna}")

    # Ajuste de limites
    bins[0] = -np.inf
    bins[-1] = np.inf
    labels = list(range(1, len(bins)))

    logger.info(f"Bins finais: {bins}")
    logger.info(f"Labels: {labels}")

    return bins, labels

# ------------------------------------------------------------
# Função principal
# ------------------------------------------------------------
def quartiles_analisys(treino_path: str = None, ativos_path: str = None, scored_path: str = None) -> pd.DataFrame:
    logger.info("===== INÍCIO DO PÓS-PROCESSAMENTO (quartiles_analisys) =====")

    treino_path = str(treino_path or config.TREINO_PATH)
    ativos_path = str(ativos_path or config.ATIVOS_PATH)
    scored_path = str(scored_path or config.SCORED_PATH)

    logger.info(f"TREINO_PATH : {treino_path}")
    logger.info(f"ATIVOS_PATH : {ativos_path}")
    logger.info(f"SCORED_PATH : {scored_path}")

    id_col = config.ID_TRACKING
    col_valor = config.COL_VALOR_PAGO_SEGURO
    col_tempo = config.COL_TEMPO_PERMANENCIA_MESES

    # Colunas de saída (nome final desejado)
    grupo_col = config.COL_GRUPO          # recomendo: "grupo"
    desc_col = config.COL_DESCRICAO_GRUPO # recomendo: "descricao_grupo"

    logger.info(f"ID_TRACKING               : {id_col}")
    logger.info(f"COL_VALOR_PAGO_SEGURO     : {col_valor}")
    logger.info(f"COL_TEMPO_PERMANENCIA     : {col_tempo}")
    logger.info(f"COL_GRUPO (saída)         : {grupo_col}")
    logger.info(f"COL_DESCRICAO (saída)     : {desc_col}")

    # -------------------------
    # 1) Ler bases
    # -------------------------
    df_train = _read_csv_auto(treino_path)
    df_ativos = _read_csv_auto(ativos_path)
    df_scored = _read_csv_auto(scored_path)

    # -------------------------
    # 2) Validar colunas
    # -------------------------
    logger.info("Validando colunas obrigatórias...")

    for nome, df, cols in [
        ("TREINO", df_train, [col_valor, col_tempo]),
        ("ATIVOS", df_ativos, [id_col, col_valor, col_tempo]),
        ("SCORED", df_scored, [id_col]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.error(f"Colunas ausentes em {nome}: {missing}")
            logger.error(f"Colunas disponíveis em {nome}: {list(df.columns)[:50]}")
            raise ValueError(f"Colunas ausentes em {nome}: {missing}")
        logger.info(f"Colunas OK em {nome}")

    # -------------------------
    # 3) Converter numéricas
    # -------------------------
    df_train[col_valor] = _to_numeric(df_train[col_valor], col_valor)
    df_train[col_tempo] = _to_numeric(df_train[col_tempo], col_tempo)
    df_ativos[col_valor] = _to_numeric(df_ativos[col_valor], col_valor)
    df_ativos[col_tempo] = _to_numeric(df_ativos[col_tempo], col_tempo)

    # -------------------------
    # 4) Aprender bins no treino
    # -------------------------
    bins_valor, labels_valor = _fit_quartile_bins(df_train[col_valor], col_valor)
    bins_tempo, labels_tempo = _fit_quartile_bins(df_train[col_tempo], col_tempo)

    # -------------------------
    # 5) Classificar ATIVOS
    # -------------------------
    logger.info("Classificando ATIVOS em quartis e grupos...")

    q_valor = pd.cut(df_ativos[col_valor], bins=bins_valor, labels=labels_valor, include_lowest=True)
    q_tempo = pd.cut(df_ativos[col_tempo], bins=bins_tempo, labels=labels_tempo, include_lowest=True)

    qv = q_valor.astype("float")
    qt = q_tempo.astype("float")

    cond_g1 = (qv == 4) & (qt == 4)
    cond_g2 = (qv == 4) & (qt.isin([2, 3]))
    cond_g3 = (qt == 4) & (qv == 1)
    cond_g4 = (qv == 4) & (qt == 1)

    grupo = np.select(
        [cond_g1, cond_g2, cond_g3, cond_g4],
        [1, 2, 3, 4],
        default=getattr(config, "GRUPO_DEFAULT", 0)
    ).astype(int)

    logger.info("Distribuição de grupos em ATIVOS (linhas):")
    logger.info(pd.Series(grupo).value_counts().sort_index())

    # ✅ Quantidade de certificados únicos por grupo (ATIVOS)
    cert_por_grupo_ativos = (
        pd.DataFrame({id_col: df_ativos[id_col], "grupo_tmp": grupo})
        .drop_duplicates(subset=[id_col])
        .groupby("grupo_tmp")[id_col]
        .nunique()
        .sort_index()
    )
    logger.info("Quantidade de CERTIFICADOS ÚNICOS por grupo (ATIVOS):")
    for g, qtd in cert_por_grupo_ativos.items():
        logger.info(f"  Grupo {int(g)}: {int(qtd)} certificados")

    # DataFrame de grupos para merge
    df_grp = df_ativos[[id_col]].copy()
    df_grp[grupo_col] = grupo
    df_grp[desc_col] = (
        pd.Series(grupo)
        .map(getattr(config, "DESCRICOES_GRUPOS", {}))
        .fillna(getattr(config, "DESCRICAO_GRUPO_DEFAULT", "Outros"))
        .values
    )

    # se ATIVOS tiver múltiplas linhas por certificado, mantém 1 por ID
    df_grp = df_grp.drop_duplicates(subset=[id_col], keep="first")
    logger.info(f"DF de grupos criado | shape={df_grp.shape}")
    logger.info(f"Colunas DF grupos: {list(df_grp.columns)}")

    # -------------------------
    # 6) Merge no SCORED (corrigindo conflito de colunas)
    # -------------------------
    logger.info("Aplicando grupos no SCORED (merge por ID)...")
    logger.info(f"Colunas do SCORED antes do tratamento: {list(df_scored.columns)}")

    # ✅ Se o scored já tiver colunas de grupo/descrição (execução anterior), remove para evitar sufixos _x/_y
    cols_para_remover = [c for c in [grupo_col, desc_col] if c in df_scored.columns]
    if cols_para_remover:
        logger.warning(f"SCORED já possui colunas {cols_para_remover}. Removendo antes do merge para evitar sufixos.")
        df_scored = df_scored.drop(columns=cols_para_remover)

    logger.info(f"Colunas do SCORED após remoção (se houve): {list(df_scored.columns)}")

    # Merge
    df_scored_out = df_scored.merge(df_grp, on=id_col, how="left", suffixes=("", "_ativos"))
    logger.info(f"Merge concluído | shape={df_scored_out.shape}")
    logger.info(f"Colunas após merge: {list(df_scored_out.columns)}")

    # ✅ Robustez extra: se ainda assim vier sufixado, normaliza
    if grupo_col not in df_scored_out.columns:
        alt = f"{grupo_col}_ativos"
        if alt in df_scored_out.columns:
            logger.warning(f"Coluna {grupo_col} não encontrada; usando {alt} e renomeando.")
            df_scored_out = df_scored_out.rename(columns={alt: grupo_col})

    if desc_col not in df_scored_out.columns:
        alt = f"{desc_col}_ativos"
        if alt in df_scored_out.columns:
            logger.warning(f"Coluna {desc_col} não encontrada; usando {alt} e renomeando.")
            df_scored_out = df_scored_out.rename(columns={alt: desc_col})

    # Preencher defaults quando não houver match de ID
    df_scored_out[grupo_col] = df_scored_out[grupo_col].fillna(getattr(config, "GRUPO_DEFAULT", 0)).astype(int)
    df_scored_out[desc_col] = df_scored_out[desc_col].fillna(getattr(config, "DESCRICAO_GRUPO_DEFAULT", "Outros"))

    logger.info("Distribuição final de grupos no SCORED (linhas):")
    logger.info(df_scored_out[grupo_col].value_counts().sort_index())

    # ✅ Quantidade de certificados únicos por grupo (SCORED)
    cert_por_grupo_scored = (
        df_scored_out
        .drop_duplicates(subset=[id_col])
        .groupby(grupo_col)[id_col]
        .nunique()
        .sort_index()
    )
    logger.info("Quantidade de CERTIFICADOS ÚNICOS por grupo (SCORED):")
    for g, qtd in cert_por_grupo_scored.items():
        logger.info(f"  Grupo {int(g)}: {int(qtd)} certificados")

    # -------------------------
    # 7) Sobrescrever SCORED
    # -------------------------
    sep_out = "|" #_sniff_delimiter(scored_path)
    logger.info(f"Sobrescrevendo SCORED em: {scored_path} | sep='{sep_out}'")
    df_scored_out.to_csv(scored_path, index=False, sep=sep_out)

    logger.info("===== FIM DO PÓS-PROCESSAMENTO (SUCESSO) =====")
    return df_scored_out


if __name__ == "__main__":
    quartiles_analisys()