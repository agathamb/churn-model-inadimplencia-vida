# ============================================================
# analysis.py
# Análises enxutas para negócio sobre perfis de inadimplência:
# - Limites de quartis do treino (com meses negativos tratados só na estatística)
# - Quantitativos por Grupo x Scored Label
# - Estatísticas simples de probabilidade por Grupo x Label
# - Contagens acima de thresholds (0.65/0.70/0.80/0.90) por Label e por Grupo x Label
# - Lista para investigação de meses negativos
# - Amostra (4 por grupo: 2 label=1 e 2 label=0)
# Saída: Excel com abas
# ============================================================

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------------
# Leitura CSV (preferir '|', fallback ',')
# -------------------------
def read_csv_prefer_pipe(path: str) -> pd.DataFrame:
    path = str(path)
    logger.info(f"Lendo arquivo (tentativa sep='|'): {path}")
    df = pd.read_csv(path, sep="|", dtype=str)

    if df.shape[1] == 1:
        logger.warning(f"Leitura com '|' resultou em 1 coluna. Fallback sep=',' em {path}")
        df = pd.read_csv(path, sep=",", dtype=str)

    logger.info(f"Arquivo carregado: {path} | shape={df.shape}")
    logger.info(f"Colunas ({len(df.columns)}): {list(df.columns)[:30]}{'...' if len(df.columns)>30 else ''}")
    return df

# -------------------------
# Conversão numérica BR (milhar '.' e decimal ',')
# -------------------------
def to_numeric_br(series: pd.Series, col_name: str) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s2 = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    out = pd.to_numeric(s2, errors="coerce")
    logger.info(f"Conversão numérica {col_name}: NaNs={out.isna().sum()} / {len(out)}")
    return out

def safe_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""

# -------------------------
# Quartis do treino (com regra: meses negativos não entram nas estatísticas)
# -------------------------
def quartile_limits_with_sanity(train_df: pd.DataFrame, col: str, non_negative_only: bool = False) -> dict:
    s_orig = train_df[col]
    neg_count = int((s_orig < 0).sum()) if non_negative_only else 0

    if non_negative_only:
        s = s_orig.where(s_orig >= 0).dropna()
    else:
        s = s_orig.dropna()

    if s.empty:
        return {
            "coluna": col, "q1": np.nan, "q2": np.nan, "q3": np.nan,
            "min": np.nan, "max": np.nan, "n": 0, "negativos": neg_count
        }

    return {
        "coluna": col,
        "q1": float(np.nanquantile(s, 0.25)),
        "q2": float(np.nanquantile(s, 0.50)),
        "q3": float(np.nanquantile(s, 0.75)),
        "min": float(np.nanmin(s)),
        "max": float(np.nanmax(s)),
        "n": int(s.shape[0]),
        "negativos": neg_count
    }

# -------------------------
# Tabelas de thresholds (>= 0.65/0.70/0.80/0.90)
# -------------------------
THRESHOLDS = [0.65, 0.70, 0.80, 0.90]

def threshold_table(df: pd.DataFrame, group_cols: list[str], prob_col: str, id_col: str) -> pd.DataFrame:
    """
    Retorna contagem e % de certificados únicos acima dos thresholds,
    agrupando pelas colunas group_cols.
    """
    base = df.drop_duplicates(subset=[id_col]).copy()

    # total por grupo
    totals = base.groupby(group_cols)[id_col].nunique().reset_index(name="total_certificados")

    rows = []
    for t in THRESHOLDS:
        tmp = (
            base[base[prob_col].notna() & (base[prob_col] >= t)]
            .groupby(group_cols)[id_col]
            .nunique()
            .reset_index(name=f"qtd_ge_{t:.2f}")
        )
        rows.append(tmp)

    # merge tudo
    out = totals
    for r in rows:
        out = out.merge(r, on=group_cols, how="left")

    # preencher NaNs com 0 e calcular percentuais
    for t in THRESHOLDS:
        c = f"qtd_ge_{t:.2f}"
        out[c] = out[c].fillna(0).astype(int)
        out[f"pct_ge_{t:.2f}"] = (out[c] / out["total_certificados"]).replace([np.inf, -np.inf], np.nan)

    return out

# -------------------------
# Amostra por grupo e label (4 por grupo: 2 label=1 e 2 label=0)
# -------------------------
def sample_por_grupo_e_label(
    df: pd.DataFrame,
    grupo_col: str,
    desc_col: str,
    label_col: str,
    id_col: str,
    prob_col: str,
    n_por_label: int = 2
) -> pd.DataFrame:
    """
    Para cada grupo, pega n_por_label casos com label=1 e n_por_label casos com label=0.
    Ordena por probabilidade desc para trazer exemplos mais informativos.
    """
    base = df.drop_duplicates(subset=[id_col]).copy()

    # garantir tipo string para comparação consistente
    base[label_col] = base[label_col].astype(str).str.strip()
    base[grupo_col] = pd.to_numeric(base[grupo_col], errors="coerce").fillna(0).astype(int)

    samples = []
    grupos = sorted(base[grupo_col].dropna().unique())

    for g in grupos:
        df_g = base[base[grupo_col] == g]

        for lab in ["1", "0"]:  # primeiro 1, depois 0 (pode inverter se quiser)
            df_gl = df_g[df_g[label_col] == lab]
            if df_gl.empty:
                continue

            df_gl = df_gl.sort_values(prob_col, ascending=False)
            samples.append(df_gl.head(n_por_label))

    if samples:
        out = pd.concat(samples, ignore_index=True)
    else:
        out = pd.DataFrame(columns=[id_col, grupo_col, desc_col, label_col, prob_col])

    # Selecionar colunas finais mais úteis
    cols_keep = [id_col, grupo_col, desc_col, label_col, prob_col]
    cols_keep = [c for c in cols_keep if c in out.columns]
    out = out[cols_keep].copy()

    # Ordenação amigável
    out = out.sort_values([grupo_col, label_col, prob_col], ascending=[True, False, False])

    return out

# -------------------------
# Build do Excel
# -------------------------
def build_tabs(scored_df: pd.DataFrame, train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    id_col = getattr(config, "ID_TRACKING", "NUM_CERTIFICADO")
    grupo_col = getattr(config, "COL_GRUPO", "grupo")
    desc_col  = getattr(config, "COL_DESCRICAO_GRUPO", "descricao_grupo")

    # ✅ do jeito que você pediu
    label_col = safe_col(scored_df, ["Scored Labels", "scored_label", "label", "SCORED_LABEL", "pred_label"])
    prob_raw_col = safe_col(scored_df, ["Scored Probabilities", "scored_probability", "probability", "SCORED_PROB", "pred_proba", "score"])

    if not label_col:
        raise ValueError(f"Não encontrei coluna de label no scored (ex.: 'Scored Labels'). Colunas: {list(scored_df.columns)[:50]}")
    if not prob_raw_col:
        raise ValueError(f"Não encontrei coluna de prob no scored (ex.: 'Scored Probabilities'). Colunas: {list(scored_df.columns)[:50]}")
    if id_col not in scored_df.columns:
        raise ValueError(f"Não encontrei coluna ID ({id_col}) no scored. Colunas: {list(scored_df.columns)[:50]}")

    scored = scored_df.copy()
    scored[label_col] = scored[label_col].astype(str).str.strip()

    # Probabilidade: assumindo 0..1 (0.65, 0.7, etc.)
    scored["prob_0_1"] = to_numeric_br(scored[prob_raw_col], prob_raw_col).clip(lower=0.0, upper=1.0)

    # Grupo e descrição
    if grupo_col not in scored.columns:
        scored[grupo_col] = 0
    else:
        scored[grupo_col] = to_numeric_br(scored[grupo_col], grupo_col).fillna(0).astype(int)

    if desc_col not in scored.columns:
        scored[desc_col] = "Grupo 0 – Cliente médio da carteira"

    # -------------------------
    # 01) Limites quartis do treino
    # -------------------------
    col_valor = getattr(config, "COL_VALOR_PAGO_SEGURO", "VLR_PREMIO")
    col_tempo = getattr(config, "COL_TEMPO_PERMANENCIA_MESES", "QTD_MESES_ATIVO")

    train = train_df.copy()
    if col_valor in train.columns:
        train[col_valor] = to_numeric_br(train[col_valor], col_valor)
    if col_tempo in train.columns:
        train[col_tempo] = to_numeric_br(train[col_tempo], col_tempo)

    df_quartis = pd.DataFrame([
        quartile_limits_with_sanity(train, col_valor, non_negative_only=False) if col_valor in train.columns else
            {"coluna": col_valor, "q1": np.nan, "q2": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan, "n": 0, "negativos": 0},
        quartile_limits_with_sanity(train, col_tempo, non_negative_only=True) if col_tempo in train.columns else
            {"coluna": col_tempo, "q1": np.nan, "q2": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan, "n": 0, "negativos": 0},
    ])

    # -------------------------
    # 02) Qtd por Grupo x Label (pivot)
    # -------------------------
    df_gl = (
        scored
        .drop_duplicates(subset=[id_col])
        .groupby([grupo_col, desc_col, label_col], dropna=False)[id_col]
        .nunique()
        .reset_index(name="qtd_certificados")
        .sort_values([grupo_col, label_col])
    )

    df_gl_pivot = (
        df_gl
        .pivot_table(index=[grupo_col, desc_col], columns=label_col, values="qtd_certificados", fill_value=0, aggfunc="sum")
        .reset_index()
    )

    # -------------------------
    # 03) Prob simples por Grupo x Label (média/mediana/p90)
    # -------------------------
    def p90(x):
        x = x.dropna()
        return float(np.nanquantile(x, 0.90)) if len(x) else np.nan

    df_prob_gl = (
        scored
        .groupby([grupo_col, desc_col, label_col], dropna=False)["prob_0_1"]
        .agg(
            n="count",
            media="mean",
            mediana="median",
            p90=p90,
            maximo="max"
        )
        .reset_index()
        .sort_values([grupo_col, label_col])
    )

    for c in ["media", "mediana", "p90", "maximo"]:
        df_prob_gl[f"{c}_%"] = df_prob_gl[c] * 100

    # -------------------------
    # 04) Thresholds por Label
    # -------------------------
    df_thr_label = threshold_table(
        df=scored,
        group_cols=[label_col],
        prob_col="prob_0_1",
        id_col=id_col
    ).sort_values([label_col])

    # -------------------------
    # 05) Thresholds por Grupo x Label
    # -------------------------
    df_thr_gl = threshold_table(
        df=scored,
        group_cols=[grupo_col, desc_col, label_col],
        prob_col="prob_0_1",
        id_col=id_col
    ).sort_values([grupo_col, label_col])

    # -------------------------
    # 06) Investigação: certificados com meses negativos no treino
    # -------------------------
    if (col_tempo in train.columns) and (id_col in train.columns):
        df_neg = train.loc[train[col_tempo] < 0, [id_col, col_tempo]].copy()
        df_neg = df_neg.drop_duplicates(subset=[id_col]).sort_values(col_tempo)
    else:
        df_neg = pd.DataFrame(columns=[id_col, col_tempo])

    # -------------------------
    # 07) Amostra por grupo (4 casos: 2 label=1 e 2 label=0)
    # -------------------------
    df_amostra = sample_por_grupo_e_label(
        df=scored,
        grupo_col=grupo_col,
        desc_col=desc_col,
        label_col=label_col,
        id_col=id_col,
        prob_col="prob_0_1",
        n_por_label=2
    )

    return {
        "01_limites_quartis_treino": df_quartis,
        "02_qtd_por_grupo_e_label": df_gl_pivot,
        "03_prob_media_por_grupo_e_label": df_prob_gl,
        "04_thresholds_por_label": df_thr_label,
        "05_thresholds_por_grupo_e_label": df_thr_gl,
        "06_investigar_meses_negativos": df_neg,
        "07_amostra_por_grupo_label": df_amostra,
    }

def save_excel(tabs: dict[str, pd.DataFrame], output_path: str):
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Salvando Excel: {output_path}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in tabs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    logger.info("Excel salvo com sucesso.")

def main():
    logger.info("===== INÍCIO analysis.py =====")

    scored_path = getattr(config, "SCORED_PATH", None)
    treino_path = getattr(config, "TREINO_PATH", None)
    if not scored_path or not treino_path:
        raise ValueError("config precisa ter SCORED_PATH e TREINO_PATH definidos.")

    default_out = str(Path(scored_path).parent / "analysis.xlsx")
    out_path = getattr(config, "ANALYSIS_XLSX_PATH", default_out)

    scored_df = read_csv_prefer_pipe(scored_path)
    train_df  = read_csv_prefer_pipe(treino_path)

    tabs = build_tabs(scored_df, train_df)
    save_excel(tabs, out_path)

    logger.info(f"===== FIM analysis.py | arquivo gerado: {out_path} =====")

if __name__ == "__main__":
    main()
