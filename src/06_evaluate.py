import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, log_loss, roc_auc_score,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from fpdf import FPDF
from config import (
    PREPROCESSADO_PATH, EVAL_PATH, HTML_RELATORIO_PATH,
    FINAL_MODEL_PATH, TARGET, MODELO_BASE,
    NUMERIC_COLS, CATEGORICAL_COLS,
    USAR_VIF, VIF_THRESHOLD,
    REBALANCEAMENTO, REBALANCEAMENTO_PARAMETROS,
    SELECAO_VARIAVEIS, SAMPLE_WEIGHT_RULES, USE_SAMPLE_WEIGHTS
)

def avaliar_modelo():
    print("📥 Lendo dados e modelo...")
    df = pd.read_csv(PREPROCESSADO_PATH)
    model = joblib.load(FINAL_MODEL_PATH)

    y = df[TARGET]

    # Use as listas do config, como no treino:
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()

    # Robustez: cria colunas ausentes com default
    missing = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c not in X.columns]
    for c in missing:
        if c in NUMERIC_COLS:
            X[c] = 0
        else:
            X[c] = "DESCONHECIDO"
    X = X[NUMERIC_COLS + CATEGORICAL_COLS]

    print("🔍 Gerando previsões...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("📊 Calculando métricas...")
    metrics = {
        "f1_score": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "log_loss": log_loss(y, y_proba),
    }

    if MODELO_BASE.lower() == "logisticregression":
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10)
        metrics["calibration_curve"] = list(zip(prob_pred, prob_true))

    if MODELO_BASE.lower() == "lgbm":
        metrics["pr_auc"] = average_precision_score(y, y_proba)

    # Salva métricas
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Métricas salvas em {EVAL_PATH}")

    print("📈 Gerando gráficos...")
    # Matriz de Confusão
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    cm_path = HTML_RELATORIO_PATH.replace(".html", "_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}", color="#1C60AB")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    roc_path = HTML_RELATORIO_PATH.replace(".html", "_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path)

    # 📄 Geração do PDF (sem SHAP)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(28, 96, 171)  # Azul #1C60AB
    pdf.cell(0, 10, "Relatório de Avaliação do Modelo", ln=1, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Detalhes do Modelo", ln=1)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Modelo utilizado: {MODELO_BASE}", ln=1)
    pdf.cell(0, 8, f"Nº linhas: {df.shape[0]}", ln=1)
    pdf.cell(0, 8, f"Nº colunas: {df.shape[1]}", ln=1)
    pdf.multi_cell(0, 8, f"Colunas consideradas: {', '.join(X.columns)}")

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Configurações do Pipeline", ln=1)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"VIF aplicado? {USAR_VIF} (limite: {VIF_THRESHOLD})", ln=1)
    pdf.cell(0, 8, f"Correlação aplicada? {SELECAO_VARIAVEIS.get('usar_correlacao', False)}", ln=1)
    if SELECAO_VARIAVEIS.get("usar_correlacao", False):
        pdf.cell(0, 8, f"Tipo de correlação: {SELECAO_VARIAVEIS.get('tipo_correlacao', 'Não definido')}", ln=1)
    pdf.cell(0, 8, f"Rebalanceamento: {REBALANCEAMENTO}", ln=1)
    pdf.multi_cell(0, 8, f"Parâmetros: {REBALANCEAMENTO_PARAMETROS.get(REBALANCEAMENTO, {})}")
    if USE_SAMPLE_WEIGHTS:
        pdf.multi_cell(0, 8, f"Regras de pesos aplicadas: {SAMPLE_WEIGHT_RULES}")

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(28, 96, 171)
    pdf.cell(0, 10, "Métricas do Modelo", ln=1)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    for k, v in metrics.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=1)

    # Imagens (sem SHAP)
    pdf.image(cm_path, x=15, w=180)
    pdf.image(roc_path, x=15, w=180)

    pdf.output(HTML_RELATORIO_PATH.replace(".html", ".pdf"))
    print(f"📄 Relatório PDF salvo em: {HTML_RELATORIO_PATH.replace('.html', '.pdf')}")

if __name__ == "__main__":
    avaliar_modelo()
