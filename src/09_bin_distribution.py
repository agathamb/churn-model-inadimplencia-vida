import pandas as pd
import matplotlib.pyplot as plt
import os
from config import ( SCORED_PATH, BIN_DIST_PATH)

def distribuir_bins_scoring():
    print(f"📥 Lendo arquivo: {SCORED_PATH}")
    df = pd.read_csv(SCORED_PATH)

    # Validar colunas esperadas
    if not {"Scored Probabilities", "NUM_CERTIFICADO"}.issubset(df.columns):
        raise ValueError("❌ Colunas esperadas não encontradas no arquivo de scoring.")

    print("🔢 Calculando bins de score (5%)...")
    df["ScoreBin"] = pd.cut(
        df["Scored Probabilities"],
        bins=[i / 20 for i in range(21)],
        labels=[f"{i*5}-{(i+1)*5}%" for i in range(20)],
        include_lowest=True
    )

    # Tabela de distribuição
    dist = df["ScoreBin"].value_counts().sort_index().reset_index()
    dist.columns = ["ScoreBin", "Quantidade"]

    print(f"💾 Salvando distribuição em: {BIN_DIST_PATH}")
    os.makedirs(os.path.dirname(BIN_DIST_PATH), exist_ok=True)
    dist.to_csv(BIN_DIST_PATH, index=False)

    print("📊 Gerando gráfico...")
    plt.figure(figsize=(10, 6))
    plt.bar(dist["ScoreBin"], dist["Quantidade"], color="#1C60AB")
    plt.xticks(rotation=45)
    plt.title("Distribuição dos Scores de Resgate por Faixa (%)", fontsize=14)
    plt.xlabel("Faixa de Score (%)")
    plt.ylabel("Quantidade de Certificados")
    plt.tight_layout()
    plt.savefig(BIN_DIST_PATH.replace(".csv", ".png"))
    print(f"📈 Gráfico salvo em: {BIN_DIST_PATH.replace('.csv', '.png')}")

if __name__ == "__main__":
    distribuir_bins_scoring()
