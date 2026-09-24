import pandas as pd
from azure.storage.blob import BlobClient
from io import BytesIO
import os
import csv

from config import (
    SAS_TOKEN,
    CONTAINER_URL,
    TREINO_BLOB_NAME,
    ATIVOS_BLOB_NAME,
    TREINO_PATH,
    ATIVOS_PATH
)
import warnings
warnings.filterwarnings("ignore")

# ---------- Helpers robustos ----------

def _sniff_sep_from_bytes(blob_bytes: bytes, default=","):
    """
    Detecta o separador a partir do cabeçalho do arquivo.
    Tenta , ; | \t — cai no default se não conseguir inferir.
    """
    head = blob_bytes[:4096]
    sample = head.decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "|", "\t"])
        return dialect.delimiter
    except Exception:
        return default

def _read_csv_robusto_from_bytes(blob_bytes: bytes, sep=None, **kwargs) -> pd.DataFrame:
    """
    Lê CSV a partir de bytes testando encodings comuns e (opcionalmente) detectando separador.
    - Tenta encodings: utf-8, utf-8-sig, latin1, cp1252
    - Usa engine='python' para tolerar linhas ruins
    - Fallback final com on_bad_lines='skip'
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    if sep is None:
        sep = _sniff_sep_from_bytes(blob_bytes, default=",")

    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                BytesIO(blob_bytes),
                sep=sep,
                encoding=enc,
                engine="python",
                on_bad_lines=kwargs.pop("on_bad_lines", "warn"),
                **kwargs
            )
            print(f"🔎 Leitura OK com encoding='{enc}' sep='{sep}'")
            return df
        except Exception as e:
            last_err = e
            continue

    print("⚠️ Fallback: encoding='latin1', on_bad_lines='skip'")
    return pd.read_csv(
        BytesIO(blob_bytes),
        sep=sep,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
        **kwargs
    )

# ---------- Função principal robusta ----------

def download_blob(container_url, blob_name, sas_token, output_path, sep=None):
    print(f"📥 Baixando '{blob_name}' do container...")

    if "?" in container_url:
        container_url = container_url.split("?")[0]
    if sas_token.startswith("?"):
        sas_token = sas_token[1:]

    blob_url = f"{container_url}/{blob_name}?{sas_token}"
    blob_client = BlobClient.from_blob_url(blob_url)

    try:
        blob_data = blob_client.download_blob().readall()

        df = _read_csv_robusto_from_bytes(
            blob_data,
            sep=sep,      # se None → autodetecta
            dtype=str     # mantém tudo como string na ingestão
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Salvo como: {output_path}")
    except Exception as e:
        print(f"❌ Erro ao baixar {blob_name}: {e}")
        raise

def main():
    # aqui uso sep=None para autodetectar; se quiser fixar, use sep="|" ou ";"
    download_blob(CONTAINER_URL, ATIVOS_BLOB_NAME, SAS_TOKEN, ATIVOS_PATH, sep=None)

if __name__ == "__main__":
    main()
