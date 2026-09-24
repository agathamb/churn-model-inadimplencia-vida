import os
from azure.storage.blob import BlobClient
from config import SAS_TOKEN, UPLOAD_MAP, CONTAINER_URL


def upload_files_to_blob(arquivos_dict, container_url, sas_token):
    print("🚀 Iniciando upload para o Blob Storage via SAS Token...")

    if not sas_token or not container_url:
        raise ValueError("❌ Parâmetros 'container_url' e 'sas_token' devem ser fornecidos.")

    # Sanitiza para evitar erro de "??"
    if "?" in container_url:
        container_url = container_url.split("?")[0]
    if sas_token.startswith("?"):
        sas_token = sas_token[1:]

    for local_path, blob_name in arquivos_dict.items():
        if not os.path.exists(local_path):
            print(f"⚠️ Arquivo não encontrado: {local_path} — pulando upload.")
            continue

        blob_url = f"{container_url}/{blob_name}?{sas_token}"
        print(f"⬆️ Enviando {local_path} para {blob_url}...")

        try:
            blob_client = BlobClient.from_blob_url(blob_url)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data=data, overwrite=True)
            print("✅ Upload concluído para:", blob_name)
        except Exception as e:
            print(f"❌ Erro no upload de {blob_name}: {e}")

    print("🏁 Upload finalizado.")


if __name__ == "__main__":
    upload_files_to_blob(UPLOAD_MAP, CONTAINER_URL, SAS_TOKEN)