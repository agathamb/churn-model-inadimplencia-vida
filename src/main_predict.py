import subprocess
from config import LOG_FILE_PRED, LOG_DIR
import os

# Lista dos scripts em ordem
scripts = [
    "01_ingestion_predict.py",
    "07_score.py",
    "08_post_process.py",
    "10_upload_files.py"
]

def main():
    # cria a pasta de logs se não existir
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOG_FILE_PRED, "w", encoding="utf-8") as logfile:
        for script in scripts:
            msg_inicio = f"\n🚀 Executando {script}...\n"
            print(msg_inicio)
            logfile.write(msg_inicio)

            result = subprocess.run(
                ["python", script],
                stdout=logfile,
                stderr=logfile,
                text=True
            )

            if result.returncode != 0:
                msg_erro = f"❌ Erro ao executar {script}. Encerrando pipeline.\n"
                print(msg_erro)
                logfile.write(msg_erro)
                break

            msg_ok = f"✅ {script} concluído.\n"
            print(msg_ok)
            logfile.write(msg_ok)

    print(f"\n📄 Log consolidado salvo em: {LOG_FILE_PRED}")

if __name__ == "__main__":
    main()
