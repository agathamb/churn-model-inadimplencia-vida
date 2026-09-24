import subprocess
import os
from config import LOG_FILE_TRAIN, LOG_DIR

# Lista dos scripts em ordem
scripts = [
    "01_ingestion_train.py",
    "02_preprocess.py",
    "03_correlation_filter.py",
    "04_tune.py",
    "05_train.py",
    "06_evaluate.py",
]

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOG_FILE_TRAIN, "w", encoding="utf-8") as logfile:
        for script in scripts:
            msg_inicio = f"\n🚀 Executando {script}...\n"
            print(msg_inicio, end="")
            logfile.write(msg_inicio)

            process = subprocess.Popen(
                ["python", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Lê linha a linha e joga no console e no log
            for line in process.stdout:
                print(line, end="")        # console
                logfile.write(line)        # arquivo

            process.wait()

            if process.returncode != 0:
                msg_erro = f"❌ Erro ao executar {script}. Encerrando pipeline.\n"
                print(msg_erro)
                logfile.write(msg_erro)
                break

            msg_ok = f"✅ {script} concluído.\n"
            print(msg_ok)
            logfile.write(msg_ok)

    print(f"\n📄 Log consolidado salvo em: {LOG_FILE_TRAIN}")

if __name__ == "__main__":
    main()
