# runLocal.py
import os
import sys

# 💡 Définir les variables d'environnement AVANT les imports
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""

# Simule l'appel en CLI avec --config_path
sys.argv = ["runLocal", "--config_path=lerobot/config.yaml"]

# Ensuite, on importe
from lerobot.scripts.control_robot import control_robot

# Et on appelle (ça lit sys.argv automatiquement via draccus)
control_robot()
