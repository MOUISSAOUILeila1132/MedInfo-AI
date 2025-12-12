# Fichier : download_model.py
# Ce script télécharge le modèle depuis Hugging Face et le sauvegarde dans un dossier local.
# Exécutez-le une seule fois.

from huggingface_hub import snapshot_download
import os

# Le nom du modèle sur le Hub Hugging Face
model_id = "microsoft/Phi-3-mini-4k-instruct"

# Le dossier local où vous voulez sauvegarder le modèle
# Nous allons le mettre dans un sous-dossier "models" pour garder le projet propre
local_model_path = os.path.join("models", model_id.split('/')[-1])

print(f"Téléchargement du modèle '{model_id}' vers '{local_model_path}'...")

# Crée le dossier s'il n'existe pas
os.makedirs(local_model_path, exist_ok=True)

# Télécharge tous les fichiers du modèle
# ignore_patterns évite de télécharger des fichiers lourds non essentiels comme les versions .bin
# si des versions .safetensors existent.
snapshot_download(
    repo_id=model_id,
    local_dir=local_model_path,
    ignore_patterns=["*.bin", "*.bin.index.json"] # On préfère les safetensors, plus sûrs
)

print("\n----------------------------------------------------------")
print(f"Modèle téléchargé avec succès dans : {local_model_path}")
print("Vous pouvez maintenant exécuter le script principal 'agent_medicaments_local.py'.")
print("----------------------------------------------------------")