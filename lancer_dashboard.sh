#!/usr/bin/env bash
# Lance le tableau de bord interactif de la simulation retraite.
# Usage : ./lancer_dashboard.sh
set -e
cd "$(dirname "$0")"

# Crée l'environnement virtuel au premier lancement
if [ ! -d ".venv" ]; then
  echo "→ Création de l'environnement virtuel et installation des dépendances…"
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

echo "→ Lancement du dashboard sur http://localhost:8501"
exec ./.venv/bin/streamlit run app.py
