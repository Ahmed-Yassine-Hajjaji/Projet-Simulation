# 🏛️ Simulation Discrète d'un Système de Retraite

Projet de simulation discrète (2025/2026) : étude de la viabilité
d'une caisse de retraite et impact d'une réforme paramétrique, par approche
Monte-Carlo (générateur `alea` de Wichmann–Hill, 40 réplications, 2026–2035).

## 📁 Contenu

| Fichier | Rôle |
|---|---|
| `simulation_retraite.py` | Moteur de simulation + tableaux + graphiques (exécution en console) |
| `app.py` | **Tableau de bord interactif** (Streamlit) |
| `requirements.txt` | Dépendances Python |
| `lancer_dashboard.sh` | Script de lancement automatique (installe tout au 1er run) |

## 🚀 Lancer le tableau de bord interactif

**Méthode simple :**
```bash
./lancer_dashboard.sh
```
Puis ouvrir http://localhost:8501 dans le navigateur.

**Méthode manuelle :**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Le dashboard permet de :
- modifier les **germes**, le **nombre de réplications** et la **réserve initiale** ;
- relancer la simulation **en direct** ;
- explorer **5 onglets** (vue d'ensemble, évolution détaillée, comparaison S1/S2,
  seniors >63 ans, données & export CSV) avec graphiques **interactifs** (zoom, survol).

## 🖥️ Exécution en console (tableaux + figures PNG)

```bash
source .venv/bin/activate
python simulation_retraite.py
```
Les graphiques sont générés dans le dossier `outputs/`.

## 🔬 Deux scénarios comparés

- **Scénario 1 — Actuel** : départ à 63 ans, cotisation employé seule.
- **Scénario 2 — Réforme** : report volontaire 63→70 ans, double cotisation
  employé + employeur, recrutements accrus, augmentations salariales plus fréquentes.

> Note : le taux de cotisation S2 pour les salaires > 10 000 dh suit la formule
> progressive `taux = min(10 + (i−1)×2, 30) %` avec `i = ⌊(sal − 10 000)/10 000⌋ + 1`,
> conformément au modèle mathématique du rapport.
