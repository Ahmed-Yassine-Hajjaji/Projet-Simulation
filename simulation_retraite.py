"""
=============================================================================
SIMULATION DISCRÈTE - SYSTÈME DE RETRAITE
=============================================================================
Scénario 1 : Situation actuelle (retraite à 63 ans)
Scénario 2 : Réforme paramétrique (retraite flexible 63–70 ans)
Période : 2026–2035 | 40 réplications Monte-Carlo
=============================================================================
"""

import math
import os
import random
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATEUR PSEUDO-ALÉATOIRE  « alea »  (congruentiel combiné)
# ─────────────────────────────────────────────────────────────────────────────

class GenerateurAlea:
    """Générateur congruentiel combiné (méthode Wichmann-Hill adaptée)."""

    def __init__(self, ix: int, iy: int, iz: int):
        self.ix = ix
        self.iy = iy
        self.iz = iz

    def suivant(self) -> float:
        self.ix = 171 * (self.ix % 177) - 2  * (self.ix // 177)
        self.iy = 172 * (self.iy % 176) - 35 * (self.iy // 176)
        self.iz = 170 * (self.iz % 178) - 63 * (self.iz // 178)
        if self.ix < 0: self.ix += 30269
        if self.iy < 0: self.iy += 30307
        if self.iz < 0: self.iz += 30323
        inter = self.ix / 30269.0 + self.iy / 30307.0 + self.iz / 30323.0
        return inter - math.floor(inter)

    def uniforme(self, a: float, b: float) -> float:
        """Tire un nombre uniforme dans [a, b]."""
        return a + (b - a) * self.suivant()

    def entier_uniforme(self, a: int, b: int) -> int:
        """Tire un entier uniforme dans [a, b]."""
        return int(self.uniforme(a, b + 1 - 1e-9))

    def discret(self, valeurs: list, cumuls: list) -> object:
        """Sélection par transformation inverse sur distribution discrète/tabulée."""
        u = self.suivant()
        for val, cum in zip(valeurs, cumuls):
            if u <= cum:
                return val
        return valeurs[-1]

    def continu_tronque(self, tranches: list, freqs: list) -> float:
        """
        Sélection d'une tranche puis valeur uniforme dans la tranche.
        tranches : liste de (borne_inf, borne_sup)
        freqs    : liste de fréquences (somme = 1)
        """
        cumul = 0.0
        cumuls = []
        for f in freqs:
            cumul += f
            cumuls.append(cumul)
        u = self.suivant()
        for (a, b), cum in zip(tranches, cumuls):
            if u <= cum:
                return self.uniforme(a, b)
        a, b = tranches[-1]
        return self.uniforme(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTIONS  (partagées entre les deux scénarios)
# ─────────────────────────────────────────────────────────────────────────────

# Salaire actuel des 10 000 employés initiaux
TRANCHES_SAL_ACTUEL = [
    (3000, 5000), (5000, 7500), (7500, 10000), (10000, 15000),
    (15000, 20000), (20000, 30000), (30000, 40000), (40000, 80000), (80000, 120000)
]
FREQS_SAL_ACTUEL = [0.16, 0.20, 0.20, 0.20, 0.10, 0.05, 0.05, 0.03, 0.01]

# Âge actuel des employés
TRANCHES_AGE_ACTUEL = [(21, 30), (31, 40), (41, 52), (53, 63)]
FREQS_AGE_ACTUEL    = [0.20, 0.30, 0.30, 0.20]

# Âge à l'embauche (identique pour les deux scénarios)
TRANCHES_AGE_EMBAUCHE = [(21, 24), (25, 28), (29, 32), (33, 36), (37, 40), (41, 45)]
FREQS_AGE_EMBAUCHE    = [0.05, 0.30, 0.30, 0.15, 0.15, 0.05]

# Salaire à l'embauche (identique pour les deux scénarios)
TRANCHES_SAL_EMBAUCHE = [
    (3000, 4000), (4000, 6000), (6000, 8000), (8000, 12000),
    (12000, 16000), (16000, 24000), (24000, 32000), (32000, 50000)
]
FREQS_SAL_EMBAUCHE = [0.18, 0.20, 0.20, 0.20, 0.10, 0.05, 0.05, 0.02]

# Taux de cotisation scénario 1 (employé seul)
def taux_cotisation_s1(salaire: float) -> float:
    if salaire < 5000:   return 0.05
    if salaire < 7000:   return 0.06
    if salaire <= 10000: return 0.08
    return 0.10

# Taux de cotisation scénario 2 (employé + employeur)
def taux_cotisation_s2(salaire: float) -> Tuple[float, float]:
    """Retourne (taux_employe, taux_employeur)."""
    if salaire < 5000:
        return (0.06, 0.06)
    if salaire < 7000:
        return (0.07, 0.07)
    if salaire <= 10000:
        return (0.08, 0.08)
    # > 10 000 : taux = min(10 + (i-1)*2, 30) %, avec i = floor((sal - 10000)/10000) + 1
    # (formule progressive par tranches de 10 000 dh — cf. rapport, modèle mathématique)
    i = int((salaire - 10000) // 10000) + 1
    taux = min(10 + (i - 1) * 2, 30) / 100.0
    return (taux, taux)


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Employe:
    id: int
    age: float
    salaire: float
    annee_embauche: int          # année réelle d'embauche (pour NAT)
    genre: str                   # 'H' ou 'F'
    en_activite: bool = True

    def nat_a(self, annee: int) -> float:
        """Nombre d'Années Travaillées (NAT) à l'année donnée."""
        return max(0, annee - self.annee_embauche)


@dataclass
class Retraite:
    id: int
    pension_mensuelle: float     # pension fixe calculée au moment du départ

    def pension_annuelle(self) -> float:
        return self.pension_mensuelle * 12


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE PRINCIPALE : CAISSE DE RETRAITE
# ─────────────────────────────────────────────────────────────────────────────

class CaisseRetraite:
    """Gère l'ensemble des employés et retraités de la caisse."""

    def __init__(self, gen: GenerateurAlea, scenario: int, reserve_initiale: float = 200e6,
                 age_depart_max: int = 70, recrut_min: Optional[int] = None,
                 recrut_max: Optional[int] = None, taux_augmentation: Optional[float] = None,
                 facteur_employeur: float = 1.0):
        self.gen = gen
        self.scenario = scenario
        self.reserve = reserve_initiale
        self.employes: List[Employe] = []
        self.retraites: List[Retraite] = []
        self._id_counter = 0
        # ── Paramètres ajustables (réforme sur mesure). None ⇒ valeurs par défaut. ──
        self.age_depart_max = age_depart_max          # âge de départ forcé (S2)
        self.recrut_min = recrut_min                  # fourchette de recrutement
        self.recrut_max = recrut_max
        self.taux_augmentation = taux_augmentation    # % d'augmentation salariale
        self.facteur_employeur = facteur_employeur    # multiplicateur part employeur (S2)

    # ── Initialisation ───────────────────────────────────────────────────────

    def initialiser(self):
        """Génère les 10 000 employés et les 3 000 retraités initiaux."""
        # 10 000 employés
        for _ in range(10000):
            self._id_counter += 1
            age = self.gen.continu_tronque(TRANCHES_AGE_ACTUEL, FREQS_AGE_ACTUEL)
            sal = self.gen.continu_tronque(TRANCHES_SAL_ACTUEL, FREQS_SAL_ACTUEL)
            genre = 'H' if self.gen.suivant() < 0.55 else 'F'
            # Année d'embauche approximative (age - age_embauche)
            age_emb = self.gen.continu_tronque(TRANCHES_AGE_EMBAUCHE, FREQS_AGE_EMBAUCHE)
            annees_travaillees = max(1, age - age_emb)
            annee_embauche = int(2026 - annees_travaillees)
            emp = Employe(
                id=self._id_counter,
                age=age,
                salaire=sal,
                annee_embauche=annee_embauche,
                genre=genre
            )
            self.employes.append(emp)

        # 3 000 retraités déjà pensionnaires
        for _ in range(3000):
            self._id_counter += 1
            # On génère un salaire de retraite plausible
            sal_ret = self.gen.continu_tronque(TRANCHES_SAL_ACTUEL, FREQS_SAL_ACTUEL)
            nat = self.gen.uniforme(15, 40)   # entre 15 et 40 ans travaillés
            pension_mensuelle = ((nat * 2) / 100.0) * sal_ret
            self.retraites.append(Retraite(id=self._id_counter,
                                           pension_mensuelle=pension_mensuelle))

    # ── Taux de cotisation ───────────────────────────────────────────────────

    def _cotisation_annuelle(self, emp: Employe) -> float:
        """Cotisation annuelle totale (employé + éventuellement employeur)."""
        sal = emp.salaire
        if self.scenario == 1:
            return sal * taux_cotisation_s1(sal) * 12
        else:
            te, tpa = taux_cotisation_s2(sal)
            return sal * (te + tpa * self.facteur_employeur) * 12

    # ── Pension ──────────────────────────────────────────────────────────────

    def _calculer_pension(self, emp: Employe, annee_depart: int) -> float:
        """PR = (NAT * 2 / 100) * DSAR  (mensuelle)."""
        nat = emp.nat_a(annee_depart)
        return ((nat * 2) / 100.0) * emp.salaire

    # ── Avancement ───────────────────────────────────────────────────────────

    def appliquer_avancement(self, annee: int):
        """Augmente les salaires selon le scénario."""
        if self.scenario == 1:
            # +5% en janvier 2026, 2030, 2034
            taux = self.taux_augmentation if self.taux_augmentation is not None else 0.05
            if annee in (2026, 2030, 2034):
                for emp in self.employes:
                    emp.salaire *= (1 + taux)
        else:
            # +10% en janvier 2028, 2030, 2032, 2034
            taux = self.taux_augmentation if self.taux_augmentation is not None else 0.10
            if annee in (2028, 2030, 2032, 2034):
                for emp in self.employes:
                    emp.salaire *= (1 + taux)

    # ── Décision de prolonger (scénario 2) ──────────────────────────────────

    def _veut_prolonger(self, emp: Employe, annee: int) -> bool:
        """Retourne True si l'employé souhaite prolonger d'un an de plus."""
        age_courant = emp.age  # âge au moment de la décision
        ans_au_dela = max(0, int(age_courant) - 63)  # 0 à 63 ans, 1 à 64, etc.

        sal = emp.salaire
        genre = emp.genre

        # Probabilités de base à 63 ans
        if genre == 'H':
            if sal >= 30000:   p_base = 0.70; decrement = 0.05
            elif sal >= 10000: p_base = 0.50; decrement = 0.04
            elif sal >= 5000:  p_base = 0.30; decrement = 0.02
            else:              p_base = 0.10; decrement = 0.01
        else:
            if sal >= 30000:   p_base = 0.50; decrement = 0.04
            elif sal >= 10000: p_base = 0.30; decrement = 0.02
            elif sal >= 5000:  p_base = 0.15; decrement = 0.01
            else:              p_base = 0.05; decrement = 0.01

        prob = max(0.0, p_base - ans_au_dela * decrement)
        return self.gen.suivant() < prob

    # ── Simulation d'une année ───────────────────────────────────────────────

    def simuler_annee(self, annee: int) -> dict:
        """
        Effectue toutes les opérations d'une année et retourne les indicateurs.
        """
        # 1. Avancement des salaires (début janvier)
        self.appliquer_avancement(annee)

        # 2. Vieillissement des employés
        for emp in self.employes:
            emp.age += 1

        # 3. Identifier les nouveaux retraités
        nouveaux_retraites = []
        rester_actif = []

        for emp in self.employes:
            doit_partir = False

            if self.scenario == 1:
                doit_partir = emp.age >= 63
            else:
                if emp.age >= self.age_depart_max:
                    doit_partir = True
                elif emp.age >= 63:
                    # Décide s'il prolonge ou non
                    doit_partir = not self._veut_prolonger(emp, annee)
                else:
                    doit_partir = False

            if doit_partir:
                pension = self._calculer_pension(emp, annee)
                nouveaux_retraites.append(Retraite(id=emp.id,
                                                   pension_mensuelle=pension))
            else:
                rester_actif.append(emp)

        self.retraites.extend(nouveaux_retraites)
        self.employes = rester_actif

        # 4. Calcul des cotisations annuelles
        total_cotisations = sum(self._cotisation_annuelle(e) for e in self.employes)

        # 5. Calcul des pensions annuelles
        total_pensions = sum(r.pension_annuelle() for r in self.retraites)

        # 6. Mise à jour de la réserve
        self.reserve += total_cotisations - total_pensions

        # 7. Nouveaux recrutés (comptabilisés en janvier de l'année suivante)
        if self.scenario == 1:
            lo = self.recrut_min if self.recrut_min is not None else 250
            hi = self.recrut_max if self.recrut_max is not None else 400
        else:
            lo = self.recrut_min if self.recrut_min is not None else 300
            hi = self.recrut_max if self.recrut_max is not None else 600
        nb_recr = self.gen.entier_uniforme(lo, hi)

        nouveaux_employes = []
        for _ in range(nb_recr):
            self._id_counter += 1
            age_emb = self.gen.continu_tronque(TRANCHES_AGE_EMBAUCHE, FREQS_AGE_EMBAUCHE)
            sal_emb = self.gen.continu_tronque(TRANCHES_SAL_EMBAUCHE, FREQS_SAL_EMBAUCHE)
            genre = 'H' if self.gen.suivant() < 0.55 else 'F'
            nouveaux_employes.append(Employe(
                id=self._id_counter,
                age=age_emb,
                salaire=sal_emb,
                annee_embauche=annee + 1,  # recrutés pour l'année suivante
                genre=genre
            ))
        # Ils rejoignent en janvier de l'année suivante
        self.employes.extend(nouveaux_employes)

        # 8. Calcul des indicateurs spécifiques scénario 2
        plus63       = sum(1 for e in self.employes if e.age > 63)
        plus63_h     = sum(1 for e in self.employes if e.age > 63 and e.genre == 'H')
        plus63_f     = sum(1 for e in self.employes if e.age > 63 and e.genre == 'F')

        indicateurs = {
            'annee'         : annee,
            'TotEmp'        : len(self.employes),
            'TotRet'        : len(self.retraites),
            'TotCotis'      : total_cotisations,
            'TotPens'       : total_pensions,
            'Reserve'       : self.reserve,
            'NouvRet'       : len(nouveaux_retraites),
            'NouvRec'       : nb_recr,
            'Plus63'        : plus63,
            'Plus63H'       : plus63_h,
            'Plus63F'       : plus63_f,
        }
        return indicateurs


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE : SIMULATION (gère les 40 réplications)
# ─────────────────────────────────────────────────────────────────────────────

class Simulation:
    """Lance n réplications d'un scénario donné."""

    ANNEES = list(range(2026, 2036))

    def __init__(self, scenario: int, ix0: int, iy0: int, iz0: int,
                 n_replications: int = 40, reserve_initiale: float = 200e6,
                 increment_germe: int = 5, verbose: bool = True,
                 params_caisse: Optional[dict] = None):
        self.scenario = scenario
        self.ix0 = ix0
        self.iy0 = iy0
        self.iz0 = iz0
        self.n   = n_replications
        self.reserve_initiale = reserve_initiale
        self.increment_germe = increment_germe
        self.verbose = verbose
        # Paramètres ajustables transmis à chaque CaisseRetraite (réforme sur mesure)
        self.params_caisse = params_caisse or {}
        # Résultats : liste de listes de dicts (une liste par réplication)
        self.resultats: List[List[dict]] = []

    def lancer(self, progress_callback=None):
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  SCÉNARIO {self.scenario} — Lancement de {self.n} réplications")
            print(f"{'='*60}")
        ix, iy, iz = self.ix0, self.iy0, self.iz0

        for rep in range(self.n):
            gen = GenerateurAlea(ix, iy, iz)
            caisse = CaisseRetraite(gen, self.scenario, self.reserve_initiale,
                                    **self.params_caisse)
            caisse.initialiser()

            annee_resultats = []
            for annee in self.ANNEES:
                res = caisse.simuler_annee(annee)
                annee_resultats.append(res)

            self.resultats.append(annee_resultats)
            if progress_callback:
                progress_callback((rep + 1) / self.n)

            # Incrément des germes pour la prochaine réplication
            ix = ix + self.increment_germe
            iy = iy + self.increment_germe
            iz = iz + self.increment_germe
            # S'assurer que les germes restent dans [1, 30000]
            if ix > 30000: ix = ix % 30000 + 1
            if iy > 30000: iy = iy % 30000 + 1
            if iz > 30000: iz = iz % 30000 + 1

            if self.verbose and (rep + 1) % 10 == 0:
                print(f"  Réplication {rep+1:3d}/{self.n} terminée.")

    # ── Extraction de données ────────────────────────────────────────────────

    def valeurs_indicateur(self, indicateur: str, annee: int) -> List[float]:
        """Retourne les 40 valeurs d'un indicateur pour une année donnée."""
        idx = annee - 2026
        return [self.resultats[r][idx][indicateur] for r in range(self.n)]

    def moyenne(self, indicateur: str, annee: int) -> float:
        v = self.valeurs_indicateur(indicateur, annee)
        return statistics.mean(v)

    def ecart_type(self, indicateur: str, annee: int) -> float:
        v = self.valeurs_indicateur(indicateur, annee)
        return statistics.stdev(v)

    def intervalle_confiance_95(self, indicateur: str, annee: int) -> Tuple[float, float]:
        v    = self.valeurs_indicateur(indicateur, annee)
        moy  = statistics.mean(v)
        std  = statistics.stdev(v)
        marge = 1.96 * std / math.sqrt(len(v))
        return (moy - marge, moy + marge)

    def reserves_par_annee(self) -> List[List[float]]:
        """Retourne une liste[40][10] des réserves."""
        return [
            [self.resultats[r][idx]['Reserve'] for idx in range(10)]
            for r in range(self.n)
        ]

    def moyennes_par_annee(self, indicateur: str) -> List[float]:
        return [self.moyenne(indicateur, a) for a in self.ANNEES]


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE DES TABLEAUX
# ─────────────────────────────────────────────────────────────────────────────

def fmt_mil(v: float) -> str:
    """Formate un nombre en millions de dirhams."""
    if abs(v) >= 1e6:
        return f"{v/1e6:+.2f}M"
    return f"{v:+.0f}"

def fmt_mdh(v: float) -> str:
    return f"{v/1e6:.2f} Mdh"

def fmt_int(v: float) -> str:
    return f"{int(round(v)):,}"

def fmt_pct(v: float) -> str:
    return f"{v/1e6:.1f} Mdh"


def afficher_tableau_40_simulations(sim: Simulation, annee: int, scenario_num: int):
    """Affiche le tableau des 40 simulations pour une année donnée."""
    if scenario_num == 1:
        indicateurs = ['TotEmp', 'TotRet', 'TotCotis', 'TotPens', 'Reserve', 'NouvRet', 'NouvRec']
        labels      = ['TotEmp', 'TotRet', 'TotCotis(Mdh)', 'TotPens(Mdh)', 'Reserve(Mdh)', 'NouvRet', 'NouvRec']
    else:
        indicateurs = ['TotEmp', 'Plus63', 'Plus63H', 'Plus63F', 'TotRet',
                       'TotCotis', 'TotPens', 'Reserve', 'NouvRet', 'NouvRec']
        labels      = ['TotEmp', 'Plus63', '+63H', '+63F', 'TotRet',
                       'TotCotis(Mdh)', 'TotPens(Mdh)', 'Reserve(Mdh)', 'NouvRet', 'NouvRec']

    col_w = 12
    header = f"{'Sim':>4} " + " ".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n{'─'*len(header)}")
    print(f"  SCÉNARIO {scenario_num} — Année {annee} — 40 Simulations")
    print(f"{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    idx = annee - 2026
    for r in range(sim.n):
        row = f"{r+1:>4} "
        for ind in indicateurs:
            v = sim.resultats[r][idx][ind]
            if 'Cotis' in ind or 'Pens' in ind or 'Reserve' in ind:
                row += f"{v/1e6:>{col_w}.2f}"
            else:
                row += f"{int(round(v)):>{col_w},}"
        print(row)

    # Ligne moyenne
    print(f"{'─'*len(header)}")
    moy_row = f"{'MOY':>4} "
    for ind in indicateurs:
        v = sim.moyenne(ind, annee)
        if 'Cotis' in ind or 'Pens' in ind or 'Reserve' in ind:
            moy_row += f"{v/1e6:>{col_w}.2f}"
        else:
            moy_row += f"{int(round(v)):>{col_w},}"
    print(moy_row)
    print(f"{'─'*len(header)}\n")


def afficher_tableau_reserve(sim: Simulation, scenario_num: int):
    """Affiche le tableau de la réserve sur les 10 années pour les 40 simulations."""
    annees = list(range(2026, 2036))
    header = f"{'Sim':>4} " + " ".join(f"{a:>10}" for a in annees)
    print(f"\n{'─'*len(header)}")
    print(f"  SCÉNARIO {scenario_num} — RÉSERVE (Mdh) — 40 simulations × 10 années")
    print(f"{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    for r in range(sim.n):
        row = f"{r+1:>4} "
        for idx in range(10):
            v = sim.resultats[r][idx]['Reserve'] / 1e6
            row += f"{v:>10.2f}"
        print(row)

    print(f"{'─'*len(header)}")
    moy_row = f"{'MOY':>4} "
    for idx in range(10):
        v = sim.moyenne('Reserve', 2026 + idx) / 1e6
        moy_row += f"{v:>10.2f}"
    print(moy_row)
    print(f"{'─'*len(header)}\n")


def afficher_tableau_plus63(sim: Simulation):
    """Tableau spécifique scénario 2 : employés > 63 ans."""
    annees = list(range(2026, 2036))
    header = f"{'Sim':>4} " + " ".join(f"{'P63-'+str(a):>8}" for a in annees)
    print(f"\n{'─'*len(header)}")
    print("  SCÉNARIO 2 — Employés > 63 ans (Total | H | F) — 40 simulations")
    print(f"{'─'*len(header)}")

    for indicateur, label in [('Plus63', 'TOTAL'), ('Plus63H', 'HOMMES'), ('Plus63F', 'FEMMES')]:
        print(f"\n  {label}")
        print(header)
        print(f"{'─'*len(header)}")
        cumuls = [0] * 10
        for r in range(sim.n):
            row = f"{r+1:>4} "
            for idx in range(10):
                v = sim.resultats[r][idx][indicateur]
                cumuls[idx] += v
                row += f"{int(round(v)):>8,}"
            print(row)
        print(f"{'─'*len(header)}")
        # Ligne cumul
        cum_row = f"{'CUM':>4} "
        for idx in range(10):
            cum_row += f"{int(cumuls[idx]):>8,}"
        print(cum_row)
        # Ligne moyenne
        moy_row = f"{'MOY':>4} "
        for idx in range(10):
            moy_row += f"{cumuls[idx]/sim.n:>8.1f}"
        print(moy_row)
        print(f"{'─'*len(header)}")


def afficher_intervalles_confiance(sim1: Simulation, sim2: Simulation):
    """Affiche les IC à 95% de la réserve pour 2026, 2030, 2035."""
    annees_ic = [2026, 2030, 2035]
    print("\n" + "═"*70)
    print("  INTERVALLES DE CONFIANCE À 95% — RÉSERVE (Mdh)")
    print("═"*70)
    print(f"{'Année':>8} {'Scénario':>12} {'Moyenne':>12} {'IC Inf':>12} {'IC Sup':>12} {'Demi-IC':>12}")
    print("─"*70)
    for a in annees_ic:
        for sc, sim in [(1, sim1), (2, sim2)]:
            moy = sim.moyenne('Reserve', a) / 1e6
            ic_inf, ic_sup = sim.intervalle_confiance_95('Reserve', a)
            ic_inf /= 1e6; ic_sup /= 1e6
            demi = (ic_sup - ic_inf) / 2
            print(f"{a:>8} {'S'+str(sc):>12} {moy:>12.2f} {ic_inf:>12.2f} {ic_sup:>12.2f} {demi:>12.2f}")
    print("═"*70)


def afficher_tableau_comparatif(sim1: Simulation, sim2: Simulation):
    """Tableau récapitulatif réserve moyenne : S1 vs S2 sur 10 ans."""
    annees = list(range(2026, 2036))
    print("\n" + "═"*60)
    print("  TABLEAU COMPARATIF — RÉSERVE MOYENNE (Mdh) S1 vs S2")
    print("═"*60)
    print(f"{'Année':>6} {'Scénario 1':>14} {'Scénario 2':>14} {'Différence':>14}")
    print("─"*60)
    for a in annees:
        m1 = sim1.moyenne('Reserve', a) / 1e6
        m2 = sim2.moyenne('Reserve', a) / 1e6
        diff = m2 - m1
        print(f"{a:>6} {m1:>14.2f} {m2:>14.2f} {diff:>14.2f}")
    print("═"*60)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

COULEURS = {'s1': '#2563EB', 's2': '#DC2626', 'neutre': '#059669', 'gris': '#6B7280'}

# Dossier de sortie des graphiques (créé automatiquement, à côté du script)
DOSSIER_SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

def graphiques_scenario(sim: Simulation, sc: int, dossier: str = DOSSIER_SORTIE):
    """Génère tous les graphiques pour un scénario donné."""
    os.makedirs(dossier, exist_ok=True)
    annees = list(range(2026, 2036))
    couleur = COULEURS['s1'] if sc == 1 else COULEURS['s2']
    nom = f"Scénario {sc}"

    # ── Fig 1 : Évolution des indicateurs clés ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{nom} — Évolution des indicateurs (moyennes des 40 simulations)",
                 fontsize=14, fontweight='bold')

    indicateurs_graphe = [
        ('TotEmp',   'Nombre d\'employés',    fmt_int),
        ('TotRet',   'Nombre de retraités',   fmt_int),
        ('TotCotis', 'Total cotisations (Mdh)', lambda v: f"{v/1e6:.1f}"),
        ('TotPens',  'Total pensions (Mdh)',   lambda v: f"{v/1e6:.1f}"),
        ('Reserve',  'Réserve (Mdh)',          lambda v: f"{v/1e6:.1f}"),
        ('NouvRet',  'Nouveaux retraités',     fmt_int),
    ]

    for ax, (ind, titre, _) in zip(axes.flat, indicateurs_graphe):
        moyennes = sim.moyennes_par_annee(ind)
        std_vals = [sim.ecart_type(ind, a) for a in annees]
        y = np.array(moyennes)
        s = np.array(std_vals)
        if 'Cotis' in ind or 'Pens' in ind or 'Reserve' in ind:
            y = y / 1e6; s = s / 1e6
        ax.plot(annees, y, color=couleur, linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(annees, y - s, y + s, alpha=0.2, color=couleur)
        ax.set_title(titre, fontsize=11, fontweight='bold')
        ax.set_xlabel('Année')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2025.5, 2035.5)

    plt.tight_layout()
    chemin = f"{dossier}/graphique_s{sc}_indicateurs.png"
    plt.savefig(chemin, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Graphique sauvegardé : {chemin}")

    # ── Fig 2 : Réserve — 40 simulations + moyenne ───────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    reserves = sim.reserves_par_annee()
    for r in range(sim.n):
        vals = [v / 1e6 for v in reserves[r]]
        ax.plot(annees, vals, alpha=0.15, color=couleur, linewidth=0.8)
    moy_res = [sim.moyenne('Reserve', a) / 1e6 for a in annees]
    ax.plot(annees, moy_res, color=couleur, linewidth=3, label='Moyenne', zorder=5)

    # IC 95%
    for annee_ic in [2026, 2030, 2035]:
        ic = sim.intervalle_confiance_95('Reserve', annee_ic)
        ax.errorbar(annee_ic, (ic[0]+ic[1])/2/1e6,
                    yerr=((ic[1]-ic[0])/2/1e6),
                    fmt='D', color='black', capsize=8, linewidth=2, markersize=6,
                    label=f'IC 95% {annee_ic}' if annee_ic == 2026 else None)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_title(f"{nom} — Réserve de la caisse (40 simulations)", fontsize=13, fontweight='bold')
    ax.set_xlabel('Année')
    ax.set_ylabel('Réserve (Mdh)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    chemin = f"{dossier}/graphique_s{sc}_reserve.png"
    plt.savefig(chemin, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Graphique sauvegardé : {chemin}")

    # ── Fig 3 (scénario 2 uniquement) : Employés > 63 ans ────────────────
    if sc == 2:
        fig, ax = plt.subplots(figsize=(13, 6))
        for ind, label, col in [('Plus63', 'Total', '#7C3AED'),
                                  ('Plus63H', 'Hommes', '#2563EB'),
                                  ('Plus63F', 'Femmes', '#DC2626')]:
            moy = sim.moyennes_par_annee(ind)
            ax.plot(annees, moy, label=label, linewidth=2.5, marker='o', markersize=5, color=col)
        ax.set_title("Scénario 2 — Employés de plus de 63 ans (moyennes)", fontsize=13, fontweight='bold')
        ax.set_xlabel('Année')
        ax.set_ylabel('Nombre d\'employés')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        chemin = f"{dossier}/graphique_s2_plus63.png"
        plt.savefig(chemin, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Graphique sauvegardé : {chemin}")


def graphiques_comparatifs(sim1: Simulation, sim2: Simulation,
                            dossier: str = DOSSIER_SORTIE):
    """Graphiques croisés S1 vs S2."""
    os.makedirs(dossier, exist_ok=True)
    annees = list(range(2026, 2036))

    # ── Fig 1 : Réserve comparée ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    for sim, sc, col, lab in [(sim1, 1, COULEURS['s1'], 'Scénario 1 (actuel)'),
                               (sim2, 2, COULEURS['s2'], 'Scénario 2 (réforme)')]:
        moy = [sim.moyenne('Reserve', a) / 1e6 for a in annees]
        std = [sim.ecart_type('Reserve', a) / 1e6 for a in annees]
        y = np.array(moy); s = np.array(std)
        ax.plot(annees, y, color=col, linewidth=2.5, marker='o', markersize=6, label=lab)
        ax.fill_between(annees, y - s, y + s, alpha=0.15, color=col)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Réserve = 0')
    ax.set_title("Comparaison S1 vs S2 — Réserve de la caisse de retraite (Mdh)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Réserve (Mdh)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    chemin = f"{dossier}/graphique_compare_reserve.png"
    plt.savefig(chemin, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Graphique sauvegardé : {chemin}")

    # ── Fig 2 : Multi-indicateurs S1 vs S2 ───────────────────────────────
    indicateurs = [
        ('TotEmp',   'Total Employés'),
        ('TotRet',   'Total Retraités'),
        ('TotCotis', 'Cotisations (Mdh)'),
        ('TotPens',  'Pensions (Mdh)'),
        ('NouvRet',  'Nouveaux Retraités'),
        ('NouvRec',  'Nouveaux Recrutés'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Comparaison S1 vs S2 — Indicateurs clés (moyennes des 40 simulations)",
                 fontsize=14, fontweight='bold')

    for ax, (ind, titre) in zip(axes.flat, indicateurs):
        for sim, sc, col, lab in [(sim1, 1, COULEURS['s1'], 'S1 actuel'),
                                   (sim2, 2, COULEURS['s2'], 'S2 réforme')]:
            moy = sim.moyennes_par_annee(ind)
            y = np.array(moy)
            if 'Cotis' in ind or 'Pens' in ind:
                y = y / 1e6
            ax.plot(annees, y, color=col, linewidth=2.5, marker='o', markersize=4, label=lab)
        ax.set_title(titre, fontsize=11, fontweight='bold')
        ax.set_xlabel('Année')
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chemin = f"{dossier}/graphique_compare_indicateurs.png"
    plt.savefig(chemin, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Graphique sauvegardé : {chemin}")

    # ── Fig 3 : IC comparés pour 2026, 2030, 2035 ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    annees_ic = [2026, 2030, 2035]
    x = np.arange(len(annees_ic))
    width = 0.35

    for i, (sim, sc, col, lab) in enumerate([(sim1, 1, COULEURS['s1'], 'S1 actuel'),
                                              (sim2, 2, COULEURS['s2'], 'S2 réforme')]):
        moys  = [sim.moyenne('Reserve', a) / 1e6 for a in annees_ic]
        errs  = [(sim.moyenne('Reserve', a) - sim.intervalle_confiance_95('Reserve', a)[0]) / 1e6
                 for a in annees_ic]
        offset = x + (i - 0.5) * width
        bars = ax.bar(offset, moys, width, label=lab, color=col, alpha=0.8, edgecolor='black')
        ax.errorbar(offset, moys, yerr=errs, fmt='none', color='black', capsize=6, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in annees_ic], fontsize=12)
    ax.set_title("Réserve moyenne et IC 95% — S1 vs S2 (années clés)", fontsize=13, fontweight='bold')
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Réserve (Mdh)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    chemin = f"{dossier}/graphique_compare_IC.png"
    plt.savefig(chemin, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Graphique sauvegardé : {chemin}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE & CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────

def afficher_analyse(sim1: Simulation, sim2: Simulation):
    annees_cles = [2026, 2030, 2035]
    print("\n" + "═"*70)
    print("  ANALYSE DES RÉSULTATS — COMPARAISON S1 VS S2")
    print("═"*70)

    print("\n  1. ÉVOLUTION DE LA RÉSERVE (Mdh)")
    print("─"*70)
    for a in annees_cles:
        m1 = sim1.moyenne('Reserve', a) / 1e6
        m2 = sim2.moyenne('Reserve', a) / 1e6
        diff = m2 - m1
        pct  = (diff / abs(m1) * 100) if m1 != 0 else float('inf')
        print(f"  {a} : S1 = {m1:+.2f} Mdh | S2 = {m2:+.2f} Mdh | Δ = {diff:+.2f} Mdh ({pct:+.1f}%)")

    print("\n  2. EMPLOYÉS ACTIFS ET RETRAITÉS EN 2035")
    print("─"*70)
    for sc, sim in [(1, sim1), (2, sim2)]:
        emp = sim.moyenne('TotEmp', 2035)
        ret = sim.moyenne('TotRet', 2035)
        ratio = ret / emp if emp > 0 else float('inf')
        print(f"  S{sc} — Employés : {emp:,.0f} | Retraités : {ret:,.0f} | Ratio ret/emp : {ratio:.3f}")

    print("\n  3. COTISATIONS VS PENSIONS EN 2035 (Mdh)")
    print("─"*70)
    for sc, sim in [(1, sim1), (2, sim2)]:
        cot = sim.moyenne('TotCotis', 2035) / 1e6
        pen = sim.moyenne('TotPens', 2035) / 1e6
        solde = cot - pen
        print(f"  S{sc} — Cotisations : {cot:.2f} | Pensions : {pen:.2f} | Solde : {solde:+.2f}")

    print("\n  4. INTERVALLES DE CONFIANCE À 95% — RÉSERVE (Mdh)")
    print("─"*70)
    for a in annees_cles:
        for sc, sim in [(1, sim1), (2, sim2)]:
            ic = sim.intervalle_confiance_95('Reserve', a)
            print(f"  S{sc} {a} : [{ic[0]/1e6:.2f} ; {ic[1]/1e6:.2f}] Mdh")

    print("\n" + "═"*70)
    print("  CONCLUSION")
    print("═"*70)

    r1_2026 = sim1.moyenne('Reserve', 2026) / 1e6
    r2_2026 = sim2.moyenne('Reserve', 2026) / 1e6
    r1_2035 = sim1.moyenne('Reserve', 2035) / 1e6
    r2_2035 = sim2.moyenne('Reserve', 2035) / 1e6

    print(f"""
  La simulation Monte-Carlo sur 40 réplications (2026–2035) révèle :

  ► SCÉNARIO 1 (Actuel) :
    · Réserve initiale (2026) : {r1_2026:+.2f} Mdh
    · Réserve finale   (2035) : {r1_2035:+.2f} Mdh
    · Tendance : {'⬇ DÉGRADATION' if r1_2035 < r1_2026 else '⬆ AMÉLIORATION'}
    {"· RISQUE D'EFFONDREMENT détecté (réserve négative)" if r1_2035 < 0 else "· Réserve positive maintenue"}

  ► SCÉNARIO 2 (Réforme) :
    · Réserve initiale (2026) : {r2_2026:+.2f} Mdh
    · Réserve finale   (2035) : {r2_2035:+.2f} Mdh
    · Tendance : {'⬇ DÉGRADATION' if r2_2035 < r2_2026 else '⬆ AMÉLIORATION'}
    {"· RISQUE D'EFFONDREMENT détecté (réserve négative)" if r2_2035 < 0 else "· Réserve positive maintenue"}

  ► IMPACT DE LA RÉFORME :
    · Gain en réserve en 2035 : {r2_2035 - r1_2035:+.2f} Mdh
    · La réforme {'améliore significativement' if r2_2035 > r1_2035 else 'ne suffit pas à améliorer'} la soutenabilité de la caisse.
    · La double cotisation (employé + employeur) et le recrutement accru
      dans S2 génèrent davantage de ressources pour la caisse.
    · Le report volontaire de la retraite réduit la pression des pensions
      et augmente la durée de cotisation des employés seniors.

  ► RECOMMANDATION :
    La réforme paramétrique (Scénario 2) est {'PERTINENTE' if r2_2035 > r1_2035 else 'INSUFFISANTE'}.
    {'Elle doit être adoptée pour éviter un effondrement de la caisse.' if r2_2035 > r1_2035 else 'Des mesures supplémentaires sont nécessaires.'}
""")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*70)
    print("  SIMULATION DISCRÈTE — CAISSE DE RETRAITE")
    print("  Approche Monte-Carlo — 2025/2026")
    print("█"*70)

    # ── Germes initiaux ───────────────────────────────────────────────────
    # Germes de base (400, 400, 400)
    IX0, IY0, IZ0 = 400, 400, 400

    N_REP = 40   # nombre de réplications

    # ── Scénario 1 ────────────────────────────────────────────────────────
    sim1 = Simulation(scenario=1, ix0=IX0, iy0=IY0, iz0=IZ0, n_replications=N_REP)
    sim1.lancer()

    # ── Scénario 2 ────────────────────────────────────────────────────────
    sim2 = Simulation(scenario=2, ix0=IX0, iy0=IY0, iz0=IZ0, n_replications=N_REP)
    sim2.lancer()

    # ── Tableaux scénario 1 ───────────────────────────────────────────────
    print("\n" + "█"*70)
    print("  RÉSULTATS — SCÉNARIO 1")
    print("█"*70)
    for annee in [2026, 2030, 2035]:
        afficher_tableau_40_simulations(sim1, annee, 1)
    afficher_tableau_reserve(sim1, 1)

    # ── Tableaux scénario 2 ───────────────────────────────────────────────
    print("\n" + "█"*70)
    print("  RÉSULTATS — SCÉNARIO 2")
    print("█"*70)
    for annee in [2026, 2030, 2035]:
        afficher_tableau_40_simulations(sim2, annee, 2)
    afficher_tableau_reserve(sim2, 2)
    afficher_tableau_plus63(sim2)

    # ── Tableau comparatif ────────────────────────────────────────────────
    afficher_tableau_comparatif(sim1, sim2)
    afficher_intervalles_confiance(sim1, sim2)

    # ── Analyse ───────────────────────────────────────────────────────────
    afficher_analyse(sim1, sim2)

    # ── Graphiques ────────────────────────────────────────────────────────
    print("\n  Génération des graphiques...")
    graphiques_scenario(sim1, 1)
    graphiques_scenario(sim2, 2)
    graphiques_comparatifs(sim1, sim2)

    print("\n  ✅  Simulation terminée avec succès.")
    return sim1, sim2


if __name__ == "__main__":
    main()
