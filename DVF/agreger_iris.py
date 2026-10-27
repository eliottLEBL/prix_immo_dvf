"""Prix moyen au m² par IRIS et par année (DVF+ Île-de-France), dans un seul CSV.

Usage (depuis la racine du dépôt) :
    python DVF/agreger_iris.py                  # tous les départements, toutes les années
    python DVF/agreger_iris.py --dep 75 92 --annee-min 2020
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

# main.py est à la racine du dépôt
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import DOSSIER_DVF, charger_iris, charger_ventes

DOSSIER = Path(__file__).parent


def prix_par_iris_et_annee(ventes, iris):
    jointes = gpd.sjoin(ventes, iris[["code_iris", "geometry"]], predicate="within")
    groupes = jointes.groupby(["code_iris", "anneemut"])
    stats = groupes.agg(
        nb_ventes=("prix_m2", "size"),
        prix_m2_moyen=("prix_m2", "mean"),
        prix_m2_median=("prix_m2", "median"),
        prix_moyen=("valeurfonc", "mean"),
    )
    par_type = jointes.pivot_table(
        index=["code_iris", "anneemut"], columns="type", values="prix_m2", aggfunc=["mean", "size"]
    )
    par_type.columns = [
        f"{'prix_m2_moyen' if stat == 'mean' else 'nb_ventes'}_{'appart' if t == 'Appartement' else 'maison'}"
        for stat, t in par_type.columns
    ]
    stats = stats.join(par_type).round().reset_index().rename(columns={"anneemut": "annee"})
    infos = iris[["code_iris", "nom_iris", "code_insee", "nom_commune"]]
    return infos.merge(stats, on="code_iris").sort_values(["code_iris", "annee"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dep", nargs="*", help="Départements à inclure (ex : 75 92). Par défaut : tous.")
    parser.add_argument("--annee-min", type=int, default=1900)
    parser.add_argument("--annee-max", type=int, default=2100)
    parser.add_argument("--sortie", default=str(DOSSIER / "prix_m2_iris_annee.csv"))
    args = parser.parse_args()

    departements = args.dep or [f.stem.removeprefix("dvf_plus_d") for f in sorted(DOSSIER_DVF.glob("dvf_plus_d*.csv"))]

    ventes = charger_ventes(departements, args.annee_min, args.annee_max)
    iris = charger_iris(departements)
    resultat = prix_par_iris_et_annee(ventes, iris)

    resultat.to_csv(args.sortie, index=False, sep=";", encoding="utf-8-sig")
    nb_lignes = f"{len(resultat):_}".replace("_", " ")
    print(f"{nb_lignes} lignes (IRIS × année) enregistrées : {args.sortie}")


if __name__ == "__main__":
    main()
