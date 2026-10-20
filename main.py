"""Carte du prix médian au m² par IRIS (DVF+ Île-de-France).

Usage :
    python main.py                          # tous les départements, ventes depuis 2024
    python main.py --dep 75 92              # seulement Paris et Hauts-de-Seine
    python main.py --annee-min 2020 --annee-max 2022
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

DOSSIER = Path(__file__).parent
DOSSIER_DVF = DOSSIER.parent / "DVF+" / "1_DONNEES_LIVRAISON"
DOSSIER_IRIS = DOSSIER / "contours_iris_2026"

COLONNES = [
    "datemut", "anneemut", "libnatmut", "valeurfonc",
    "sbati", "libtypbien", "geompar_x", "geompar_y",
]

# Une seule maison ou un seul appartement par mutation : sinon le prix au m² n'a pas de sens
TYPES_BIENS = {"UN APPARTEMENT": "Appartement", "UNE MAISON": "Maison"}
NATURES = {"Vente", "Vente en l'état futur d'achèvement", "Adjudication"}

# Bornes pour écarter les valeurs aberrantes (erreurs de saisie, ventes symboliques…)
PRIX_M2_MIN = 500
PRIX_M2_MAX = 30_000

# En dessous de ce nombre de ventes, la médiane d'un IRIS n'est pas fiable
VENTES_MIN_PAR_IRIS = 5

# Échelle séquentielle à une seule teinte (bleu clair → bleu foncé), commune aux deux cartes
BLEUS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COULEURS = LinearSegmentedColormap.from_list("bleus", BLEUS)
GRIS_SANS_DONNEES = "#e4e3df"


def charger_ventes(departements, annee_min, annee_max):
    fichiers = [DOSSIER_DVF / f"dvf_plus_d{d}.csv" for d in departements]
    manquants = [f.name for f in fichiers if not f.exists()]
    if manquants:
        raise SystemExit(f"Fichiers introuvables : {', '.join(manquants)}")

    morceaux = []
    for f in fichiers:
        print(f"Lecture de {f.name}…")
        df = pd.read_csv(f, sep="|", usecols=COLONNES)
        df = df[
            df["libtypbien"].isin(TYPES_BIENS.keys())
            & df["libnatmut"].isin(NATURES)
            & df["anneemut"].between(annee_min, annee_max)
            & (df["sbati"] > 0)
            & (df["valeurfonc"] > 0)
        ]
        morceaux.append(df)

    df = pd.concat(morceaux, ignore_index=True)
    df = df.dropna(subset=["geompar_x", "geompar_y"])

    df["type"] = df["libtypbien"].map(TYPES_BIENS)
    df["prix_m2"] = df["valeurfonc"] / df["sbati"]
    df = df[df["prix_m2"].between(PRIX_M2_MIN, PRIX_M2_MAX)]

    # Les coordonnées DVF+ sont en Lambert-93, comme les contours IRIS
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["geompar_x"], df["geompar_y"]), crs="EPSG:2154"
    )


def charger_iris(departements):
    fichier = next(DOSSIER_IRIS.rglob("iris.gpkg"), None)
    if fichier is None:
        raise SystemExit(f"iris.gpkg introuvable dans {DOSSIER_IRIS}")

    codes = ", ".join(f"'{d}'" for d in departements)
    print(f"Lecture des contours IRIS ({fichier.name})…")
    iris = gpd.read_file(
        fichier,
        columns=["code_iris", "code_insee", "nom_iris", "nom_commune"],
        where=f"substr(code_insee, 1, 2) IN ({codes})",
    )
    iris["dep"] = iris["code_insee"].str[:2]
    return iris


def prix_par_iris(ventes, iris):
    jointes = gpd.sjoin(ventes, iris[["code_iris", "geometry"]], predicate="within")
    stats = jointes.groupby("code_iris")["prix_m2"].agg(prix_m2="median", nb_ventes="size")
    par_type = jointes.pivot_table(index="code_iris", columns="type", values="prix_m2", aggfunc="median")
    par_type = par_type.rename(columns={"Appartement": "prix_m2_appart", "Maison": "prix_m2_maison"})
    stats = stats.join(par_type)
    stats = stats[stats["nb_ventes"] >= VENTES_MIN_PAR_IRIS].round()
    # Jointure à gauche : les IRIS sans assez de ventes restent sur la carte, en gris
    return iris.merge(stats, left_on="code_iris", right_index=True, how="left")


def dessiner(ax, carte, departements_contours, vmin, vmax):
    carte.plot(
        ax=ax, column="prix_m2", cmap=COULEURS, vmin=vmin, vmax=vmax,
        edgecolor="white", linewidth=0.1,
        missing_kwds={"color": GRIS_SANS_DONNEES, "edgecolor": "white", "linewidth": 0.1},
    )
    departements_contours.boundary.plot(ax=ax, color="#3d3d3a", linewidth=0.6)
    ax.set_axis_off()


def creer_carte(carte, sortie, titre):
    vmin, vmax = carte["prix_m2"].quantile([0.02, 0.98])
    departements_contours = carte.dissolve(by="dep")

    fig, ax = plt.subplots(figsize=(12, 11))
    dessiner(ax, carte, departements_contours, vmin, vmax)

    # Encart zoomé sur Paris, illisible à l'échelle régionale
    if "75" in carte["dep"].values:
        paris = carte[carte["dep"] == "75"]
        encart = ax.inset_axes([0.0, 0.0, 0.36, 0.30])
        dessiner(encart, carte, departements_contours, vmin, vmax)
        xmin, ymin, xmax, ymax = paris.total_bounds
        marge = 500
        encart.set_xlim(xmin - marge, xmax + marge)
        encart.set_ylim(ymin - marge, ymax + marge)
        encart.set_axis_on()
        encart.set_xticks([])
        encart.set_yticks([])
        encart.set_xlabel("")
        encart.set_ylabel("")
        for bord in encart.spines.values():
            bord.set_edgecolor("#3d3d3a")
        encart.set_title("Paris", fontsize=10, color="#3d3d3a", loc="left")
        ax.indicate_inset_zoom(encart, edgecolor="#3d3d3a")

    barre = fig.colorbar(
        plt.cm.ScalarMappable(cmap=COULEURS, norm=plt.Normalize(vmin, vmax)),
        ax=ax, shrink=0.5, pad=0.01,
    )
    barre.set_label("Prix médian au m² (€)", color="#3d3d3a")
    barre.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    barre.outline.set_visible(False)

    ax.set_title(titre, fontsize=15, loc="left", color="#1f1f1d")
    ax.text(
        0.0, -0.02,
        f"Source : DVF+ (Cerema). Appartements et maisons vendus seuls. "
        f"En gris : IRIS avec moins de {VENTES_MIN_PAR_IRIS} ventes. "
        f"Couleurs bornées aux 2e–98e centiles.",
        fontsize=8, color="#6b6b66", transform=ax.transAxes, va="top",
    )
    fig.savefig(sortie, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Carte enregistrée : {sortie}")


def creer_carte_interactive(carte, sortie, titre):
    carte = carte.dropna(subset=["prix_m2"]).copy()
    vmin, vmax = carte["prix_m2"].quantile([0.02, 0.98])

    # Contours simplifiés (≈ 10 m) et arrondis pour alléger le fichier HTML
    carte["geometry"] = carte.geometry.simplify(10)
    carte = carte.to_crs("EPSG:4326")
    carte["geometry"] = shapely.set_precision(carte.geometry.array, 1e-5)
    carte = carte.set_index("code_iris")
    centre = carte.geometry.union_all().centroid

    fig = px.choropleth_map(
        carte,
        geojson=carte.geometry.__geo_interface__,
        locations=carte.index,
        color="prix_m2",
        range_color=(vmin, vmax),
        color_continuous_scale=BLEUS,
        hover_name="nom_iris",
        hover_data={
            "nom_commune": True,
            "prix_m2": ":,.0f",
            "prix_m2_appart": ":,.0f",
            "prix_m2_maison": ":,.0f",
            "nb_ventes": True,
        },
        labels={
            "nom_commune": "Commune",
            "prix_m2": "Prix médian/m² (€)",
            "prix_m2_appart": "Appartements (€/m²)",
            "prix_m2_maison": "Maisons (€/m²)",
            "nb_ventes": "Nombre de ventes",
        },
        zoom=9,
        center={"lat": centre.y, "lon": centre.x},
        map_style="carto-positron",
        opacity=0.8,
        title=titre,
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="white")
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
    fig.write_html(sortie, include_plotlyjs="cdn")
    print(f"Carte interactive enregistrée : {sortie}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dep", nargs="*", help="Départements à inclure (ex : 75 92). Par défaut : tous.")
    parser.add_argument("--annee-min", type=int, default=2024)
    parser.add_argument("--annee-max", type=int, default=2100)
    parser.add_argument(
        "--sortie", default=str(DOSSIER / "carte_prix_m2"),
        help="Chemin sans extension : produit <sortie>.png et <sortie>.html",
    )
    args = parser.parse_args()

    departements = args.dep or [f.stem.removeprefix("dvf_plus_d") for f in sorted(DOSSIER_DVF.glob("dvf_plus_d*.csv"))]

    ventes = charger_ventes(departements, args.annee_min, args.annee_max)
    iris = charger_iris(departements)
    carte = prix_par_iris(ventes, iris)

    annees = f"{ventes['anneemut'].min()}–{ventes['anneemut'].max()}"
    nb_iris = carte["prix_m2"].notna().sum()
    print(f"{len(ventes):_} ventes, {nb_iris:_} IRIS sur {len(iris):_} renseignés".replace("_", " "))
    titre = f"Prix médian au m² par IRIS — ventes {annees}"
    sortie = Path(args.sortie).with_suffix("")
    creer_carte(carte, sortie.with_suffix(".png"), titre)
    creer_carte_interactive(carte, sortie.with_suffix(".html"), titre)


if __name__ == "__main__":
    main()
