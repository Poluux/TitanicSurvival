import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

def plotSurvivalBySexWithPercent():
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="Sex", hue="Survived", palette="pastel")

    # Calcul des totaux par sexe
    total_by_sex = df["Sex"].value_counts()

    # Affichage des pourcentages sur les barres
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            sex = bar.get_x() + bar.get_width() / 2
            label = bar.get_label()
            index = int(bar.get_x() + 0.5)
            # Trouver le sexe correspondant
            sex_label = bar.get_label()
            # Obtenir le sexe depuis l'axe x
            sex_index = int(bar.get_x() + bar.get_width() / 2)
            sex_value = ax.get_xticks()
            # Obtenir le nom de la catégorie depuis l'axe
            category = ax.get_xticklabels()[int(bar.get_x() + 0.5)].get_text()
            total = total_by_sex[category]
            percent = height / total * 100
            ax.annotate(f'{percent:.1f}%', 
                        (bar.get_x() + bar.get_width() / 2, height), 
                        ha='center', va='bottom', fontsize=9)

    plt.title("Survival count by sex (with percentages)")
    plt.xlabel("Sex")
    plt.ylabel("Number of passengers")
    plt.legend(title="Survived", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()

   # 2. Graphique : distribution des âges par groupes de 5 ans, barres côte à côte
def plot_age_groups_by_survival(df):
    df_age = df.dropna(subset=["Age"])
    # Création de groupes d’âges de 5 ans
    bins = range(0, int(df_age["Age"].max()) + 5, 5)
    df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=bins, right=False)

    # Regroupement et comptage
    grouped = df_age.groupby(["AgeGroup", "Survived"]).size().unstack(fill_value=0)

    # Tracé
    grouped.plot(kind="bar", figsize=(12, 6), color=["salmon", "skyblue"])
    plt.title("Survival by Age Group (5-year intervals)")
    plt.xlabel("Age Group")
    plt.ylabel("Number of passengers")
    plt.legend(title="Survived", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()

    # Nettoyage des données de cabines
df['Cabin'] = df['Cabin'].fillna('Unknown')  # Remplacer les valeurs manquantes par 'Unknown'
df['Cabin'] = df['Cabin'].str.extract('([A-Za-z])')  # Garder seulement la lettre (indicateur de section)

# Graphique : survie par section de cabine
def plot_survival_by_cabin(df):
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x="Cabin", hue="Survived", palette="pastel")
    plt.title("Survival by Cabin Section")
    plt.xlabel("Cabin Section")
    plt.ylabel("Number of Passengers")
    plt.legend(title="Survived", labels=["No", "Yes"])
    plt.tight_layout()

# Affichage des graphiques

plotSurvivalBySexWithPercent()
plot_age_groups_by_survival(df)
plot_survival_by_cabin(df)
