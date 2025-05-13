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

# Survival rate by Fare price
def plot_survival_counts_by_fare(df):
    df_fare = df.dropna(subset=["Fare", "Survived"])

    # Créer des tranches de 25£
    max_fare = int(df_fare["Fare"].max()) + 25
    bins = range(0, max_fare, 25)
    df_fare["FareGroup"] = pd.cut(df_fare["Fare"], bins=bins, right=False)

    # Regroupement des données
    grouped = df_fare.groupby(["FareGroup", "Survived"]).size().unstack(fill_value=0)

    # Tracé
    ax = grouped.plot(kind="bar", figsize=(12, 6), color=["salmon", "skyblue"], edgecolor="black")

    plt.title("Survivants et morts par tranche de prix payé (£25)")
    plt.xlabel("Tranche de prix (Fare)")
    plt.ylabel("Nombre de passagers")
    plt.xticks(rotation=45)
    plt.legend(title="Survived", labels=["Non", "Oui"])
    plt.tight_layout()

    # Ajout des annotations au-dessus de chaque barre
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # décalage vertical
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    plt.show()

def plot_survival_rate_by_feature(df, feature):
    # Grouper et calculer le taux de survie pour chaque valeur
    grouped = df.groupby(feature)['Survived'].mean() * 100  # Pourcentage
    counts = df[feature].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.index, y=grouped.values, palette="coolwarm")

    # Annoter les barres avec le pourcentage
    for i, value in enumerate(grouped.values):
        plt.text(i, value + 1, f"{value:.1f}%", ha='center')

    plt.title(f"Taux de survie en fonction de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Taux de survie (%)")
    plt.tight_layout()
    plt.show()

# Affichage des graphiques

plotSurvivalBySexWithPercent()
plot_age_groups_by_survival(df)
plot_survival_by_cabin(df)
plot_survival_counts_by_fare(df)
plot_survival_rate_by_feature(df, "SibSp")
plot_survival_rate_by_feature(df, "Parch")



