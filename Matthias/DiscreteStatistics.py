import pandas as pd

def displayStats(datasetFile):
    print("Statistiques :\n")

    colonnes = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    noms = {
        "Survived": "Survival rate",
        "Pclass": "Ticket class",
        "Age": "Age",
        "SibSp": "Siblings/Spouses",
        "Parch": "Parents/Children",
        "Fare": "Fare"
    }

    for col in colonnes:
        data = datasetFile[col].dropna()
        moyenne = round(computeMean(data), 2)
        mediane = round(computeMedian(data), 2)
        mode = computeMode(data)
        mode_affiche = round(mode, 2) if isinstance(mode, (int, float)) else mode
        etendue = round(computeRange(data), 2)
        variance = round(computeVariance(data), 2)
        std_dev = round(computeStdDev(data), 2)

        print(f"{noms[col]:<20} → Moyenne : {moyenne} | Médiane : {mediane} | Mode : {mode_affiche} | "
              f"Range : {etendue} | Variance : {variance} | Std Dev : {std_dev}")

def computeMean(data):
    return data.mean()

def computeMedian(data):
    return data.median()

def computeMode(data):
    mode_series = data.mode()
    return mode_series.iloc[0] if not mode_series.empty else "N/A"

def computeRange(data):
    return data.max() - data.min()

def computeVariance(data):
    return data.var()

def computeStdDev(data):
    return data.std()

# Chargement des données
datasetFile = pd.read_csv("../Project 2_Titanic-Dataset.csv")

# Affichage des statistiques
displayStats(datasetFile)