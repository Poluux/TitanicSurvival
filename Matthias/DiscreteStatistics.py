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
        moyenne = round(computeMean(datasetFile, col), 2)
        mediane = round(computeMedian(datasetFile, col), 2)
        mode = computeMode(datasetFile, col)
        mode_affiche = round(mode, 2) if isinstance(mode, (int, float)) else mode
        print(f"{noms[col]:<20} → Moyenne: {moyenne} | Médiane: {mediane} | Mode: {mode_affiche}")

def computeMean(datasetFile, column):
    return datasetFile[column].mean()

def computeMedian(datasetFile, column):
    return datasetFile[column].median()

def computeMode(datasetFile, column):
    mode_series = datasetFile[column].mode()
    return mode_series.iloc[0] if not mode_series.empty else "N/A"

datasetFile = pd.read_csv("../Project 2_Titanic-Dataset.csv")

displayStats(datasetFile)
