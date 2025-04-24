import pandas as pd
from tabulate import tabulate

df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

def displayContingencyTable(column):
    table = pd.crosstab(df[column], df["Survived"])
    print("Contingency table (raw counts):")
    print(tabulate(table, headers='keys', tablefmt='grid'))
    print()

def displayContingencyTableInPercentage(column):
    table = pd.crosstab(df[column], df["Survived"], normalize='index') * 100
    print("Contingency table (percentages):")
    print(tabulate(table.round(2), headers='keys', tablefmt='grid'))
    print()

def displayContingency(column):
    print(f"=== Tables for '{column}' ===\n")
    displayContingencyTable(column)
    displayContingencyTableInPercentage(column)
    print("=" * 40 + "\n")

displayContingency("Sex")
displayContingency("Pclass")
displayContingency("Embarked")
