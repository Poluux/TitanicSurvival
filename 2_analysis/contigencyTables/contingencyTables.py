import pandas as pd
from tabulate import tabulate

# This file creates contingency tables for some interesting pairs of fields
# We can see the result both in absolute value and in percentage.

df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

def displayContingencyTable(column1, column2):
    table = pd.crosstab(df[column1], df[column2])
    print("Contingency table (raw counts):")
    print(tabulate(table, headers='keys', tablefmt='grid'))
    print()

def displayContingencyTableInPercentage(column1, column2):
    table = pd.crosstab(df[column1], df[column2], normalize='index') * 100
    print("Contingency table (percentages):")
    print(tabulate(table.round(2), headers='keys', tablefmt='grid'))
    print()

def displayContingency(column1, column2):
    print(f"=== Tables for '{column1}' / '{column2}' ===\n")
    displayContingencyTable(column1, column2)
    displayContingencyTableInPercentage(column1, column2)
    print("=" * 40 + "\n")

displayContingency("Sex", "Survived")
displayContingency("Pclass", "Survived")
displayContingency("Embarked", "Survived")
displayContingency("Embarked", "Pclass")
