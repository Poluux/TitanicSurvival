import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

def displayAgeDistribution():
    df_age = df.dropna(subset=["Age"])
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_age, x="Age", bins=range(0, 85, 5), color="skyblue", edgecolor="black")
    plt.title("Age distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of passengers")
    plt.xticks(range(0, 85, 5))
    plt.tight_layout()
    plt.show()

displayAgeDistribution()

def displayDistribution(column):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=column, palette="pastel", order=df[column].value_counts().index)
    plt.title(f"{column} distribution")
    plt.xlabel(column)
    plt.ylabel("Number of passengers")
    plt.tight_layout()
    plt.show()

columns = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]
for column in columns:
    displayDistribution(column)

# We differatiate the display of the age distribution from the other distributions
# because we group passengers by slice of 5 years