import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This file display distribution bar diagrams for all variables
# Close a diagram will open the next one

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
    
    age_bins = pd.cut(df_age["Age"], bins=range(0, 90, 5), right=False)
    age_counts = age_bins.value_counts().sort_index()

    print("Nombre de passagers par tranche d'âge (tranches de 5 ans) :")
    for interval, count in age_counts.items():
        print(f"{interval}: {count}")

    plt.show()

def displayFareDistribution():
    df_fare = df.dropna(subset=["Fare"])
    max_fare = int(df_fare["Fare"].max()) + 25
    bins = range(0, max_fare, 25)
    counts, bin_edges, patches = plt.hist(df_fare["Fare"], bins=bins, color="skyblue", edgecolor="black")
    for count, edge in zip(counts, bin_edges[:-1]):
        if count > 0:
            plt.text(edge + 12.5, count + 1, str(int(count)), ha='center', va='bottom', fontsize=9)
    plt.title("Fare distribution (by £25 bins)")
    plt.xlabel("Fare (£)")
    plt.ylabel("Number of passengers")
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    plt.show()

def displayDistribution(column):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=column, palette="pastel", order=df[column].value_counts().index)
    plt.title(f"{column} distribution")
    plt.xlabel(column)
    plt.ylabel("Number of passengers")
    plt.tight_layout()
    plt.show()

def displayCheeseDistribution(column):
    counts = df[column].value_counts()
    labels = counts.index
    sizes = counts.values
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    plt.title(f"{column} distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


displayAgeDistribution()
displayFareDistribution()
# We differatiate the display of the age / fare distribution from the other distributions
# because we group passengers by slice of 5 years

columns = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]
for column in columns:
    displayDistribution(column)

    # Uncomment the next line if you want to see the cheese distributions as well.
    # displayCheeseDistribution(column)

