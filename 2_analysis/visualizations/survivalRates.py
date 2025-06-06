import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This file creates visualizations of the survival rate
# based on different parameters.

# Load the dataset
df = pd.read_csv("../../Project 2_Titanic-Dataset.csv")

def plotSurvivalBySexWithPercent():
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="Sex", hue="Survived", palette="pastel")

    # Calculate total number by sex
    total_by_sex = df["Sex"].value_counts()

    # Display percentages on top of bars
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            sex = bar.get_x() + bar.get_width() / 2
            label = bar.get_label()
            index = int(bar.get_x() + 0.5)
            # Get the corresponding sex label
            sex_label = bar.get_label()
            # Get the category from the x-axis
            sex_index = int(bar.get_x() + bar.get_width() / 2)
            sex_value = ax.get_xticks()
            # Get the category name from the axis
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

# 2. Graph: age distribution by 5-year groups, side-by-side bars
def plot_age_groups_by_survival(df):
    df_age = df.dropna(subset=["Age"])
    # Create 5-year age groups
    bins = range(0, int(df_age["Age"].max()) + 5, 5)
    df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=bins, right=False)

    # Group and count
    grouped = df_age.groupby(["AgeGroup", "Survived"]).size().unstack(fill_value=0)

    # Plot
    grouped.plot(kind="bar", figsize=(12, 6), color=["salmon", "skyblue"])
    plt.title("Survival by Age Group (5-year intervals)")
    plt.xlabel("Age Group")
    plt.ylabel("Number of passengers")
    plt.legend(title="Survived", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()

# Clean up the Cabin data
df['Cabin'] = df['Cabin'].fillna('Unknown')  # Replace missing values with 'Unknown'
df['Cabin'] = df['Cabin'].str.extract('([A-Za-z])')  # Keep only the letter (section indicator)

def plot_survival_rate_by_age_group(df):
    df_age = df.dropna(subset=["Age", "Survived"])

    # Create age bins (5-year groups)
    bins = range(0, int(df_age["Age"].max()) + 5, 5)
    df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=bins, right=False)

    # Calculate survival rate (%)
    survival_rate = df_age.groupby("AgeGroup")["Survived"].mean() * 100

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=survival_rate.index.astype(str), y=survival_rate.values, color="mediumseagreen")

    # Add annotations
    for i, rate in enumerate(survival_rate.values):
        plt.text(i, rate + 1, f"{rate:.1f}%", ha='center')

    plt.title("Survival rate (%) by range of 5 years")
    plt.xlabel("Age range")
    plt.ylabel("Survival rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Graph: survival by cabin section
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

    # Create £25 intervals
    max_fare = int(df_fare["Fare"].max()) + 25
    bins = range(0, max_fare, 25)
    df_fare["FareGroup"] = pd.cut(df_fare["Fare"], bins=bins, right=False)

    # Group data
    grouped = df_fare.groupby(["FareGroup", "Survived"]).size().unstack(fill_value=0)

    # Plot
    ax = grouped.plot(kind="bar", figsize=(12, 6), color=["salmon", "skyblue"], edgecolor="black")

    plt.title("Survival rate by fare range (£25)")
    plt.xlabel("Fare range (Fare)")
    plt.ylabel("Number of passengers")
    plt.xticks(rotation=45)
    plt.legend(title="Survived", labels=["Yes", "No"])
    plt.tight_layout()

    # Add annotations above bars
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    plt.show()

def plot_survival_rate_by_feature(df, feature):
    # Group and calculate survival rate per value
    grouped = df.groupby(feature)['Survived'].mean() * 100  # Percent
    counts = df[feature].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.index, y=grouped.values, palette="coolwarm")

    # Annotate bars with percentage
    for i, value in enumerate(grouped.values):
        plt.text(i, value + 1, f"{value:.1f}%", ha='center')

    plt.title(f"Survival rate by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Survival rate (%)")
    plt.tight_layout()
    plt.show()

# Display the plots

plotSurvivalBySexWithPercent()
plot_age_groups_by_survival(df)
plot_survival_rate_by_age_group(df)
plot_survival_by_cabin(df)
plot_survival_counts_by_fare(df)
plot_survival_rate_by_feature(df, "SibSp")
plot_survival_rate_by_feature(df, "Parch")
